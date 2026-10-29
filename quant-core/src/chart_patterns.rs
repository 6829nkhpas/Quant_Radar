// quant/chart_patterns.rs — Phase 9.2 Institutional-Grade Chart Pattern Detection Engine.
//
// Detects 20+ structural chart patterns across any timeframe with Volume Validation:
//   1. Identifying swing highs/lows (local extrema) using a rolling window.
//   2. Building a cleaned structural skeleton (alternating Peak → Trough → Peak).
//   3. Fitting support/resistance trendlines via linear regression.
//   4. Evaluating geometric rules + volume confirmation against pattern archetypes.
//
// Volume Validation Engine:
//   • Reversal Exhaustion: v_final < v_first  (Reversals)
//   • Consolidation Drying: volume_slope < 0.0  (Continuations)
//   • Breakout Volume Boost: curr_vol > 1.2 × SMA-20(Volume)  (All breakouts)
//
// Pattern Categories:
//   Harmonic (5):      Gartley, Bat, Butterfly, Crab, Shark
//   Reversal (8):      Head & Shoulders Top, Inverse H&S, Double Top/Bottom,
//                      Triple Top/Bottom, Rising Wedge, Falling Wedge
//   Institutional (3): Quasimodo (Bull/Bear), Three Drives
//   Continuation (6):  Bullish/Bearish Flag, Bullish/Bearish Pennant,
//                      Cup and Handle, Inverse Cup and Handle
//   Bilateral (4):     Symmetrical Triangle, Ascending Triangle,
//                      Descending Triangle, Rectangle

use super::patterns::Candle;

// ── Chart Pattern Data Structures ──────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChartPattern {
    pub pattern_type: String,
    pub sentiment: String, // "Bullish", "Bearish", "Neutral"
    pub confidence: f64,   // 0.0 to 1.0
    pub start_idx: usize,
    pub end_idx: usize,
    pub description: String,
    // Phase 9.2 fields:
    pub structural_bias: String,
    pub geometric_strictness: f64,
    pub volume_validation: String,
    pub breakout_status: String,
    // Phase 10: Forming pattern fields
    #[serde(default)]
    pub is_forming: bool,
    /// 0.0 to 1.0 — how close a FORMING pattern is to completion. Not serialised at
    /// 0.0, which is what every completed detection carries (each forming detector
    /// sets 0.50 or more): `get_chart_patterns` served `formation_progress: 0.0` on
    /// completed patterns and the model read them as "0% formed" (2026-10-16,
    /// XAUTUSDT). Absent deserialises back to 0.0, which is what the panel defaults to.
    #[serde(default, skip_serializing_if = "progress_unset")]
    pub formation_progress: f64,
    /// Task 11 levels, flattened into the pattern's JSON. Only `analyze`'s detectors
    /// set them; every forming detector leaves them empty, so `analyze_forming` output
    /// (the multi-TF panel, the pattern context) serializes exactly as before.
    #[serde(flatten)]
    pub levels: PatternLevels,
}

fn progress_unset(progress: &f64) -> bool {
    *progress == 0.0
}

/// Where a completed pattern triggers, where its measured move points, where it is
/// proven wrong, and what the closed bars since it ended say (Task 11). Every field is
/// absent when the detector's own geometry does not define it — never a guess.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PatternLevels {
    /// The neckline / boundary a close must cross: its value at the breakout bar once
    /// a close has crossed it, else at the newest bar (where it would break now).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub breakout_level: Option<f64>,
    /// breakout_level ± the pattern's height (measured move); absent when not positive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<f64>,
    /// The price whose close says the pattern is wrong (its opposite extreme).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invalidation: Option<f64>,
    /// `pending` (no breakout close yet) / `confirmed` / `failed` / `target_hit`; see
    /// `ChartPatternEngine::measure`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

/// A swing point in the price series: either a local high (Peak) or low (Trough).
///
/// `pub(crate)` so the market-structure engine (`structure.rs`) builds on the SAME
/// swing detector rather than a second copy that could drift from this one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SwingPoint {
    pub(crate) idx: usize,
    pub(crate) price: f64,
    pub(crate) kind: SwingKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SwingKind {
    Peak,
    Trough,
}

// ── Configuration Constants ────────────────────────────────────────────────

const SWING_WINDOW: usize = 5;
const MATCH_TOLERANCE: f64 = 0.015;
const SHOULDER_TOLERANCE: f64 = 0.08;
const MIN_FLAGPOLE_CANDLES: usize = 5;
const FLAGPOLE_MIN_RANGE_RATIO: f64 = 0.015;
const FLAT_SLOPE_THRESHOLD: f64 = 0.0005;
const CUP_ASYMMETRY_TOLERANCE: f64 = 0.05;
const FIB_TOLERANCE: f64 = 0.05;

// ── Chart Pattern Engine ───────────────────────────────────────────────────

pub struct ChartPatternEngine;

impl ChartPatternEngine {
    /// Analyze a slice of candles and return all detected chart patterns
    /// with volume validation.
    pub fn analyze(candles: &[Candle]) -> Vec<ChartPattern> {
        if candles.len() < 20 {
            return Vec::new();
        }

        let raw_swings = Self::find_swings(candles);
        if raw_swings.len() < 3 {
            return Vec::new();
        }

        let swings = Self::alternate_swings(&raw_swings);
        if swings.len() < 3 {
            return Vec::new();
        }

        let mut patterns: Vec<ChartPattern> = Vec::new();

        // Phase 9.2 detection modules
        Self::detect_harmonics(candles, &swings, &mut patterns);
        Self::detect_head_and_shoulders(candles, &swings, &mut patterns);
        Self::detect_inverse_head_and_shoulders(candles, &swings, &mut patterns);
        Self::detect_double_top(candles, &swings, &mut patterns);
        Self::detect_double_bottom(candles, &swings, &mut patterns);
        Self::detect_triple_top(candles, &swings, &mut patterns);
        Self::detect_triple_bottom(candles, &swings, &mut patterns);
        Self::detect_quasimodo(candles, &swings, &mut patterns);
        Self::detect_three_drives(candles, &swings, &mut patterns);
        Self::detect_rising_wedge(candles, &swings, &mut patterns);
        Self::detect_falling_wedge(candles, &swings, &mut patterns);
        Self::detect_flags_and_pennants(candles, &swings, &mut patterns);
        Self::detect_cup_and_handle(candles, &swings, &mut patterns);
        Self::detect_inverse_cup_and_handle(candles, &swings, &mut patterns);
        Self::detect_triangles(candles, &swings, &mut patterns);
        Self::detect_rectangle(candles, &swings, &mut patterns);

        // Sort by confidence descending
        patterns.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        patterns
    }

    // ── Swing Detection ────────────────────────────────────────────────────

    /// The symmetric 5-bar fractal every pattern detector here was tuned on. The loop
    /// itself lives in `structure::find_swings_with` (parameterised there for the
    /// market-structure engine); with 5/5 it is the exact loop this used to inline, and
    /// `structure`'s tests pin that equality against a verbatim copy of the old body.
    fn find_swings(candles: &[Candle]) -> Vec<SwingPoint> {
        crate::structure::find_swings_with(candles, SWING_WINDOW, SWING_WINDOW)
    }

    pub(crate) fn alternate_swings(raw: &[SwingPoint]) -> Vec<SwingPoint> {
        if raw.is_empty() { return Vec::new(); }
        let mut result: Vec<SwingPoint> = vec![raw[0]];
        for sp in &raw[1..] {
            let last = result.last().unwrap();
            if sp.kind == last.kind {
                let better = match sp.kind {
                    SwingKind::Peak => sp.price > last.price,
                    SwingKind::Trough => sp.price < last.price,
                };
                if better { *result.last_mut().unwrap() = *sp; }
            } else {
                result.push(*sp);
            }
        }
        result
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    #[inline]
    fn prices_match(a: f64, b: f64, tolerance: f64) -> bool {
        let avg = (a + b) / 2.0;
        if avg.abs() < 1e-9 { return (a - b).abs() < 1e-9; }
        ((a - b).abs() / avg) <= tolerance
    }

    #[inline]
    fn matches_fib(ratio: f64, target: f64) -> bool {
        (ratio - target).abs() <= FIB_TOLERANCE
    }

    fn linear_regression(points: &[(f64, f64)]) -> (f64, f64) {
        let n = points.len() as f64;
        if n < 2.0 { return (0.0, points.first().map(|p| p.1).unwrap_or(0.0)); }
        let sum_x: f64 = points.iter().map(|p| p.0).sum();
        let sum_y: f64 = points.iter().map(|p| p.1).sum();
        let sum_xy: f64 = points.iter().map(|p| p.0 * p.1).sum();
        let sum_xx: f64 = points.iter().map(|p| p.0 * p.0).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 { return (0.0, sum_y / n); }
        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;
        (slope, intercept)
    }

    fn get_volume_sma(candles: &[Candle], idx: usize, period: usize) -> f64 {
        if idx < period || candles.len() < period { return 0.0; }
        let start = idx - period;
        let sum: f64 = candles[start..idx].iter().map(|c| c.volume).sum();
        sum / period as f64
    }

    fn get_volume_slope(candles: &[Candle], start_idx: usize, end_idx: usize) -> f64 {
        if start_idx >= end_idx || end_idx >= candles.len() { return 0.0; }
        let points: Vec<(f64, f64)> = (start_idx..=end_idx)
            .enumerate()
            .map(|(i, idx)| (i as f64, candles[idx].volume))
            .collect();
        let (slope, _) = Self::linear_regression(&points);
        slope
    }

    /// The levels of one directional pattern (Task 11). `line(k)` is the breakout
    /// boundary at bar k (a constant for a horizontal neckline), `height` the measured
    /// move, `invalidation` the price whose close says the pattern is wrong, `from` the
    /// first bar after the pattern's last swing. Walks the CLOSES of bars from..newest:
    ///   - a close beyond `invalidation` (below it for a bullish pattern) -> failed, stop;
    ///   - else the first close beyond line(k) (above it for bullish) -> confirmed at k0;
    ///   - once confirmed, a close at or beyond line(k0) ± height -> target_hit, stop;
    ///   - none of these -> pending.
    /// Every field is None for degenerate geometry: a non-finite level, height or
    /// invalidation, height <= 0, or the invalidation not beyond the breakout level on
    /// the far side.
    /// ponytail: the newest bar is never read — quant-core cannot tell a forming bar
    /// from a closed one (tool-server passes the live bar), so a status is at most one
    /// closed bar late; closes only, so a wick through the target is not a hit.
    fn measure(
        candles: &[Candle],
        from: usize,
        bullish: bool,
        line: impl Fn(usize) -> f64,
        height: f64,
        invalidation: f64,
    ) -> PatternLevels {
        let newest = candles.len().saturating_sub(1);
        let beyond = |a: f64, b: f64| if bullish { a > b } else { a < b };
        let step = if bullish { height } else { -height };
        let mut broke: Option<usize> = None;
        let mut status = "pending";
        for k in from..newest {
            let c = candles[k].close;
            if beyond(invalidation, c) {
                status = "failed";
                break;
            }
            if broke.is_none() && beyond(c, line(k)) {
                broke = Some(k);
                status = "confirmed";
            }
            if let Some(b) = broke {
                if !beyond(line(b) + step, c) {
                    status = "target_hit";
                    break;
                }
            }
        }
        let level = line(broke.unwrap_or(newest));
        if !(level.is_finite()
            && height.is_finite()
            && height > 0.0
            && invalidation.is_finite()
            && beyond(level, invalidation))
        {
            return PatternLevels::default();
        }
        let target = level + step;
        PatternLevels {
            breakout_level: Some(level),
            target: (target > 0.0).then_some(target),
            invalidation: Some(invalidation),
            status: Some(status.to_string()),
        }
    }

    /// A bilateral pattern (Symmetrical Triangle, Rectangle) has no direction until a
    /// closed bar breaks one side: the first close in from..newest above upper(k) makes
    /// it bullish (invalidated below `last_low`), below lower(k) bearish (above
    /// `last_high`), and `measure` takes it from there. No break -> status pending with
    /// no levels: there are two candidate levels and neither is the breakout yet.
    fn measure_bilateral(
        candles: &[Candle],
        from: usize,
        upper: impl Fn(usize) -> f64,
        lower: impl Fn(usize) -> f64,
        height: f64,
        last_low: f64,
        last_high: f64,
    ) -> PatternLevels {
        let newest = candles.len().saturating_sub(1);
        let side = (from..newest).find_map(|k| {
            let c = candles[k].close;
            if c > upper(k) {
                Some(true)
            } else if c < lower(k) {
                Some(false)
            } else {
                None
            }
        });
        match side {
            Some(true) => Self::measure(candles, from, true, upper, height, last_low),
            Some(false) => Self::measure(candles, from, false, lower, height, last_high),
            None => PatternLevels { status: Some("pending".to_string()), ..Default::default() },
        }
    }

    // ── 1. Harmonic Pattern Detection ──────────────────────────────────────

    fn detect_harmonics(
        candles: &[Candle],
        swings: &[SwingPoint],
        out: &mut Vec<ChartPattern>,
    ) {
        let n = swings.len();
        if n < 5 { return; }

        // Confirmed: 5-point structure
        let x = swings[n - 5];
        let a = swings[n - 4];
        let b = swings[n - 3];
        let c = swings[n - 2];
        let d = swings[n - 1];

        if x.kind == a.kind || a.kind == b.kind || b.kind == c.kind || c.kind == d.kind {
            return;
        }

        let is_bullish = x.kind == SwingKind::Trough;
        let xa = (a.price - x.price).abs();
        let ab = (b.price - a.price).abs();
        if xa < 1e-9 || ab < 1e-9 { return; }

        let ab_xa = ab / xa;
        let ad_xa = (d.price - a.price).abs() / xa;

        let classification = Self::classify_harmonic(ab_xa, ad_xa);
        if let Some((name, target_d)) = classification {
            // Volume: Reversal Exhaustion at D vs X
            if x.idx < candles.len() && d.idx < candles.len() {
                let v_x = candles[x.idx].volume;
                let v_d = candles[d.idx].volume;
                if v_d < v_x {
                    let bias = if is_bullish { "Bullish Reversal" } else { "Bearish Reversal" };
                    out.push(ChartPattern {
                        pattern_type: format!("Harmonic {}", name),
                        sentiment: if is_bullish { "Bullish".to_string() } else { "Bearish".to_string() },
                        confidence: 0.90,
                        start_idx: x.idx,
                        end_idx: d.idx,
                        description: format!(
                            "Harmonic {}: X→A→B→C→D confirmed. D at {:.4} retracement of XA. {}.",
                            name, target_d, bias
                        ),
                        structural_bias: bias.to_string(),
                        geometric_strictness: 0.95,
                        volume_validation: "Confirmed: Reversal Exhaustion".to_string(),
                        breakout_status: "Confirmed".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                    // A reversal at point D has no breakout line or measured-move
                    // convention: only where it is wrong, beyond the X / D extreme.
                    levels: PatternLevels {
                        invalidation: Some(if is_bullish { x.price.min(d.price) } else { x.price.max(d.price) }),
                        ..Default::default()
                    },
                    });
                }
            }
        }
    }

    fn classify_harmonic(ab_xa: f64, ad_xa: f64) -> Option<(&'static str, f64)> {
        if Self::matches_fib(ab_xa, 0.618) && Self::matches_fib(ad_xa, 0.786) {
            return Some(("Gartley", 0.786));
        }
        if (Self::matches_fib(ab_xa, 0.382) || Self::matches_fib(ab_xa, 0.50)) && Self::matches_fib(ad_xa, 0.886) {
            return Some(("Bat", 0.886));
        }
        if Self::matches_fib(ab_xa, 0.786) && Self::matches_fib(ad_xa, 1.272) {
            return Some(("Butterfly", 1.272));
        }
        if ab_xa >= 0.33 && ab_xa <= 0.66 && Self::matches_fib(ad_xa, 1.618) {
            return Some(("Crab", 1.618));
        }
        if ab_xa >= 1.08 && ab_xa <= 1.66 && Self::matches_fib(ad_xa, 0.886) {
            return Some(("Shark", 0.886));
        }
        None
    }

    // ── 2. Head & Shoulders Top ────────────────────────────────────────────

    fn detect_head_and_shoulders(
        candles: &[Candle],
        swings: &[SwingPoint],
        out: &mut Vec<ChartPattern>,
    ) {
        let peaks: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Peak).collect();
        if peaks.len() < 3 { return; }

        for window in peaks.windows(3) {
            let (left, head, right) = (window[0], window[1], window[2]);
            if head.price <= left.price || head.price <= right.price { continue; }
            if !Self::prices_match(left.price, right.price, SHOULDER_TOLERANCE) { continue; }

            // Volume: Reversal Exhaustion — RS volume < LS volume
            if left.idx < candles.len() && right.idx < candles.len() {
                let v_ls = candles[left.idx].volume;
                let v_rs = candles[right.idx].volume;
                if v_rs >= v_ls { continue; } // Failed volume filter

                // Neckline calculation
                let troughs_between: Vec<&SwingPoint> = swings.iter()
                    .filter(|s| s.kind == SwingKind::Trough && s.idx > left.idx && s.idx < right.idx)
                    .collect();
                
                let current_price = candles.last().map(|c| c.close).unwrap_or(0.0);
                let current_idx = candles.len().saturating_sub(1);

                let (breakout_status, volume_val) = if troughs_between.len() >= 2 {
                    let t1 = troughs_between[0];
                    let t2 = troughs_between[troughs_between.len() - 1];
                    let neckline_slope = if t2.idx != t1.idx {
                        (t2.price - t1.price) / (t2.idx as f64 - t1.idx as f64)
                    } else { 0.0 };
                    let neckline_val = t1.price + neckline_slope * (current_idx as f64 - t1.idx as f64);
                    let is_breakout = current_price < neckline_val;

                    if is_breakout {
                        let sma = Self::get_volume_sma(candles, current_idx, 20);
                        let curr_vol = candles.last().map(|c| c.volume).unwrap_or(0.0);
                        if curr_vol > 1.2 * sma && sma > 0.0 {
                            ("Confirmed Breakout".to_string(), "Confirmed: Breakout Volume Boost".to_string())
                        } else {
                            ("Pending Neckline Test".to_string(), "Confirmed: Reversal Exhaustion".to_string())
                        }
                    } else {
                        ("Pending Neckline Test".to_string(), "Confirmed: Reversal Exhaustion".to_string())
                    }
                } else {
                    ("Pending Neckline Test".to_string(), "Confirmed: Reversal Exhaustion".to_string())
                };

                let shoulder_avg = (left.price + right.price) / 2.0;
                let head_prominence = (head.price - shoulder_avg) / shoulder_avg;
                let shoulder_symmetry = 1.0 - ((left.price - right.price).abs() / shoulder_avg);
                let confidence = (0.5 + head_prominence.min(0.3) + shoulder_symmetry * 0.2).min(1.0);

                // Task 11: the neckline the breakout check above draws (first and last
                // trough between the shoulders), the head's height above it, and the head
                // as the invalidation. No neckline -> no levels.
                let levels = if troughs_between.len() >= 2 {
                    let (t1, t2) = (troughs_between[0], troughs_between[troughs_between.len() - 1]);
                    let slope = if t2.idx != t1.idx {
                        (t2.price - t1.price) / (t2.idx as f64 - t1.idx as f64)
                    } else { 0.0 };
                    let line = move |k: usize| t1.price + slope * (k as f64 - t1.idx as f64);
                    let height = head.price - line(head.idx);
                    Self::measure(candles, right.idx + 1, false, line, height, head.price)
                } else {
                    PatternLevels::default()
                };

                out.push(ChartPattern {
                    pattern_type: "Head & Shoulders Top".to_string(),
                    sentiment: "Bearish".to_string(),
                    confidence,
                    start_idx: left.idx,
                    end_idx: right.idx,
                    description: format!(
                        "H&S Top: LS {:.2}, Head {:.2}, RS {:.2}. Volume exhaustion confirmed. {}.",
                        left.price, head.price, right.price, breakout_status
                    ),
                    structural_bias: "Bearish Reversal".to_string(),
                    geometric_strictness: shoulder_symmetry,
                    volume_validation: volume_val,
                    breakout_status,
                    is_forming: false,
                    formation_progress: 0.0,
                    levels,
                });
            }
        }
    }

    // ── 3. Inverse Head & Shoulders ────────────────────────────────────────

    fn detect_inverse_head_and_shoulders(
        candles: &[Candle],
        swings: &[SwingPoint],
        out: &mut Vec<ChartPattern>,
    ) {
        let troughs: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Trough).collect();
        if troughs.len() < 3 { return; }

        for window in troughs.windows(3) {
            let (left, head, right) = (window[0], window[1], window[2]);
            if head.price >= left.price || head.price >= right.price { continue; }
            if !Self::prices_match(left.price, right.price, SHOULDER_TOLERANCE) { continue; }

            if left.idx < candles.len() && right.idx < candles.len() {
                let v_ls = candles[left.idx].volume;
                let v_rs = candles[right.idx].volume;
                if v_rs >= v_ls { continue; }

                let peaks_between: Vec<&SwingPoint> = swings.iter()
                    .filter(|s| s.kind == SwingKind::Peak && s.idx > left.idx && s.idx < right.idx)
                    .collect();

                let current_price = candles.last().map(|c| c.close).unwrap_or(0.0);
                let current_idx = candles.len().saturating_sub(1);

                let (breakout_status, volume_val) = if peaks_between.len() >= 2 {
                    let p1 = peaks_between[0];
                    let p2 = peaks_between[peaks_between.len() - 1];
                    let neckline_slope = if p2.idx != p1.idx {
                        (p2.price - p1.price) / (p2.idx as f64 - p1.idx as f64)
                    } else { 0.0 };
                    let neckline_val = p1.price + neckline_slope * (current_idx as f64 - p1.idx as f64);
                    let is_breakout = current_price > neckline_val;

                    if is_breakout {
                        let sma = Self::get_volume_sma(candles, current_idx, 20);
                        let curr_vol = candles.last().map(|c| c.volume).unwrap_or(0.0);
                        if curr_vol > 1.2 * sma && sma > 0.0 {
                            ("Confirmed Breakout".to_string(), "Confirmed: Breakout Volume Boost".to_string())
                        } else {
                            ("Pending Neckline Test".to_string(), "Confirmed: Reversal Exhaustion".to_string())
                        }
                    } else {
                        ("Pending Neckline Test".to_string(), "Confirmed: Reversal Exhaustion".to_string())
                    }
                } else {
                    ("Pending Neckline Test".to_string(), "Confirmed: Reversal Exhaustion".to_string())
                };

                let shoulder_avg = (left.price + right.price) / 2.0;
                let head_depth = (shoulder_avg - head.price) / shoulder_avg;
                let shoulder_symmetry = 1.0 - ((left.price - right.price).abs() / shoulder_avg);
                let confidence = (0.5 + head_depth.min(0.3) + shoulder_symmetry * 0.2).min(1.0);

                // Task 11: the mirror of the H&S Top's levels — the neckline through the
                // first and last peak between the shoulders, the head below it.
                let levels = if peaks_between.len() >= 2 {
                    let (p1, p2) = (peaks_between[0], peaks_between[peaks_between.len() - 1]);
                    let slope = if p2.idx != p1.idx {
                        (p2.price - p1.price) / (p2.idx as f64 - p1.idx as f64)
                    } else { 0.0 };
                    let line = move |k: usize| p1.price + slope * (k as f64 - p1.idx as f64);
                    let height = line(head.idx) - head.price;
                    Self::measure(candles, right.idx + 1, true, line, height, head.price)
                } else {
                    PatternLevels::default()
                };

                out.push(ChartPattern {
                    pattern_type: "Inverse Head & Shoulders".to_string(),
                    sentiment: "Bullish".to_string(),
                    confidence,
                    start_idx: left.idx,
                    end_idx: right.idx,
                    description: format!(
                        "IH&S: LS {:.2}, Head {:.2}, RS {:.2}. Volume exhaustion confirmed. {}.",
                        left.price, head.price, right.price, breakout_status
                    ),
                    structural_bias: "Bullish Reversal".to_string(),
                    geometric_strictness: shoulder_symmetry,
                    volume_validation: volume_val,
                    breakout_status,
                    is_forming: false,
                    formation_progress: 0.0,
                    levels,
                });
            }
        }
    }

    // ── 4. Double Top ──────────────────────────────────────────────────────

    fn detect_double_top(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let peaks: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Peak).collect();
        if peaks.len() < 2 { return; }

        for window in peaks.windows(2) {
            let (p1, p2) = (window[0], window[1]);
            if !Self::prices_match(p1.price, p2.price, MATCH_TOLERANCE) { continue; }

            let trough_between = swings.iter()
                .find(|s| s.kind == SwingKind::Trough && s.idx > p1.idx && s.idx < p2.idx);
            if trough_between.is_none() { continue; }

            // Volume: Reversal Exhaustion
            if p1.idx < candles.len() && p2.idx < candles.len() {
                let v_p1 = candles[p1.idx].volume;
                let v_p2 = candles[p2.idx].volume;
                if v_p2 >= v_p1 { continue; }

                let avg_peak = (p1.price + p2.price) / 2.0;
                let trough = trough_between.unwrap();
                let depth = (avg_peak - trough.price) / avg_peak;
                if depth < 0.005 { continue; }

                let current_price = candles.last().map(|c| c.close).unwrap_or(0.0);
                let is_breakout = current_price < trough.price;
                let current_idx = candles.len().saturating_sub(1);

                let (breakout_status, volume_val) = if is_breakout {
                    let sma = Self::get_volume_sma(candles, current_idx, 20);
                    let curr_vol = candles.last().map(|c| c.volume).unwrap_or(0.0);
                    if curr_vol > 1.2 * sma && sma > 0.0 {
                        ("Confirmed Breakout".to_string(), "Confirmed: Breakout Volume Boost".to_string())
                    } else {
                        ("Pending Neckline Test".to_string(), "Confirmed: Peak Exhaustion".to_string())
                    }
                } else {
                    ("Pending Neckline Test".to_string(), "Confirmed: Peak Exhaustion".to_string())
                };

                let strictness = 1.0 - (p1.price - p2.price).abs() / p1.price.max(p2.price);
                let confidence = (0.55 + strictness * 0.25 + depth.min(0.2)).min(1.0);
                // Task 11: the trough between the peaks is the neckline; the measured move
                // is the peaks' average height above it; the higher peak invalidates.
                let neck = trough.price;
                let levels = Self::measure(
                    candles, p2.idx + 1, false, move |_| neck, avg_peak - neck, p1.price.max(p2.price),
                );

                out.push(ChartPattern {
                    pattern_type: "Double Top".to_string(),
                    sentiment: "Bearish".to_string(),
                    confidence,
                    start_idx: p1.idx,
                    end_idx: p2.idx,
                    description: format!(
                        "Double Top at {:.2} and {:.2}. Vol exhaustion: P2 < P1. {}.",
                        p1.price, p2.price, breakout_status
                    ),
                    structural_bias: "Bearish Reversal".to_string(),
                    geometric_strictness: strictness,
                    volume_validation: volume_val,
                    breakout_status,
                    is_forming: false,
                    formation_progress: 0.0,
                    levels,
                });
            }
        }
    }

    // ── 5. Double Bottom ───────────────────────────────────────────────────

    fn detect_double_bottom(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let troughs: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Trough).collect();
        if troughs.len() < 2 { return; }

        for window in troughs.windows(2) {
            let (t1, t2) = (window[0], window[1]);
            if !Self::prices_match(t1.price, t2.price, MATCH_TOLERANCE) { continue; }

            let peak_between = swings.iter()
                .find(|s| s.kind == SwingKind::Peak && s.idx > t1.idx && s.idx < t2.idx);
            if peak_between.is_none() { continue; }

            if t1.idx < candles.len() && t2.idx < candles.len() {
                let v_t1 = candles[t1.idx].volume;
                let v_t2 = candles[t2.idx].volume;
                if v_t2 >= v_t1 { continue; }

                let peak = peak_between.unwrap();
                let avg_trough = (t1.price + t2.price) / 2.0;
                let height = (peak.price - avg_trough) / avg_trough;
                if height < 0.005 { continue; }

                let current_price = candles.last().map(|c| c.close).unwrap_or(0.0);
                let is_breakout = current_price > peak.price;
                let current_idx = candles.len().saturating_sub(1);

                let (breakout_status, volume_val) = if is_breakout {
                    let sma = Self::get_volume_sma(candles, current_idx, 20);
                    let curr_vol = candles.last().map(|c| c.volume).unwrap_or(0.0);
                    if curr_vol > 1.2 * sma && sma > 0.0 {
                        ("Confirmed Breakout".to_string(), "Confirmed: Breakout Volume Boost".to_string())
                    } else {
                        ("Pending Neckline Test".to_string(), "Confirmed: Trough Exhaustion".to_string())
                    }
                } else {
                    ("Pending Neckline Test".to_string(), "Confirmed: Trough Exhaustion".to_string())
                };

                let strictness = 1.0 - (t1.price - t2.price).abs() / t1.price.max(t2.price);
                let confidence = (0.55 + strictness * 0.25 + height.min(0.2)).min(1.0);
                // Task 11: the mirror of the Double Top — the peak between is the
                // neckline, the lower trough invalidates.
                let neck = peak.price;
                let levels = Self::measure(
                    candles, t2.idx + 1, true, move |_| neck, neck - avg_trough, t1.price.min(t2.price),
                );

                out.push(ChartPattern {
                    pattern_type: "Double Bottom".to_string(),
                    sentiment: "Bullish".to_string(),
                    confidence,
                    start_idx: t1.idx,
                    end_idx: t2.idx,
                    description: format!(
                        "Double Bottom at {:.2} and {:.2}. Vol exhaustion: T2 < T1. {}.",
                        t1.price, t2.price, breakout_status
                    ),
                    structural_bias: "Bullish Reversal".to_string(),
                    geometric_strictness: strictness,
                    volume_validation: volume_val,
                    breakout_status,
                    is_forming: false,
                    formation_progress: 0.0,
                    levels,
                });
            }
        }
    }

    // ── 6. Triple Top ──────────────────────────────────────────────────────

    fn detect_triple_top(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let peaks: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Peak).collect();
        if peaks.len() < 3 { return; }

        for window in peaks.windows(3) {
            let (p1, p2, p3) = (window[0], window[1], window[2]);
            let avg = (p1.price + p2.price + p3.price) / 3.0;
            if !Self::prices_match(p1.price, avg, MATCH_TOLERANCE)
                || !Self::prices_match(p2.price, avg, MATCH_TOLERANCE)
                || !Self::prices_match(p3.price, avg, MATCH_TOLERANCE)
            { continue; }

            // Volume: p3 < p1 (exhaustion)
            if p1.idx < candles.len() && p3.idx < candles.len() {
                let v_p1 = candles[p1.idx].volume;
                let v_p3 = candles[p3.idx].volume;
                let vol_valid = v_p3 < v_p1;

                let max_dev = [(p1.price - avg).abs(), (p2.price - avg).abs(), (p3.price - avg).abs()]
                    .iter().cloned().fold(0.0_f64, f64::max);
                let confidence = (0.6 + 0.3 * (1.0 - max_dev / avg)).min(1.0);
                // Task 11: the neckline is the lowest trough between the first and third
                // peak (none -> infinite -> no levels); the highest peak invalidates.
                let neck = swings.iter()
                    .filter(|s| s.kind == SwingKind::Trough && s.idx > p1.idx && s.idx < p3.idx)
                    .map(|s| s.price)
                    .fold(f64::INFINITY, f64::min);
                let levels = Self::measure(
                    candles, p3.idx + 1, false, move |_| neck, avg - neck,
                    p1.price.max(p2.price).max(p3.price),
                );

                out.push(ChartPattern {
                    pattern_type: "Triple Top".to_string(),
                    sentiment: "Bearish".to_string(),
                    confidence,
                    start_idx: p1.idx,
                    end_idx: p3.idx,
                    description: format!(
                        "Triple Top at {:.2}, {:.2}, {:.2}. Strong resistance zone.",
                        p1.price, p2.price, p3.price
                    ),
                    structural_bias: "Bearish Reversal".to_string(),
                    geometric_strictness: 1.0 - max_dev / avg,
                    volume_validation: if vol_valid { "Confirmed: Reversal Exhaustion".to_string() } else { "Unconfirmed".to_string() },
                    breakout_status: "Pending Breakout".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                    levels,
                });
            }
        }
    }

    // ── 7. Triple Bottom ───────────────────────────────────────────────────

    fn detect_triple_bottom(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let troughs: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Trough).collect();
        if troughs.len() < 3 { return; }

        for window in troughs.windows(3) {
            let (t1, t2, t3) = (window[0], window[1], window[2]);
            let avg = (t1.price + t2.price + t3.price) / 3.0;
            if !Self::prices_match(t1.price, avg, MATCH_TOLERANCE)
                || !Self::prices_match(t2.price, avg, MATCH_TOLERANCE)
                || !Self::prices_match(t3.price, avg, MATCH_TOLERANCE)
            { continue; }

            if t1.idx < candles.len() && t3.idx < candles.len() {
                let v_t1 = candles[t1.idx].volume;
                let v_t3 = candles[t3.idx].volume;
                let vol_valid = v_t3 < v_t1;

                let max_dev = [(t1.price - avg).abs(), (t2.price - avg).abs(), (t3.price - avg).abs()]
                    .iter().cloned().fold(0.0_f64, f64::max);
                let confidence = (0.6 + 0.3 * (1.0 - max_dev / avg)).min(1.0);
                // Task 11: the mirror of the Triple Top — the highest peak between the
                // first and third trough is the neckline; the lowest trough invalidates.
                let neck = swings.iter()
                    .filter(|s| s.kind == SwingKind::Peak && s.idx > t1.idx && s.idx < t3.idx)
                    .map(|s| s.price)
                    .fold(f64::NEG_INFINITY, f64::max);
                let levels = Self::measure(
                    candles, t3.idx + 1, true, move |_| neck, neck - avg,
                    t1.price.min(t2.price).min(t3.price),
                );

                out.push(ChartPattern {
                    pattern_type: "Triple Bottom".to_string(),
                    sentiment: "Bullish".to_string(),
                    confidence,
                    start_idx: t1.idx,
                    end_idx: t3.idx,
                    description: format!(
                        "Triple Bottom at {:.2}, {:.2}, {:.2}. Strong support zone.",
                        t1.price, t2.price, t3.price
                    ),
                    structural_bias: "Bullish Reversal".to_string(),
                    geometric_strictness: 1.0 - max_dev / avg,
                    volume_validation: if vol_valid { "Confirmed: Reversal Exhaustion".to_string() } else { "Unconfirmed".to_string() },
                    breakout_status: "Pending Breakout".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                    levels,
                });
            }
        }
    }

    // ── 8. Quasimodo (QM) ──────────────────────────────────────────────────

    fn detect_quasimodo(
        candles: &[Candle],
        swings: &[SwingPoint],
        out: &mut Vec<ChartPattern>,
    ) {
        let n = swings.len();
        if n < 4 { return; }

        let s_x = swings[n - 4];
        let s_a = swings[n - 3];
        let s_b = swings[n - 2];
        let s_c = swings[n - 1];

        // Bullish QM: Trough(X) → Peak(A) → Lower-Trough(B) → Higher-Peak(C)
        if s_x.kind == SwingKind::Trough && s_a.kind == SwingKind::Peak
            && s_b.kind == SwingKind::Trough && s_c.kind == SwingKind::Peak
        {
            if s_b.price < s_x.price && s_c.price > s_a.price {
                if s_x.idx < candles.len() {
                    let v_x = candles[s_x.idx].volume;
                    let v_curr = candles.last().map(|c| c.volume).unwrap_or(0.0);
                    if v_curr < v_x {
                        out.push(ChartPattern {
                            pattern_type: "Quasimodo (Bullish)".to_string(),
                            sentiment: "Bullish".to_string(),
                            confidence: 0.85,
                            start_idx: s_x.idx,
                            end_idx: s_c.idx,
                            description: format!(
                                "Bullish QM: X={:.2}, A={:.2}, B={:.2}(lower low), C={:.2}(higher high). Volume exhaustion confirmed.",
                                s_x.price, s_a.price, s_b.price, s_c.price
                            ),
                            structural_bias: "Bullish Reversal".to_string(),
                            geometric_strictness: 0.90,
                            volume_validation: "Confirmed: Reversal Exhaustion".to_string(),
                            breakout_status: "Pending Neckline Test".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                            // Task 11: a reversal at C with no measured-move convention;
                            // B's lower low is where it is wrong.
                            levels: PatternLevels { invalidation: Some(s_b.price), ..Default::default() },
                        });
                    }
                }
            }
        }

        // Bearish QM: Peak(X) → Trough(A) → Higher-Peak(B) → Lower-Trough(C)
        if s_x.kind == SwingKind::Peak && s_a.kind == SwingKind::Trough
            && s_b.kind == SwingKind::Peak && s_c.kind == SwingKind::Trough
        {
            if s_b.price > s_x.price && s_c.price < s_a.price {
                if s_x.idx < candles.len() {
                    let v_x = candles[s_x.idx].volume;
                    let v_curr = candles.last().map(|c| c.volume).unwrap_or(0.0);
                    if v_curr < v_x {
                        out.push(ChartPattern {
                            pattern_type: "Quasimodo (Bearish)".to_string(),
                            sentiment: "Bearish".to_string(),
                            confidence: 0.85,
                            start_idx: s_x.idx,
                            end_idx: s_c.idx,
                            description: format!(
                                "Bearish QM: X={:.2}, A={:.2}, B={:.2}(higher high), C={:.2}(lower low). Volume exhaustion confirmed.",
                                s_x.price, s_a.price, s_b.price, s_c.price
                            ),
                            structural_bias: "Bearish Reversal".to_string(),
                            geometric_strictness: 0.90,
                            volume_validation: "Confirmed: Reversal Exhaustion".to_string(),
                            breakout_status: "Pending Neckline Test".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                            // Task 11: B's higher high is where the bearish QM is wrong.
                            levels: PatternLevels { invalidation: Some(s_b.price), ..Default::default() },
                        });
                    }
                }
            }
        }
    }

    // ── 9. Three Drives ────────────────────────────────────────────────────

    fn detect_three_drives(
        candles: &[Candle],
        swings: &[SwingPoint],
        out: &mut Vec<ChartPattern>,
    ) {
        let n = swings.len();
        if n < 6 { return; }

        let d1 = swings[n - 6];
        let c1 = swings[n - 5];
        let d2 = swings[n - 4];
        let c2 = swings[n - 3];
        let d3 = swings[n - 2];
        let _final = swings[n - 1];

        if d1.kind != d2.kind || d2.kind != d3.kind { return; }

        let is_bullish = d1.kind == SwingKind::Trough;
        let d1_range = (c1.price - d1.price).abs();
        let d2_range = (c2.price - d2.price).abs();
        if d1_range < 1e-9 || d2_range < 1e-9 { return; }

        let d2_ext = (d2.price - c1.price).abs() / d1_range;
        let d3_ext = (d3.price - c2.price).abs() / d2_range;

        if (Self::matches_fib(d2_ext, 1.272) || Self::matches_fib(d2_ext, 1.618))
            && (Self::matches_fib(d3_ext, 1.272) || Self::matches_fib(d3_ext, 1.618))
        {
            if d1.idx < candles.len() && d3.idx < candles.len() {
                let v_d1 = candles[d1.idx].volume;
                let v_d3 = candles[d3.idx].volume;
                if v_d3 < v_d1 {
                    let bias = if is_bullish { "Bullish Reversal" } else { "Bearish Reversal" };
                    out.push(ChartPattern {
                        pattern_type: "Three Drives".to_string(),
                        sentiment: if is_bullish { "Bullish".to_string() } else { "Bearish".to_string() },
                        confidence: 0.90,
                        start_idx: d1.idx,
                        end_idx: d3.idx,
                        description: format!(
                            "Three Drives: D2 ext {:.3}, D3 ext {:.3}. Volume exhaustion at Drive 3. {}.",
                            d2_ext, d3_ext, bias
                        ),
                        structural_bias: bias.to_string(),
                        geometric_strictness: 0.90,
                        volume_validation: "Confirmed: Reversal Exhaustion".to_string(),
                        breakout_status: "Confirmed".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                    // Task 11: an exhaustion reversal at drive 3 with no breakout line;
                    // only the extreme of the three drives says it is wrong.
                    levels: PatternLevels {
                        invalidation: Some(if is_bullish {
                            d1.price.min(d2.price).min(d3.price)
                        } else {
                            d1.price.max(d2.price).max(d3.price)
                        }),
                        ..Default::default()
                    },
                    });
                }
            }
        }
    }

    // ── 10. Rising Wedge ───────────────────────────────────────────────────

    fn detect_rising_wedge(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let peaks: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Peak).map(|s| (s.idx as f64, s.price)).collect();
        let troughs: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Trough).map(|s| (s.idx as f64, s.price)).collect();
        if peaks.len() < 2 || troughs.len() < 2 { return; }

        let (res_slope, res_int) = Self::linear_regression(&peaks);
        let (sup_slope, sup_int) = Self::linear_regression(&troughs);

        if res_slope > 0.0 && sup_slope > 0.0 && sup_slope > res_slope {
            let first_idx = swings.first().unwrap().idx;
            let last_idx = swings.last().unwrap().idx;
            let vol_slope = Self::get_volume_slope(candles, first_idx, last_idx);

            let convergence = (sup_slope - res_slope) / sup_slope.abs().max(1e-9);
            let confidence = (0.45 + convergence.min(0.4) * 0.5).min(0.9);
            // Task 11: the support line breaks it (x = absolute bar index, as fitted);
            // the measured move is the wedge's width at its first swing; the highest
            // peak invalidates.
            let res = move |k: usize| res_slope * k as f64 + res_int;
            let sup = move |k: usize| sup_slope * k as f64 + sup_int;
            let height = res(first_idx) - sup(first_idx);
            let top = peaks.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
            let levels = Self::measure(candles, last_idx + 1, false, sup, height, top);

            out.push(ChartPattern {
                pattern_type: "Rising Wedge".to_string(),
                sentiment: "Bearish".to_string(),
                confidence,
                start_idx: first_idx,
                end_idx: last_idx,
                description: format!(
                    "Rising Wedge: sup slope {:.6}, res slope {:.6}. Converging upward — bearish reversal.",
                    sup_slope, res_slope
                ),
                structural_bias: "Bearish Reversal".to_string(),
                geometric_strictness: 0.85,
                volume_validation: if vol_slope < 0.0 { "Confirmed: Consolidation Drying".to_string() } else { "Unconfirmed".to_string() },
                breakout_status: "Pending Breakout".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                levels,
            });
        }
    }

    // ── 11. Falling Wedge ──────────────────────────────────────────────────

    fn detect_falling_wedge(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let peaks: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Peak).map(|s| (s.idx as f64, s.price)).collect();
        let troughs: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Trough).map(|s| (s.idx as f64, s.price)).collect();
        if peaks.len() < 2 || troughs.len() < 2 { return; }

        let (res_slope, res_int) = Self::linear_regression(&peaks);
        let (sup_slope, sup_int) = Self::linear_regression(&troughs);

        if res_slope < 0.0 && sup_slope < 0.0 && res_slope < sup_slope {
            let first_idx = swings.first().unwrap().idx;
            let last_idx = swings.last().unwrap().idx;
            let vol_slope = Self::get_volume_slope(candles, first_idx, last_idx);

            let convergence = (sup_slope - res_slope).abs() / res_slope.abs().max(1e-9);
            let confidence = (0.45 + convergence.min(0.4) * 0.5).min(0.9);
            // Task 11: the mirror of the Rising Wedge — the resistance line breaks it,
            // the lowest trough invalidates.
            let res = move |k: usize| res_slope * k as f64 + res_int;
            let sup = move |k: usize| sup_slope * k as f64 + sup_int;
            let height = res(first_idx) - sup(first_idx);
            let bottom = troughs.iter().map(|t| t.1).fold(f64::INFINITY, f64::min);
            let levels = Self::measure(candles, last_idx + 1, true, res, height, bottom);

            out.push(ChartPattern {
                pattern_type: "Falling Wedge".to_string(),
                sentiment: "Bullish".to_string(),
                confidence,
                start_idx: first_idx,
                end_idx: last_idx,
                description: format!(
                    "Falling Wedge: res slope {:.6}, sup slope {:.6}. Converging downward — bullish reversal.",
                    res_slope, sup_slope
                ),
                structural_bias: "Bullish Reversal".to_string(),
                geometric_strictness: 0.85,
                volume_validation: if vol_slope < 0.0 { "Confirmed: Consolidation Drying".to_string() } else { "Unconfirmed".to_string() },
                breakout_status: "Pending Breakout".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                levels,
            });
        }
    }

    // ── 12. Flags & Pennants ───────────────────────────────────────────────

    fn detect_flags_and_pennants(
        candles: &[Candle],
        swings: &[SwingPoint],
        out: &mut Vec<ChartPattern>,
    ) {
        let n = candles.len();
        if n < MIN_FLAGPOLE_CANDLES + 10 || swings.len() < 4 { return; }

        let pole_end = n.saturating_sub(10);
        if pole_end < MIN_FLAGPOLE_CANDLES { return; }

        let pole_start = pole_end.saturating_sub(20).max(0);
        let pole_candles = &candles[pole_start..pole_end];
        if pole_candles.is_empty() { return; }

        let pole_open = pole_candles.first().unwrap().open;
        let pole_close = pole_candles.last().unwrap().close;
        let pole_range = (pole_close - pole_open).abs();
        let avg_price = (pole_open + pole_close) / 2.0;

        if avg_price < 1e-9 || pole_range / avg_price < FLAGPOLE_MIN_RANGE_RATIO { return; }

        let is_bullish_pole = pole_close > pole_open;

        // Consolidation zone swings
        let consol_swings: Vec<&SwingPoint> = swings.iter().filter(|s| s.idx >= pole_end).collect();
        if consol_swings.len() < 2 { return; }

        let consol_peaks: Vec<(f64, f64)> = consol_swings.iter()
            .filter(|s| s.kind == SwingKind::Peak).map(|s| (s.idx as f64, s.price)).collect();
        let consol_troughs: Vec<(f64, f64)> = consol_swings.iter()
            .filter(|s| s.kind == SwingKind::Trough).map(|s| (s.idx as f64, s.price)).collect();
        if consol_peaks.is_empty() || consol_troughs.is_empty() { return; }

        let (res_slope, res_int) = if consol_peaks.len() >= 2 { Self::linear_regression(&consol_peaks) } else { (0.0, consol_peaks[0].1) };
        let (sup_slope, sup_int) = if consol_troughs.len() >= 2 { Self::linear_regression(&consol_troughs) } else { (0.0, consol_troughs[0].1) };

        let consol_high = consol_swings.iter().filter(|s| s.kind == SwingKind::Peak).map(|s| s.price).fold(f64::MIN, f64::max);
        let consol_low = consol_swings.iter().filter(|s| s.kind == SwingKind::Trough).map(|s| s.price).fold(f64::MAX, f64::min);
        let consol_range = consol_high - consol_low;

        // Volume drying during consolidation
        let vol_slope = Self::get_volume_slope(candles, pole_end, n - 1);
        let vol_drying = vol_slope < 0.0;

        let slopes_parallel = (res_slope - sup_slope).abs() < 0.001;
        let slopes_converge = (res_slope < 0.0 && sup_slope > 0.0)
            || (res_slope.abs() > 0.0001 && sup_slope.abs() > 0.0001 && res_slope * sup_slope < 0.0);

        let ratio = if pole_range > 1e-9 { consol_range / pole_range } else { 1.0 };
        if ratio > 0.5 { return; }

        let base_confidence = 0.50 + (1.0 - ratio) * 0.3;

        // Task 11, shared by the flag and the pennant: the consolidation's far boundary
        // (resistance after a bullish pole, support after a bearish one; x = absolute bar
        // index, as fitted) breaks it, the pole is the measured move, and the
        // consolidation's last swing on the other side (its last low after a bullish
        // pole, its last high after a bearish one) invalidates.
        let from = consol_swings[consol_swings.len() - 1].idx + 1;
        let levels = if is_bullish_pole {
            let inv = consol_troughs[consol_troughs.len() - 1].1;
            Self::measure(candles, from, true, move |k: usize| res_slope * k as f64 + res_int, pole_range, inv)
        } else {
            let inv = consol_peaks[consol_peaks.len() - 1].1;
            Self::measure(candles, from, false, move |k: usize| sup_slope * k as f64 + sup_int, pole_range, inv)
        };

        if slopes_parallel {
            let (pattern, sentiment, bias) = if is_bullish_pole {
                ("Bull Flag", "Bullish", "Bullish Continuation")
            } else {
                ("Bear Flag", "Bearish", "Bearish Continuation")
            };
            out.push(ChartPattern {
                pattern_type: pattern.to_string(),
                sentiment: sentiment.to_string(),
                confidence: base_confidence.min(0.9),
                start_idx: pole_start,
                end_idx: n - 1,
                description: format!(
                    "{}: Pole {:.2} → {:.2} ({:.1}% move). Parallel consolidation channel.",
                    pattern, pole_open, pole_close, (pole_range / avg_price) * 100.0
                ),
                structural_bias: bias.to_string(),
                geometric_strictness: 1.0 - ratio,
                volume_validation: if vol_drying { "Confirmed: Consolidation Drying".to_string() } else { "Unconfirmed".to_string() },
                breakout_status: "Pending Breakout".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                levels: levels.clone(),
            });
        }

        if slopes_converge {
            let (pattern, sentiment, bias) = if is_bullish_pole {
                ("Bull Pennant", "Bullish", "Bullish Continuation")
            } else {
                ("Bear Pennant", "Bearish", "Bearish Continuation")
            };
            out.push(ChartPattern {
                pattern_type: pattern.to_string(),
                sentiment: sentiment.to_string(),
                confidence: (base_confidence + 0.05).min(0.9),
                start_idx: pole_start,
                end_idx: n - 1,
                description: format!(
                    "{}: Pole {:.2} → {:.2} ({:.1}% move). Converging triangle after sharp move.",
                    pattern, pole_open, pole_close, (pole_range / avg_price) * 100.0
                ),
                structural_bias: bias.to_string(),
                geometric_strictness: 1.0 - ratio,
                volume_validation: if vol_drying { "Confirmed: Consolidation Drying".to_string() } else { "Unconfirmed".to_string() },
                breakout_status: "Pending Breakout".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                levels,
            });
        }
    }

    // ── 13. Cup and Handle ─────────────────────────────────────────────────

    fn detect_cup_and_handle(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let troughs: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Trough).collect();
        let peaks: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Peak).collect();
        if troughs.len() < 2 || peaks.len() < 3 { return; }

        for i in 0..peaks.len().saturating_sub(2) {
            let left_rim = peaks[i];
            let right_rim = peaks[i + 1];
            if !Self::prices_match(left_rim.price, right_rim.price, CUP_ASYMMETRY_TOLERANCE) { continue; }

            let cup_bottom = troughs.iter()
                .filter(|t| t.idx > left_rim.idx && t.idx < right_rim.idx)
                .min_by(|a, b| a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal))
                .copied();
            if cup_bottom.is_none() { continue; }
            let bottom = cup_bottom.unwrap();

            let rim_avg = (left_rim.price + right_rim.price) / 2.0;
            let cup_depth = (rim_avg - bottom.price) / rim_avg;
            if cup_depth < 0.02 || cup_depth > 0.35 { continue; }

            let handle_trough = troughs.iter().filter(|t| t.idx > right_rim.idx).next().copied();
            if let Some(handle) = handle_trough {
                let handle_depth = (right_rim.price - handle.price) / right_rim.price;
                if handle_depth > 0.0 && handle_depth < cup_depth * 0.5 {
                    let confidence = (0.55 + cup_depth.min(0.2) + (1.0 - handle_depth / cup_depth) * 0.15).min(0.95);
                    // Task 11: the right rim is the breakout, the cup's depth below the
                    // rims the measured move, the handle's low the invalidation.
                    let rim = right_rim.price;
                    let levels = Self::measure(
                        candles, handle.idx + 1, true, move |_| rim, rim_avg - bottom.price, handle.price,
                    );
                    out.push(ChartPattern {
                        pattern_type: "Cup and Handle".to_string(),
                        sentiment: "Bullish".to_string(),
                        confidence,
                        start_idx: left_rim.idx,
                        end_idx: handle.idx,
                        description: format!(
                            "Cup and Handle: rims {:.2}/{:.2}, bottom {:.2} ({:.1}% deep), handle {:.2}.",
                            left_rim.price, right_rim.price, bottom.price, cup_depth * 100.0, handle.price
                        ),
                        structural_bias: "Bullish Continuation".to_string(),
                        geometric_strictness: 0.85,
                        volume_validation: "Geometric Only".to_string(),
                        breakout_status: "Pending Breakout".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                    levels,
                    });
                }
            }
        }
    }

    // ── 14. Inverse Cup and Handle ─────────────────────────────────────────

    fn detect_inverse_cup_and_handle(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let peaks: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Peak).collect();
        let troughs: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Trough).collect();
        if peaks.len() < 2 || troughs.len() < 3 { return; }

        for i in 0..troughs.len().saturating_sub(2) {
            let left_rim = troughs[i];
            let right_rim = troughs[i + 1];
            if !Self::prices_match(left_rim.price, right_rim.price, CUP_ASYMMETRY_TOLERANCE) { continue; }

            let dome_top = peaks.iter()
                .filter(|p| p.idx > left_rim.idx && p.idx < right_rim.idx)
                .max_by(|a, b| a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal))
                .copied();
            if dome_top.is_none() { continue; }
            let dome = dome_top.unwrap();

            let rim_avg = (left_rim.price + right_rim.price) / 2.0;
            let dome_height = (dome.price - rim_avg) / rim_avg;
            if dome_height < 0.02 || dome_height > 0.35 { continue; }

            let handle_peak = peaks.iter().filter(|p| p.idx > right_rim.idx).next().copied();
            if let Some(handle) = handle_peak {
                let handle_height = (handle.price - right_rim.price) / right_rim.price;
                if handle_height > 0.0 && handle_height < dome_height * 0.5 {
                    let confidence = (0.55 + dome_height.min(0.2) + (1.0 - handle_height / dome_height) * 0.15).min(0.95);
                    // Task 11: the mirror of the Cup — the right rim breaks it, the dome's
                    // height above the rims is the measured move, the handle's high the
                    // invalidation.
                    let rim = right_rim.price;
                    let levels = Self::measure(
                        candles, handle.idx + 1, false, move |_| rim, dome.price - rim_avg, handle.price,
                    );
                    out.push(ChartPattern {
                        pattern_type: "Inverse Cup and Handle".to_string(),
                        sentiment: "Bearish".to_string(),
                        confidence,
                        start_idx: left_rim.idx,
                        end_idx: handle.idx,
                        description: format!(
                            "Inverse Cup: rims {:.2}/{:.2}, dome {:.2} ({:.1}% high), handle {:.2}.",
                            left_rim.price, right_rim.price, dome.price, dome_height * 100.0, handle.price
                        ),
                        structural_bias: "Bearish Continuation".to_string(),
                        geometric_strictness: 0.85,
                        volume_validation: "Geometric Only".to_string(),
                        breakout_status: "Pending Breakout".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                    levels,
                    });
                }
            }
        }
    }

    // ── 15-17. Triangles ───────────────────────────────────────────────────

    fn detect_triangles(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let peaks: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Peak).map(|s| (s.idx as f64, s.price)).collect();
        let troughs: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Trough).map(|s| (s.idx as f64, s.price)).collect();
        if peaks.len() < 2 || troughs.len() < 2 { return; }

        let (res_slope, res_int) = Self::linear_regression(&peaks);
        let (sup_slope, sup_int) = Self::linear_regression(&troughs);

        let start = swings.first().map(|s| s.idx).unwrap_or(0);
        let end = swings.last().map(|s| s.idx).unwrap_or(0);
        let vol_slope = Self::get_volume_slope(candles, start, end);
        let vol_drying = vol_slope < 0.0;

        // Task 11, for all three: the fitted lines (x = absolute bar index), the
        // triangle's height at its first swing as the measured move, and the last swing
        // on the far side of the breakout as the invalidation.
        let res = move |k: usize| res_slope * k as f64 + res_int;
        let sup = move |k: usize| sup_slope * k as f64 + sup_int;
        let height = res(start) - sup(start);
        let last_low = troughs[troughs.len() - 1].1;
        let last_high = peaks[peaks.len() - 1].1;

        // Ascending Triangle
        if res_slope.abs() < FLAT_SLOPE_THRESHOLD && sup_slope > FLAT_SLOPE_THRESHOLD {
            let confidence = (0.5 + sup_slope.abs().min(0.01) * 30.0).min(0.85);
            out.push(ChartPattern {
                pattern_type: "Ascending Triangle".to_string(),
                sentiment: "Bullish".to_string(),
                confidence,
                start_idx: start, end_idx: end,
                description: format!("Ascending Triangle: flat resistance, rising support."),
                structural_bias: "Bullish Breakout".to_string(),
                geometric_strictness: 0.90,
                volume_validation: if vol_drying { "Confirmed: Consolidation Drying".to_string() } else { "Unconfirmed".to_string() },
                breakout_status: "Pending Breakout".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                levels: Self::measure(candles, end + 1, true, res, height, last_low),
            });
        }

        // Descending Triangle
        if sup_slope.abs() < FLAT_SLOPE_THRESHOLD && res_slope < -FLAT_SLOPE_THRESHOLD {
            let confidence = (0.5 + res_slope.abs().min(0.01) * 30.0).min(0.85);
            out.push(ChartPattern {
                pattern_type: "Descending Triangle".to_string(),
                sentiment: "Bearish".to_string(),
                confidence,
                start_idx: start, end_idx: end,
                description: format!("Descending Triangle: falling resistance, flat support."),
                structural_bias: "Bearish Breakout".to_string(),
                geometric_strictness: 0.90,
                volume_validation: if vol_drying { "Confirmed: Consolidation Drying".to_string() } else { "Unconfirmed".to_string() },
                breakout_status: "Pending Breakout".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                levels: Self::measure(candles, end + 1, false, sup, height, last_high),
            });
        }

        // Symmetrical Triangle
        if res_slope < -FLAT_SLOPE_THRESHOLD && sup_slope > FLAT_SLOPE_THRESHOLD {
            let convergence_rate = (res_slope.abs() + sup_slope.abs()) / 2.0;
            let confidence = (0.45 + convergence_rate.min(0.01) * 30.0).min(0.85);
            out.push(ChartPattern {
                pattern_type: "Symmetrical Triangle".to_string(),
                sentiment: "Neutral".to_string(),
                confidence,
                start_idx: start, end_idx: end,
                description: format!("Symmetrical Triangle: converging trendlines. Breakout direction determines bias."),
                structural_bias: "Bilateral Breakout".to_string(),
                geometric_strictness: 0.90,
                volume_validation: if vol_drying { "Confirmed: Consolidation Drying".to_string() } else { "Unconfirmed".to_string() },
                breakout_status: "Pending Breakout".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                // Neutral until a closed bar breaks a side; `sentiment` stays Neutral either
                // way (the tier reads only directional patterns), the levels follow the break.
                levels: Self::measure_bilateral(candles, end + 1, res, sup, height, last_low, last_high),
            });
        }
    }

    // ── 18. Rectangle ──────────────────────────────────────────────────────

    fn detect_rectangle(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let peaks: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Peak).map(|s| (s.idx as f64, s.price)).collect();
        let troughs: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Trough).map(|s| (s.idx as f64, s.price)).collect();
        if peaks.len() < 2 || troughs.len() < 2 { return; }

        let (res_slope, _) = Self::linear_regression(&peaks);
        let (sup_slope, _) = Self::linear_regression(&troughs);

        if res_slope.abs() < FLAT_SLOPE_THRESHOLD && sup_slope.abs() < FLAT_SLOPE_THRESHOLD {
            let avg_res = peaks.iter().map(|p| p.1).sum::<f64>() / peaks.len() as f64;
            let avg_sup = troughs.iter().map(|p| p.1).sum::<f64>() / troughs.len() as f64;
            let channel_width = avg_res - avg_sup;
            let mid = (avg_res + avg_sup) / 2.0;

            if mid.abs() < 1e-9 || channel_width / mid < 0.005 { return; }

            let res_dev: f64 = peaks.iter().map(|p| (p.1 - avg_res).abs()).sum::<f64>() / peaks.len() as f64;
            let sup_dev: f64 = troughs.iter().map(|p| (p.1 - avg_sup).abs()).sum::<f64>() / troughs.len() as f64;
            let tightness = 1.0 - ((res_dev + sup_dev) / (2.0 * channel_width)).min(0.5);
            let confidence = (0.45 + tightness * 0.4).min(0.9);

            let start = swings.first().map(|s| s.idx).unwrap_or(0);
            let end = swings.last().map(|s| s.idx).unwrap_or(0);

            out.push(ChartPattern {
                pattern_type: "Rectangle".to_string(),
                sentiment: "Neutral".to_string(),
                confidence,
                start_idx: start,
                end_idx: end,
                description: format!(
                    "Rectangle: support {:.2}, resistance {:.2}. Width {:.2} ({:.1}%).",
                    avg_sup, avg_res, channel_width, (channel_width / mid) * 100.0
                ),
                structural_bias: "Bilateral Breakout".to_string(),
                geometric_strictness: tightness,
                volume_validation: "Geometric Only".to_string(),
                breakout_status: "Pending Breakout".to_string(),
                    is_forming: false,
                    formation_progress: 0.0,
                // Task 11: the averaged horizontal boundaries, the channel's width as the
                // measured move; Neutral until a closed bar breaks a side (as the triangle).
                levels: Self::measure_bilateral(
                    candles, end + 1, move |_| avg_res, move |_| avg_sup, channel_width,
                    troughs[troughs.len() - 1].1, peaks[peaks.len() - 1].1,
                ),
            });
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Phase 10: FORMING PATTERN ANALYSIS
    // Focuses on recent candles to detect patterns that are currently forming.
    // Uses a smaller swing window to catch recent pivots and treats the latest
    // candle as a provisional (unconfirmed) swing point.
    // ════════════════════════════════════════════════════════════════════════

    const FORMING_SWING_WINDOW: usize = 2;

    /// Analyze the most recent `window` candles for patterns that are currently
    /// forming. Unlike `analyze()`, this uses a smaller swing detection window
    /// (2 instead of 5), includes the latest candle as a provisional swing
    /// point, and returns patterns tagged with `is_forming = true` and a
    /// `formation_progress` estimate.
    pub fn analyze_forming(candles: &[Candle], window: usize) -> Vec<ChartPattern> {
        if candles.len() < 10 {
            return Vec::new();
        }

        // Focus on the recent window of candles (but keep enough lookback for
        // volume SMA calculations — take max(window, 30) from the end)
        let lookback = window.max(30).min(candles.len());
        let focus_start = candles.len().saturating_sub(lookback);
        let focus = &candles[focus_start..];

        // Use a smaller swing window to detect more recent pivots
        let mut raw_swings = Self::find_swings_forming(focus);

        // Add the current (latest) candle as a provisional swing point if it
        // could be a local extremum within a small neighborhood
        Self::add_provisional_swing(focus, &mut raw_swings);

        if raw_swings.len() < 2 {
            return Vec::new();
        }

        let swings = Self::alternate_swings(&raw_swings);
        if swings.len() < 2 {
            return Vec::new();
        }

        let mut patterns: Vec<ChartPattern> = Vec::new();

        // Run forming-aware detection for each pattern archetype
        Self::detect_forming_double_top(focus, &swings, &mut patterns);
        Self::detect_forming_double_bottom(focus, &swings, &mut patterns);
        Self::detect_forming_head_and_shoulders(focus, &swings, &mut patterns);
        Self::detect_forming_inverse_head_and_shoulders(focus, &swings, &mut patterns);
        Self::detect_forming_triple_top(focus, &swings, &mut patterns);
        Self::detect_forming_triple_bottom(focus, &swings, &mut patterns);
        Self::detect_forming_triangles(focus, &swings, &mut patterns);
        Self::detect_forming_wedges(focus, &swings, &mut patterns);
        Self::detect_forming_flags_pennants(focus, &swings, &mut patterns);
        Self::detect_forming_harmonics(focus, &swings, &mut patterns);
        Self::detect_forming_rectangle(focus, &swings, &mut patterns);

        // Adjust indices to be relative to the full candle array
        for p in &mut patterns {
            p.start_idx += focus_start;
            p.end_idx += focus_start;
        }

        // Sort by formation progress descending (most complete first),
        // then by confidence
        patterns.sort_by(|a, b| {
            b.formation_progress
                .partial_cmp(&a.formation_progress)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        patterns
    }

    /// Swing detection with a smaller window (2) for forming-pattern analysis.
    /// This catches pivots closer to the current bar.
    fn find_swings_forming(candles: &[Candle]) -> Vec<SwingPoint> {
        let mut swings = Vec::new();
        let n = candles.len();
        let w = Self::FORMING_SWING_WINDOW;
        if n < w * 2 + 1 {
            return swings;
        }

        for i in w..(n - w) {
            let mut is_high = true;
            let mut is_low = true;

            for j in (i.saturating_sub(w))..=(i + w).min(n - 1) {
                if j == i { continue; }
                if candles[j].high >= candles[i].high { is_high = false; }
                if candles[j].low <= candles[i].low { is_low = false; }
            }

            if is_high {
                swings.push(SwingPoint { idx: i, price: candles[i].high, kind: SwingKind::Peak });
            }
            if is_low {
                swings.push(SwingPoint { idx: i, price: candles[i].low, kind: SwingKind::Trough });
            }
        }

        swings.sort_by_key(|s| s.idx);
        swings
    }

    /// Add the latest candle as a provisional swing point if it is a local
    /// extremum compared to the last few candles (only needs left-side
    /// confirmation, no right-side since it's the current bar).
    fn add_provisional_swing(candles: &[Candle], swings: &mut Vec<SwingPoint>) {
        if candles.len() < 3 { return; }
        let n = candles.len();
        let last_idx = n - 1;
        let last = &candles[last_idx];

        // Check against the last 2 candles (left-side only)
        let lookback = 2.min(last_idx);
        let mut is_high = true;
        let mut is_low = true;

        for j in (last_idx - lookback)..last_idx {
            if candles[j].high >= last.high { is_high = false; }
            if candles[j].low <= last.low { is_low = false; }
        }

        if is_high {
            swings.push(SwingPoint { idx: last_idx, price: last.high, kind: SwingKind::Peak });
        } else if is_low {
            swings.push(SwingPoint { idx: last_idx, price: last.low, kind: SwingKind::Trough });
        }
    }

    // ── Forming Pattern Detectors ──────────────────────────────────────────

    /// Forming Double Top: Detects when one peak has formed and the current
    /// price is approaching the same level — potential second peak forming.
    fn detect_forming_double_top(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let peaks: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Peak).collect();
        if peaks.is_empty() { return; }

        let current_price = candles.last().map(|c| c.close).unwrap_or(0.0);
        let current_high = candles.last().map(|c| c.high).unwrap_or(0.0);
        let n = candles.len();

        // Check completed double tops (both peaks detected in recent swings)
        for window in peaks.windows(2) {
            let (p1, p2) = (window[0], window[1]);
            if Self::prices_match(p1.price, p2.price, MATCH_TOLERANCE * 2.0) {
                let trough_between = swings.iter()
                    .find(|s| s.kind == SwingKind::Trough && s.idx > p1.idx && s.idx < p2.idx);
                if trough_between.is_none() { continue; }
                let trough = trough_between.unwrap();

                let avg_peak = (p1.price + p2.price) / 2.0;
                let depth = (avg_peak - trough.price) / avg_peak;
                if depth < 0.003 { continue; }

                // Check if price is now moving below the trough (breakout forming)
                let progress = if current_price < trough.price { 0.95 } else { 0.80 };

                out.push(ChartPattern {
                    pattern_type: "Double Top".to_string(),
                    sentiment: "Bearish".to_string(),
                    confidence: (0.50 + depth.min(0.2)).min(0.85),
                    start_idx: p1.idx,
                    end_idx: p2.idx.max(n.saturating_sub(1)),
                    description: format!(
                        "Forming Double Top: peaks at {:.2} and {:.2}, trough at {:.2}. {}.",
                        p1.price, p2.price, trough.price,
                        if current_price < trough.price { "Neckline breaking down" } else { "Watching for neckline break" }
                    ),
                    structural_bias: "Bearish Reversal".to_string(),
                    geometric_strictness: 1.0 - (p1.price - p2.price).abs() / p1.price.max(1e-9),
                    volume_validation: "Forming".to_string(),
                    breakout_status: if current_price < trough.price { "Breaking Down".to_string() } else { "Forming".to_string() },
                    is_forming: true,
                    formation_progress: progress,
                    levels: PatternLevels::default(),
                });
            }
        }

        // Check for POTENTIAL double top: one peak exists and price is approaching it
        for peak in &peaks {
            let price_proximity = (current_high - peak.price).abs() / peak.price.max(1e-9);
            if price_proximity < 0.02 && current_high > peak.price * 0.97 {
                // Price is within 2% of a previous peak — potential second top forming
                let has_trough_after = swings.iter().any(|s|
                    s.kind == SwingKind::Trough && s.idx > peak.idx
                );
                if !has_trough_after { continue; }

                out.push(ChartPattern {
                    pattern_type: "Double Top".to_string(),
                    sentiment: "Bearish".to_string(),
                    confidence: 0.40 + (1.0 - price_proximity) * 0.2,
                    start_idx: peak.idx,
                    end_idx: n.saturating_sub(1),
                    description: format!(
                        "Potential Double Top forming: first peak at {:.2}, price now at {:.2} (within {:.1}%). Watching for rejection.",
                        peak.price, current_high, price_proximity * 100.0
                    ),
                    structural_bias: "Bearish Reversal".to_string(),
                    geometric_strictness: 1.0 - price_proximity,
                    volume_validation: "Forming".to_string(),
                    breakout_status: "Forming".to_string(),
                    is_forming: true,
                    formation_progress: 0.50,
                    levels: PatternLevels::default(),
                });
            }
        }
    }

    /// Forming Double Bottom: One trough exists and price is approaching the same level.
    fn detect_forming_double_bottom(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let troughs: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Trough).collect();
        if troughs.is_empty() { return; }

        let current_price = candles.last().map(|c| c.close).unwrap_or(0.0);
        let current_low = candles.last().map(|c| c.low).unwrap_or(f64::MAX);
        let n = candles.len();

        // Check completed double bottoms
        for window in troughs.windows(2) {
            let (t1, t2) = (window[0], window[1]);
            if Self::prices_match(t1.price, t2.price, MATCH_TOLERANCE * 2.0) {
                let peak_between = swings.iter()
                    .find(|s| s.kind == SwingKind::Peak && s.idx > t1.idx && s.idx < t2.idx);
                if peak_between.is_none() { continue; }
                let peak = peak_between.unwrap();

                let avg_trough = (t1.price + t2.price) / 2.0;
                let height = (peak.price - avg_trough) / avg_trough;
                if height < 0.003 { continue; }

                let progress = if current_price > peak.price { 0.95 } else { 0.80 };

                out.push(ChartPattern {
                    pattern_type: "Double Bottom".to_string(),
                    sentiment: "Bullish".to_string(),
                    confidence: (0.50 + height.min(0.2)).min(0.85),
                    start_idx: t1.idx,
                    end_idx: t2.idx.max(n.saturating_sub(1)),
                    description: format!(
                        "Forming Double Bottom: troughs at {:.2} and {:.2}, peak at {:.2}. {}.",
                        t1.price, t2.price, peak.price,
                        if current_price > peak.price { "Neckline breaking up" } else { "Watching for neckline break" }
                    ),
                    structural_bias: "Bullish Reversal".to_string(),
                    geometric_strictness: 1.0 - (t1.price - t2.price).abs() / t1.price.max(1e-9),
                    volume_validation: "Forming".to_string(),
                    breakout_status: if current_price > peak.price { "Breaking Up".to_string() } else { "Forming".to_string() },
                    is_forming: true,
                    formation_progress: progress,
                    levels: PatternLevels::default(),
                });
            }
        }

        // Potential: one trough exists and price approaching it
        for trough in &troughs {
            let price_proximity = (current_low - trough.price).abs() / trough.price.max(1e-9);
            if price_proximity < 0.02 && current_low < trough.price * 1.03 {
                let has_peak_after = swings.iter().any(|s|
                    s.kind == SwingKind::Peak && s.idx > trough.idx
                );
                if !has_peak_after { continue; }

                out.push(ChartPattern {
                    pattern_type: "Double Bottom".to_string(),
                    sentiment: "Bullish".to_string(),
                    confidence: 0.40 + (1.0 - price_proximity) * 0.2,
                    start_idx: trough.idx,
                    end_idx: n.saturating_sub(1),
                    description: format!(
                        "Potential Double Bottom forming: first trough at {:.2}, price now at {:.2} (within {:.1}%). Watching for bounce.",
                        trough.price, current_low, price_proximity * 100.0
                    ),
                    structural_bias: "Bullish Reversal".to_string(),
                    geometric_strictness: 1.0 - price_proximity,
                    volume_validation: "Forming".to_string(),
                    breakout_status: "Forming".to_string(),
                    is_forming: true,
                    formation_progress: 0.50,
                    levels: PatternLevels::default(),
                });
            }
        }
    }

    /// Forming H&S Top: Detects partial H&S formations (left shoulder + head, waiting for right shoulder).
    fn detect_forming_head_and_shoulders(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let peaks: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Peak).collect();
        let current_price = candles.last().map(|c| c.close).unwrap_or(0.0);
        let current_high = candles.last().map(|c| c.high).unwrap_or(0.0);
        let n = candles.len();

        // Full H&S in recent swings (3 peaks)
        if peaks.len() >= 3 {
            for window in peaks.windows(3) {
                let (left, head, right) = (window[0], window[1], window[2]);
                if head.price <= left.price || head.price <= right.price { continue; }
                if !Self::prices_match(left.price, right.price, SHOULDER_TOLERANCE * 1.5) { continue; }

                let shoulder_avg = (left.price + right.price) / 2.0;
                let head_prominence = (head.price - shoulder_avg) / shoulder_avg;
                let shoulder_symmetry = 1.0 - ((left.price - right.price).abs() / shoulder_avg);

                out.push(ChartPattern {
                    pattern_type: "Head & Shoulders Top".to_string(),
                    sentiment: "Bearish".to_string(),
                    confidence: (0.50 + head_prominence.min(0.25) + shoulder_symmetry * 0.15).min(0.90),
                    start_idx: left.idx,
                    end_idx: right.idx.max(n.saturating_sub(1)),
                    description: format!(
                        "Forming H&S Top: LS {:.2}, Head {:.2}, RS {:.2}. Watching for neckline break.",
                        left.price, head.price, right.price
                    ),
                    structural_bias: "Bearish Reversal".to_string(),
                    geometric_strictness: shoulder_symmetry,
                    volume_validation: "Forming".to_string(),
                    breakout_status: "Forming".to_string(),
                    is_forming: true,
                    formation_progress: 0.85,
                    levels: PatternLevels::default(),
                });
            }
        }

        // Partial H&S: Left shoulder + head formed, watching for right shoulder
        if peaks.len() >= 2 {
            for window in peaks.windows(2) {
                let (left, head) = (window[0], window[1]);
                if head.price <= left.price { continue; }

                // Check if price has pulled back from head and is now near shoulder level
                let last_trough = swings.iter().rev()
                    .find(|s| s.kind == SwingKind::Trough && s.idx > head.idx);

                if let Some(trough) = last_trough {
                    // Price pulled back after head — check if forming right shoulder
                    let shoulder_target = left.price;
                    let price_near_shoulder = (current_high - shoulder_target).abs() / shoulder_target.max(1e-9) < 0.05;

                    if price_near_shoulder || current_price > trough.price {
                        out.push(ChartPattern {
                            pattern_type: "Head & Shoulders Top".to_string(),
                            sentiment: "Bearish".to_string(),
                            confidence: 0.45,
                            start_idx: left.idx,
                            end_idx: n.saturating_sub(1),
                            description: format!(
                                "Potential H&S Top forming: LS {:.2}, Head {:.2}. Price at {:.2} — watching for right shoulder near {:.2}.",
                                left.price, head.price, current_price, shoulder_target
                            ),
                            structural_bias: "Bearish Reversal".to_string(),
                            geometric_strictness: 0.70,
                            volume_validation: "Forming".to_string(),
                            breakout_status: "Forming".to_string(),
                            is_forming: true,
                            formation_progress: 0.55,
                            levels: PatternLevels::default(),
                        });
                    }
                }
            }
        }
    }

    /// Forming Inverse H&S: Partial inverse formations.
    fn detect_forming_inverse_head_and_shoulders(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let troughs: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Trough).collect();
        let current_price = candles.last().map(|c| c.close).unwrap_or(0.0);
        let current_low = candles.last().map(|c| c.low).unwrap_or(f64::MAX);
        let n = candles.len();

        // Full inverse H&S
        if troughs.len() >= 3 {
            for window in troughs.windows(3) {
                let (left, head, right) = (window[0], window[1], window[2]);
                if head.price >= left.price || head.price >= right.price { continue; }
                if !Self::prices_match(left.price, right.price, SHOULDER_TOLERANCE * 1.5) { continue; }

                let shoulder_avg = (left.price + right.price) / 2.0;
                let head_depth = (shoulder_avg - head.price) / shoulder_avg;
                let shoulder_symmetry = 1.0 - ((left.price - right.price).abs() / shoulder_avg);

                out.push(ChartPattern {
                    pattern_type: "Inverse Head & Shoulders".to_string(),
                    sentiment: "Bullish".to_string(),
                    confidence: (0.50 + head_depth.min(0.25) + shoulder_symmetry * 0.15).min(0.90),
                    start_idx: left.idx,
                    end_idx: right.idx.max(n.saturating_sub(1)),
                    description: format!(
                        "Forming IH&S: LS {:.2}, Head {:.2}, RS {:.2}. Watching for neckline break.",
                        left.price, head.price, right.price
                    ),
                    structural_bias: "Bullish Reversal".to_string(),
                    geometric_strictness: shoulder_symmetry,
                    volume_validation: "Forming".to_string(),
                    breakout_status: "Forming".to_string(),
                    is_forming: true,
                    formation_progress: 0.85,
                    levels: PatternLevels::default(),
                });
            }
        }

        // Partial: Left shoulder + head, watching for right shoulder
        if troughs.len() >= 2 {
            for window in troughs.windows(2) {
                let (left, head) = (window[0], window[1]);
                if head.price >= left.price { continue; }

                let last_peak = swings.iter().rev()
                    .find(|s| s.kind == SwingKind::Peak && s.idx > head.idx);

                if let Some(peak) = last_peak {
                    let shoulder_target = left.price;
                    let price_near_shoulder = (current_low - shoulder_target).abs() / shoulder_target.max(1e-9) < 0.05;

                    if price_near_shoulder || current_price < peak.price {
                        out.push(ChartPattern {
                            pattern_type: "Inverse Head & Shoulders".to_string(),
                            sentiment: "Bullish".to_string(),
                            confidence: 0.45,
                            start_idx: left.idx,
                            end_idx: n.saturating_sub(1),
                            description: format!(
                                "Potential IH&S forming: LS {:.2}, Head {:.2}. Price at {:.2} — watching for right shoulder near {:.2}.",
                                left.price, head.price, current_price, shoulder_target
                            ),
                            structural_bias: "Bullish Reversal".to_string(),
                            geometric_strictness: 0.70,
                            volume_validation: "Forming".to_string(),
                            breakout_status: "Forming".to_string(),
                            is_forming: true,
                            formation_progress: 0.55,
                            levels: PatternLevels::default(),
                        });
                    }
                }
            }
        }
    }

    /// Forming Triple Top: 2 peaks at the same level, watching for 3rd test.
    fn detect_forming_triple_top(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let peaks: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Peak).collect();
        let current_high = candles.last().map(|c| c.high).unwrap_or(0.0);
        let n = candles.len();

        // Full triple top
        if peaks.len() >= 3 {
            for window in peaks.windows(3) {
                let (p1, p2, p3) = (window[0], window[1], window[2]);
                let avg = (p1.price + p2.price + p3.price) / 3.0;
                if !Self::prices_match(p1.price, avg, MATCH_TOLERANCE * 2.0)
                    || !Self::prices_match(p2.price, avg, MATCH_TOLERANCE * 2.0)
                    || !Self::prices_match(p3.price, avg, MATCH_TOLERANCE * 2.0)
                { continue; }

                out.push(ChartPattern {
                    pattern_type: "Triple Top".to_string(),
                    sentiment: "Bearish".to_string(),
                    confidence: 0.75,
                    start_idx: p1.idx,
                    end_idx: p3.idx.max(n.saturating_sub(1)),
                    description: format!(
                        "Forming Triple Top at {:.2}, {:.2}, {:.2}. Strong resistance being tested.",
                        p1.price, p2.price, p3.price
                    ),
                    structural_bias: "Bearish Reversal".to_string(),
                    geometric_strictness: 0.85,
                    volume_validation: "Forming".to_string(),
                    breakout_status: "Forming".to_string(),
                    is_forming: true,
                    formation_progress: 0.90,
                    levels: PatternLevels::default(),
                });
            }
        }

        // Partial: 2 peaks at same level, price approaching for 3rd test
        if peaks.len() >= 2 {
            for window in peaks.windows(2) {
                let (p1, p2) = (window[0], window[1]);
                if !Self::prices_match(p1.price, p2.price, MATCH_TOLERANCE * 2.0) { continue; }
                let avg = (p1.price + p2.price) / 2.0;
                let proximity = (current_high - avg).abs() / avg.max(1e-9);
                if proximity < 0.03 {
                    out.push(ChartPattern {
                        pattern_type: "Triple Top".to_string(),
                        sentiment: "Bearish".to_string(),
                        confidence: 0.45,
                        start_idx: p1.idx,
                        end_idx: n.saturating_sub(1),
                        description: format!(
                            "Potential Triple Top: 2 peaks at {:.2} and {:.2}. Price approaching for 3rd test at {:.2}.",
                            p1.price, p2.price, current_high
                        ),
                        structural_bias: "Bearish Reversal".to_string(),
                        geometric_strictness: 1.0 - proximity,
                        volume_validation: "Forming".to_string(),
                        breakout_status: "Forming".to_string(),
                        is_forming: true,
                        formation_progress: 0.60,
                        levels: PatternLevels::default(),
                    });
                }
            }
        }
    }

    /// Forming Triple Bottom: 2 troughs at the same level, watching for 3rd test.
    fn detect_forming_triple_bottom(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let troughs: Vec<&SwingPoint> = swings.iter().filter(|s| s.kind == SwingKind::Trough).collect();
        let current_low = candles.last().map(|c| c.low).unwrap_or(f64::MAX);
        let n = candles.len();

        // Full triple bottom
        if troughs.len() >= 3 {
            for window in troughs.windows(3) {
                let (t1, t2, t3) = (window[0], window[1], window[2]);
                let avg = (t1.price + t2.price + t3.price) / 3.0;
                if !Self::prices_match(t1.price, avg, MATCH_TOLERANCE * 2.0)
                    || !Self::prices_match(t2.price, avg, MATCH_TOLERANCE * 2.0)
                    || !Self::prices_match(t3.price, avg, MATCH_TOLERANCE * 2.0)
                { continue; }

                out.push(ChartPattern {
                    pattern_type: "Triple Bottom".to_string(),
                    sentiment: "Bullish".to_string(),
                    confidence: 0.75,
                    start_idx: t1.idx,
                    end_idx: t3.idx.max(n.saturating_sub(1)),
                    description: format!(
                        "Forming Triple Bottom at {:.2}, {:.2}, {:.2}. Strong support being tested.",
                        t1.price, t2.price, t3.price
                    ),
                    structural_bias: "Bullish Reversal".to_string(),
                    geometric_strictness: 0.85,
                    volume_validation: "Forming".to_string(),
                    breakout_status: "Forming".to_string(),
                    is_forming: true,
                    formation_progress: 0.90,
                    levels: PatternLevels::default(),
                });
            }
        }

        // Partial: 2 troughs at same level, price approaching for 3rd test
        if troughs.len() >= 2 {
            for window in troughs.windows(2) {
                let (t1, t2) = (window[0], window[1]);
                if !Self::prices_match(t1.price, t2.price, MATCH_TOLERANCE * 2.0) { continue; }
                let avg = (t1.price + t2.price) / 2.0;
                let proximity = (current_low - avg).abs() / avg.max(1e-9);
                if proximity < 0.03 {
                    out.push(ChartPattern {
                        pattern_type: "Triple Bottom".to_string(),
                        sentiment: "Bullish".to_string(),
                        confidence: 0.45,
                        start_idx: t1.idx,
                        end_idx: n.saturating_sub(1),
                        description: format!(
                            "Potential Triple Bottom: 2 troughs at {:.2} and {:.2}. Price approaching for 3rd test at {:.2}.",
                            t1.price, t2.price, current_low
                        ),
                        structural_bias: "Bullish Reversal".to_string(),
                        geometric_strictness: 1.0 - proximity,
                        volume_validation: "Forming".to_string(),
                        breakout_status: "Forming".to_string(),
                        is_forming: true,
                        formation_progress: 0.60,
                        levels: PatternLevels::default(),
                    });
                }
            }
        }
    }

    /// Forming triangles: Detect converging/diverging trendlines in recent swings.
    fn detect_forming_triangles(_candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let peaks: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Peak).map(|s| (s.idx as f64, s.price)).collect();
        let troughs: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Trough).map(|s| (s.idx as f64, s.price)).collect();
        if peaks.len() < 2 || troughs.len() < 2 { return; }

        let (res_slope, _) = Self::linear_regression(&peaks);
        let (sup_slope, _) = Self::linear_regression(&troughs);

        let start = swings.first().map(|s| s.idx).unwrap_or(0);
        let end = swings.last().map(|s| s.idx).unwrap_or(0);
        let span = (end as f64 - start as f64).max(1.0);

        // Normalize slopes per candle to price range
        let price_range = peaks.iter().map(|p| p.1).chain(troughs.iter().map(|p| p.1))
            .fold(f64::MIN, f64::max) - peaks.iter().map(|p| p.1).chain(troughs.iter().map(|p| p.1))
            .fold(f64::MAX, f64::min);
        let norm_threshold = (price_range / span * 0.1).max(FLAT_SLOPE_THRESHOLD);

        // Ascending Triangle: flat resistance, rising support
        if res_slope.abs() < norm_threshold && sup_slope > norm_threshold {
            let convergence = sup_slope / norm_threshold.max(1e-9);
            out.push(ChartPattern {
                pattern_type: "Ascending Triangle".to_string(),
                sentiment: "Bullish".to_string(),
                confidence: (0.45 + convergence.min(3.0) * 0.1).min(0.80),
                start_idx: start,
                end_idx: end,
                description: format!("Forming Ascending Triangle: flat resistance with rising support. Bullish breakout likely."),
                structural_bias: "Bullish Breakout".to_string(),
                geometric_strictness: 0.85,
                volume_validation: "Forming".to_string(),
                breakout_status: "Forming".to_string(),
                is_forming: true,
                formation_progress: 0.70,
                levels: PatternLevels::default(),
            });
        }

        // Descending Triangle: flat support, falling resistance
        if sup_slope.abs() < norm_threshold && res_slope < -norm_threshold {
            let convergence = res_slope.abs() / norm_threshold.max(1e-9);
            out.push(ChartPattern {
                pattern_type: "Descending Triangle".to_string(),
                sentiment: "Bearish".to_string(),
                confidence: (0.45 + convergence.min(3.0) * 0.1).min(0.80),
                start_idx: start,
                end_idx: end,
                description: format!("Forming Descending Triangle: falling resistance with flat support. Bearish breakdown likely."),
                structural_bias: "Bearish Breakout".to_string(),
                geometric_strictness: 0.85,
                volume_validation: "Forming".to_string(),
                breakout_status: "Forming".to_string(),
                is_forming: true,
                formation_progress: 0.70,
                levels: PatternLevels::default(),
            });
        }

        // Symmetrical Triangle: converging trendlines
        if res_slope < -norm_threshold && sup_slope > norm_threshold {
            out.push(ChartPattern {
                pattern_type: "Symmetrical Triangle".to_string(),
                sentiment: "Neutral".to_string(),
                confidence: 0.55,
                start_idx: start,
                end_idx: end,
                description: format!("Forming Symmetrical Triangle: converging trendlines. Breakout direction determines bias."),
                structural_bias: "Bilateral Breakout".to_string(),
                geometric_strictness: 0.85,
                volume_validation: "Forming".to_string(),
                breakout_status: "Forming".to_string(),
                is_forming: true,
                formation_progress: 0.65,
                levels: PatternLevels::default(),
            });
        }
    }

    /// Forming wedges: Rising or falling wedge patterns in recent price action.
    fn detect_forming_wedges(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let peaks: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Peak).map(|s| (s.idx as f64, s.price)).collect();
        let troughs: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Trough).map(|s| (s.idx as f64, s.price)).collect();
        if peaks.len() < 2 || troughs.len() < 2 { return; }

        let (res_slope, _) = Self::linear_regression(&peaks);
        let (sup_slope, _) = Self::linear_regression(&troughs);

        let start = swings.first().map(|s| s.idx).unwrap_or(0);
        let end = swings.last().map(|s| s.idx).unwrap_or(0);
        let vol_slope = Self::get_volume_slope(candles, start, end);

        // Rising Wedge: both slopes positive, support steeper than resistance (converging up)
        if res_slope > 0.0 && sup_slope > 0.0 && sup_slope > res_slope {
            out.push(ChartPattern {
                pattern_type: "Rising Wedge".to_string(),
                sentiment: "Bearish".to_string(),
                confidence: 0.55,
                start_idx: start,
                end_idx: end,
                description: format!(
                    "Forming Rising Wedge: converging upward. {} volume. Bearish reversal expected.",
                    if vol_slope < 0.0 { "Declining" } else { "Steady" }
                ),
                structural_bias: "Bearish Reversal".to_string(),
                geometric_strictness: 0.80,
                volume_validation: if vol_slope < 0.0 { "Confirmed: Consolidation Drying".to_string() } else { "Forming".to_string() },
                breakout_status: "Forming".to_string(),
                is_forming: true,
                formation_progress: 0.65,
                levels: PatternLevels::default(),
            });
        }

        // Falling Wedge: both slopes negative, resistance steeper than support (converging down)
        if res_slope < 0.0 && sup_slope < 0.0 && res_slope < sup_slope {
            out.push(ChartPattern {
                pattern_type: "Falling Wedge".to_string(),
                sentiment: "Bullish".to_string(),
                confidence: 0.55,
                start_idx: start,
                end_idx: end,
                description: format!(
                    "Forming Falling Wedge: converging downward. {} volume. Bullish reversal expected.",
                    if vol_slope < 0.0 { "Declining" } else { "Steady" }
                ),
                structural_bias: "Bullish Reversal".to_string(),
                geometric_strictness: 0.80,
                volume_validation: if vol_slope < 0.0 { "Confirmed: Consolidation Drying".to_string() } else { "Forming".to_string() },
                breakout_status: "Forming".to_string(),
                is_forming: true,
                formation_progress: 0.65,
                levels: PatternLevels::default(),
            });
        }
    }

    /// Forming flags and pennants: Detects consolidation after a strong move.
    fn detect_forming_flags_pennants(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let n = candles.len();
        if n < 10 || swings.len() < 3 { return; }

        // Look for a strong pole in the earlier portion, followed by consolidation
        let consol_start = n.saturating_sub(10);
        let pole_end = consol_start;
        let pole_start = pole_end.saturating_sub(15).max(0);
        if pole_end <= pole_start + 3 { return; }

        let pole_candles = &candles[pole_start..pole_end];
        let pole_open = pole_candles.first().map(|c| c.open).unwrap_or(0.0);
        let pole_close = pole_candles.last().map(|c| c.close).unwrap_or(0.0);
        let pole_range = (pole_close - pole_open).abs();
        let avg_price = (pole_open + pole_close) / 2.0;

        if avg_price < 1e-9 || pole_range / avg_price < 0.01 { return; }

        let is_bullish_pole = pole_close > pole_open;

        // Consolidation swings
        let consol_swings: Vec<&SwingPoint> = swings.iter().filter(|s| s.idx >= consol_start).collect();
        if consol_swings.len() < 2 { return; }

        let consol_high = consol_swings.iter().filter(|s| s.kind == SwingKind::Peak).map(|s| s.price).fold(f64::MIN, f64::max);
        let consol_low = consol_swings.iter().filter(|s| s.kind == SwingKind::Trough).map(|s| s.price).fold(f64::MAX, f64::min);

        if consol_high <= consol_low { return; }
        let consol_range = consol_high - consol_low;
        let ratio = consol_range / pole_range;
        if ratio > 0.50 { return; } // Too wide for a flag/pennant

        let (pattern, sentiment, bias) = if is_bullish_pole {
            ("Bull Flag", "Bullish", "Bullish Continuation")
        } else {
            ("Bear Flag", "Bearish", "Bearish Continuation")
        };

        let vol_slope = Self::get_volume_slope(candles, consol_start, n - 1);

        out.push(ChartPattern {
            pattern_type: pattern.to_string(),
            sentiment: sentiment.to_string(),
            confidence: (0.50 + (1.0 - ratio) * 0.25).min(0.80),
            start_idx: pole_start,
            end_idx: n.saturating_sub(1),
            description: format!(
                "Forming {}: pole {:.2}→{:.2} ({:.1}% move), now consolidating. {} volume in consolidation.",
                pattern, pole_open, pole_close, (pole_range / avg_price) * 100.0,
                if vol_slope < 0.0 { "Declining" } else { "Steady" }
            ),
            structural_bias: bias.to_string(),
            geometric_strictness: 1.0 - ratio,
            volume_validation: if vol_slope < 0.0 { "Confirmed: Consolidation Drying".to_string() } else { "Forming".to_string() },
            breakout_status: "Forming".to_string(),
            is_forming: true,
            formation_progress: 0.60,
            levels: PatternLevels::default(),
        });
    }

    /// Forming harmonics: Detect partial X-A-B-C-D patterns.
    fn detect_forming_harmonics(candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let n_swings = swings.len();
        let n = candles.len();

        // Full 5-point harmonic in recent swings
        if n_swings >= 5 {
            let x = swings[n_swings - 5];
            let a = swings[n_swings - 4];
            let b = swings[n_swings - 3];
            let c = swings[n_swings - 2];
            let d = swings[n_swings - 1];

            if x.kind != a.kind && a.kind != b.kind && b.kind != c.kind && c.kind != d.kind {
                let is_bullish = x.kind == SwingKind::Trough;
                let xa = (a.price - x.price).abs();
                let ab = (b.price - a.price).abs();
                if xa > 1e-9 && ab > 1e-9 {
                    let ab_xa = ab / xa;
                    let ad_xa = (d.price - a.price).abs() / xa;
                    if let Some((name, _)) = Self::classify_harmonic(ab_xa, ad_xa) {
                        out.push(ChartPattern {
                            pattern_type: format!("Harmonic {}", name),
                            sentiment: if is_bullish { "Bullish".to_string() } else { "Bearish".to_string() },
                            confidence: 0.80,
                            start_idx: x.idx,
                            end_idx: d.idx.max(n.saturating_sub(1)),
                            description: format!(
                                "Forming Harmonic {}: X→A→B→C→D structure detected. D at {:.4} retracement of XA.",
                                name, ad_xa
                            ),
                            structural_bias: if is_bullish { "Bullish Reversal".to_string() } else { "Bearish Reversal".to_string() },
                            geometric_strictness: 0.90,
                            volume_validation: "Forming".to_string(),
                            breakout_status: "Forming".to_string(),
                            is_forming: true,
                            formation_progress: 0.90,
                            levels: PatternLevels::default(),
                        });
                    }
                }
            }
        }

        // Partial harmonic: X-A-B-C formed, D forming
        if n_swings >= 4 {
            let x = swings[n_swings - 4];
            let a = swings[n_swings - 3];
            let b = swings[n_swings - 2];
            let c = swings[n_swings - 1];

            if x.kind != a.kind && a.kind != b.kind && b.kind != c.kind {
                let xa = (a.price - x.price).abs();
                let ab = (b.price - a.price).abs();
                if xa > 1e-9 && ab > 1e-9 {
                    let ab_xa = ab / xa;
                    let current_price = candles.last().map(|c| c.close).unwrap_or(0.0);
                    let ad_xa_current = (current_price - a.price).abs() / xa;
                    let is_bullish = x.kind == SwingKind::Trough;

                    // Check if current D leg is approaching any harmonic ratio
                    let targets = [
                        (0.786, "Gartley"), (0.886, "Bat"), (1.272, "Butterfly"), (1.618, "Crab"),
                    ];
                    for (target, name) in targets {
                        if (ad_xa_current - target).abs() < 0.15 {
                            out.push(ChartPattern {
                                pattern_type: format!("Harmonic {}", name),
                                sentiment: if is_bullish { "Bullish".to_string() } else { "Bearish".to_string() },
                                confidence: 0.40,
                                start_idx: x.idx,
                                end_idx: n.saturating_sub(1),
                                description: format!(
                                    "Potential Harmonic {}: X→A→B→C formed (AB/XA={:.3}). D leg forming — approaching {:.3} retracement (current: {:.3}).",
                                    name, ab_xa, target, ad_xa_current
                                ),
                                structural_bias: if is_bullish { "Bullish Reversal".to_string() } else { "Bearish Reversal".to_string() },
                                geometric_strictness: 0.70,
                                volume_validation: "Forming".to_string(),
                                breakout_status: "Forming".to_string(),
                                is_forming: true,
                                formation_progress: 0.65,
                                levels: PatternLevels::default(),
                            });
                            break; // Only report the closest match
                        }
                    }
                }
            }
        }
    }

    /// Forming rectangle: Flat support and resistance in recent action.
    fn detect_forming_rectangle(_candles: &[Candle], swings: &[SwingPoint], out: &mut Vec<ChartPattern>) {
        let peaks: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Peak).map(|s| (s.idx as f64, s.price)).collect();
        let troughs: Vec<(f64, f64)> = swings.iter().filter(|s| s.kind == SwingKind::Trough).map(|s| (s.idx as f64, s.price)).collect();
        if peaks.len() < 2 || troughs.len() < 2 { return; }

        let (res_slope, _) = Self::linear_regression(&peaks);
        let (sup_slope, _) = Self::linear_regression(&troughs);

        if res_slope.abs() < FLAT_SLOPE_THRESHOLD && sup_slope.abs() < FLAT_SLOPE_THRESHOLD {
            let avg_res = peaks.iter().map(|p| p.1).sum::<f64>() / peaks.len() as f64;
            let avg_sup = troughs.iter().map(|p| p.1).sum::<f64>() / troughs.len() as f64;
            let channel_width = avg_res - avg_sup;
            let mid = (avg_res + avg_sup) / 2.0;

            if mid.abs() < 1e-9 || channel_width / mid < 0.003 { return; }

            let start = swings.first().map(|s| s.idx).unwrap_or(0);
            let end = swings.last().map(|s| s.idx).unwrap_or(0);

            out.push(ChartPattern {
                pattern_type: "Rectangle".to_string(),
                sentiment: "Neutral".to_string(),
                confidence: 0.55,
                start_idx: start,
                end_idx: end,
                description: format!(
                    "Forming Rectangle: support {:.2}, resistance {:.2}. Width {:.2} ({:.1}%). Breakout direction determines bias.",
                    avg_sup, avg_res, channel_width, (channel_width / mid) * 100.0
                ),
                structural_bias: "Bilateral Breakout".to_string(),
                geometric_strictness: 0.85,
                volume_validation: "Forming".to_string(),
                breakout_status: "Forming".to_string(),
                is_forming: true,
                formation_progress: 0.60,
                levels: PatternLevels::default(),
            });
        }
    }
}

// ── Task 11: breakout level, measured-move target, invalidation, status ──────
#[cfg(test)]
mod levels_tests {
    use super::*;
    use proptest::prelude::*;

    /// Flat bars (open = high = low = close) at volume 10: the detectors below are
    /// handed their swings directly, so only closes and a few volumes matter.
    fn bars(closes: &[f64]) -> Vec<Candle> {
        closes
            .iter()
            .map(|&c| Candle { open: c, high: c, low: c, close: c, volume: 10.0, taker_buy_volume: None })
            .collect()
    }

    /// `n` bars at `fill`, with `set` closes overriding by index.
    fn series(n: usize, fill: f64, set: &[(usize, f64)]) -> Vec<Candle> {
        let mut closes = vec![fill; n];
        for &(i, c) in set {
            closes[i] = c;
        }
        bars(&closes)
    }

    fn peak(idx: usize, price: f64) -> SwingPoint {
        SwingPoint { idx, price, kind: SwingKind::Peak }
    }

    fn trough(idx: usize, price: f64) -> SwingPoint {
        SwingPoint { idx, price, kind: SwingKind::Trough }
    }

    fn approx(a: Option<f64>, b: f64) -> bool {
        a.is_some_and(|a| (a - b).abs() < 1e-9)
    }

    fn status(l: &PatternLevels) -> &str {
        l.status.as_deref().unwrap_or("none")
    }

    #[test]
    fn measure_walks_the_closed_bars_through_each_status() {
        let flat = |_: usize| 100.0;
        let bull = |closes: &[f64]| ChartPatternEngine::measure(&bars(closes), 0, true, flat, 10.0, 95.0);
        assert_eq!(
            bull(&[98.0, 99.0, 99.5, 150.0]),
            PatternLevels {
                breakout_level: Some(100.0),
                target: Some(110.0),
                invalidation: Some(95.0),
                status: Some("pending".into()),
            },
            "the newest bar (150) is never read: it may still be forming"
        );
        assert_eq!(status(&bull(&[99.0, 101.0, 103.0, 0.0])), "confirmed");
        assert_eq!(status(&bull(&[99.0, 101.0, 110.0, 0.0])), "target_hit", "a close AT the target hits it");
        assert_eq!(status(&bull(&[99.0, 94.0, 101.0, 0.0])), "failed", "invalidated before the break");
        assert_eq!(status(&bull(&[101.0, 94.0, 0.0])), "failed", "invalidated after the break");

        let bear = ChartPatternEngine::measure(&bars(&[101.0, 99.0, 90.0, 0.0]), 0, false, flat, 10.0, 105.0);
        assert_eq!(status(&bear), "target_hit");
        assert_eq!(bear.target, Some(90.0));

        // A sloped line is read at the breakout bar: 99 < 100, 100.5 < 101, 103 > 102.
        let sloped = ChartPatternEngine::measure(
            &bars(&[99.0, 100.5, 103.0, 0.0]), 0, true, |k| 100.0 + k as f64, 10.0, 95.0,
        );
        assert_eq!(status(&sloped), "confirmed");
        assert_eq!(sloped.breakout_level, Some(102.0));
        assert_eq!(sloped.target, Some(112.0));

        // Degenerate geometry defines nothing.
        let closes = bars(&[99.0, 101.0, 103.0, 0.0]);
        assert_eq!(ChartPatternEngine::measure(&closes, 0, true, flat, 0.0, 95.0), PatternLevels::default());
        assert_eq!(
            ChartPatternEngine::measure(&closes, 0, true, |_| f64::NAN, 10.0, 95.0),
            PatternLevels::default()
        );
        assert_eq!(
            ChartPatternEngine::measure(&closes, 0, true, flat, 10.0, 101.0),
            PatternLevels::default(),
            "a bullish invalidation above the breakout level"
        );
        // `from` past the newest bar: nothing to walk, no panic.
        assert_eq!(status(&ChartPatternEngine::measure(&closes, 9, true, flat, 10.0, 95.0)), "pending");
    }

    #[test]
    fn double_top_measures_from_its_trough_and_hits_its_target() {
        let swings = [peak(10, 110.0), trough(20, 100.0), peak(30, 110.5)];
        let mut candles = series(36, 105.0, &[(31, 104.0), (32, 99.0), (33, 95.0), (34, 89.0), (35, 89.0)]);
        candles[10].volume = 20.0;
        let mut out = Vec::new();
        ChartPatternEngine::detect_double_top(&candles, &swings, &mut out);
        assert_eq!(out.len(), 1);
        let p = &out[0];
        assert_eq!(p.levels.breakout_level, Some(100.0));
        assert!(approx(p.levels.target, 89.75), "100 - (110.25 - 100): {:?}", p.levels.target);
        assert_eq!(p.levels.invalidation, Some(110.5), "the higher peak");
        assert_eq!(status(&p.levels), "target_hit", "89 closed through 89.75");
        // Confidence is the pre-Task-11 formula, untouched (the tier reads it).
        // f64 annotations: `.min` on an untyped float literal is E0689 (ambiguous numeric type).
        let strictness: f64 = 1.0 - 0.5 / 110.5;
        let depth: f64 = (110.25 - 100.0) / 110.25;
        assert_eq!(p.confidence, (0.55 + strictness * 0.25 + depth.min(0.2)).min(1.0));

        let mut failed = series(36, 105.0, &[(31, 104.0), (32, 111.0)]);
        failed[10].volume = 20.0;
        let mut out = Vec::new();
        ChartPatternEngine::detect_double_top(&failed, &swings, &mut out);
        assert_eq!(status(&out[0].levels), "failed", "111 closed above the 110.5 peak");
    }

    #[test]
    fn head_and_shoulders_breaks_its_neckline_with_the_head_as_invalidation() {
        let swings = [peak(10, 105.0), trough(15, 100.0), peak(20, 110.0), trough(25, 100.0), peak(30, 104.5)];
        let mut candles = series(35, 103.0, &[(31, 102.0), (32, 99.0), (33, 97.0)]);
        candles[10].volume = 20.0;
        let mut out = Vec::new();
        ChartPatternEngine::detect_head_and_shoulders(&candles, &swings, &mut out);
        assert_eq!(out.len(), 1);
        let l = &out[0].levels;
        assert_eq!(l.breakout_level, Some(100.0), "the flat neckline through both troughs");
        assert_eq!(l.target, Some(90.0), "the head's 10 above it, measured down");
        assert_eq!(l.invalidation, Some(110.0), "the head");
        assert_eq!(status(l), "confirmed", "99 closed below 100; 97 has not reached 90");
    }

    #[test]
    fn triple_top_neckline_is_the_lowest_trough_between() {
        let swings = [peak(10, 110.0), trough(15, 100.0), peak(20, 110.2), trough(25, 101.0), peak(30, 109.9)];
        let candles = series(33, 105.0, &[(31, 99.0)]);
        let mut out = Vec::new();
        ChartPatternEngine::detect_triple_top(&candles, &swings, &mut out);
        assert_eq!(out.len(), 1);
        let l = &out[0].levels;
        let avg = (110.0 + 110.2 + 109.9) / 3.0;
        assert_eq!(l.breakout_level, Some(100.0));
        assert!(approx(l.target, 100.0 - (avg - 100.0)), "{:?}", l.target);
        assert_eq!(l.invalidation, Some(110.2), "the highest peak");
        assert_eq!(status(l), "confirmed");
    }

    #[test]
    fn cup_and_handle_breaks_its_right_rim() {
        let swings = [peak(5, 110.0), trough(15, 95.0), peak(25, 110.5), trough(30, 108.0), peak(35, 109.5)];
        let candles = series(40, 112.0, &[(31, 109.0), (32, 111.0)]);
        let mut out = Vec::new();
        ChartPatternEngine::detect_cup_and_handle(&candles, &swings, &mut out);
        assert_eq!(out.len(), 1);
        let l = &out[0].levels;
        assert_eq!(l.breakout_level, Some(110.5));
        assert_eq!(l.target, Some(125.75), "110.5 + (110.25 - 95)");
        assert_eq!(l.invalidation, Some(108.0), "the handle's low");
        assert_eq!(status(l), "confirmed");
    }

    #[test]
    fn ascending_triangle_breaks_its_flat_top() {
        let swings = [
            peak(10, 110.0), trough(15, 100.0), peak(20, 110.0),
            trough(25, 102.0), peak(30, 110.0), trough(35, 104.0),
        ];
        let candles = series(40, 107.0, &[(36, 108.0), (37, 111.0), (38, 115.0)]);
        let mut out = Vec::new();
        ChartPatternEngine::detect_triangles(&candles, &swings, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pattern_type, "Ascending Triangle");
        let l = &out[0].levels;
        assert!(approx(l.breakout_level, 110.0), "{:?}", l.breakout_level);
        // Support 0.2k + 97 is 99 at the first swing (bar 10): height 110 - 99.
        assert!(approx(l.target, 121.0), "{:?}", l.target);
        assert_eq!(l.invalidation, Some(104.0), "the last swing low");
        assert_eq!(status(l), "confirmed");

        let hit = series(40, 107.0, &[(36, 108.0), (37, 111.0), (38, 121.5)]);
        let mut out = Vec::new();
        ChartPatternEngine::detect_triangles(&hit, &swings, &mut out);
        assert_eq!(status(&out[0].levels), "target_hit");
    }

    #[test]
    fn symmetrical_triangle_takes_the_direction_of_the_side_that_breaks() {
        // Resistance 111 - 0.1k, support 98 + 0.1k: 11 apart at the first swing (bar 10).
        let swings = [peak(10, 110.0), trough(20, 100.0), peak(30, 108.0), trough(40, 102.0)];
        let run = |set: &[(usize, f64)]| {
            let mut out = Vec::new();
            ChartPatternEngine::detect_triangles(&series(45, 104.0, set), &swings, &mut out);
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].pattern_type, "Symmetrical Triangle");
            assert_eq!(out[0].sentiment, "Neutral", "the tier keeps reading it as non-directional");
            out.remove(0).levels
        };

        let up = run(&[(41, 105.0), (42, 108.0), (43, 110.0)]);
        assert!(approx(up.breakout_level, 106.8), "{:?}", up.breakout_level);
        assert!(approx(up.target, 117.8), "{:?}", up.target);
        assert_eq!(up.invalidation, Some(102.0));
        assert_eq!(status(&up), "confirmed");

        let down = run(&[(41, 105.0), (42, 101.0), (43, 99.0)]);
        assert!(approx(down.breakout_level, 102.2), "{:?}", down.breakout_level);
        assert!(approx(down.target, 91.2), "{:?}", down.target);
        assert_eq!(down.invalidation, Some(108.0));
        assert_eq!(status(&down), "confirmed");

        let inside = run(&[]);
        assert_eq!(
            inside,
            PatternLevels { status: Some("pending".into()), ..Default::default() },
            "no side broken: no level is the breakout yet"
        );
    }

    #[test]
    fn bull_flag_measures_the_pole_from_its_upper_boundary() {
        let mut closes: Vec<f64> = (0..40)
            .map(|k| if k <= 10 { 100.0 } else if k <= 29 { 100.0 + (k - 10) as f64 * 20.0 / 19.0 } else { 120.0 })
            .collect();
        closes[38] = 122.0;
        closes[39] = 50.0; // the newest bar is never read
        let swings = [peak(31, 121.0), trough(33, 118.0), peak(35, 121.0), trough(37, 118.0)];
        let mut out = Vec::new();
        ChartPatternEngine::detect_flags_and_pennants(&bars(&closes), &swings, &mut out);
        assert_eq!(out.len(), 1, "flat boundaries: a flag, not a pennant");
        assert_eq!(out[0].pattern_type, "Bull Flag");
        let l = &out[0].levels;
        assert_eq!(l.breakout_level, Some(121.0));
        assert_eq!(l.target, Some(141.0), "the 20-point pole");
        assert_eq!(l.invalidation, Some(118.0), "the flag's last low");
        assert_eq!(status(l), "confirmed");
    }

    #[test]
    fn rising_wedge_breaks_its_support_line() {
        // Resistance 108 + 0.2k, support 98 + 0.3k: 9 apart at the first swing (bar 10).
        let swings = [peak(10, 110.0), trough(20, 104.0), peak(30, 114.0), trough(40, 110.0)];
        let candles = series(44, 111.0, &[(41, 109.0), (42, 105.0)]);
        let mut out = Vec::new();
        ChartPatternEngine::detect_rising_wedge(&candles, &swings, &mut out);
        assert_eq!(out.len(), 1);
        let l = &out[0].levels;
        assert!(approx(l.breakout_level, 110.3), "{:?}", l.breakout_level);
        assert!(approx(l.target, 101.3), "{:?}", l.target);
        assert_eq!(l.invalidation, Some(114.0), "the highest peak");
        assert_eq!(status(l), "confirmed");
    }

    #[test]
    fn quasimodo_carries_only_its_invalidation() {
        let swings = [trough(10, 100.0), peak(20, 110.0), trough(30, 95.0), peak(40, 112.0)];
        let mut candles = series(45, 105.0, &[]);
        candles[10].volume = 20.0;
        let mut out = Vec::new();
        ChartPatternEngine::detect_quasimodo(&candles, &swings, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].levels, PatternLevels { invalidation: Some(95.0), ..Default::default() });
    }

    #[test]
    fn levels_flatten_into_the_json_and_are_absent_when_unset() {
        let swings = [peak(10, 110.0), trough(20, 100.0), peak(30, 110.5)];
        let mut candles = series(36, 105.0, &[(31, 104.0), (32, 99.0), (33, 95.0), (34, 89.0), (35, 89.0)]);
        candles[10].volume = 20.0;
        let mut out = Vec::new();
        ChartPatternEngine::detect_double_top(&candles, &swings, &mut out);
        let p = out.remove(0);

        let json = serde_json::to_value(&p).unwrap();
        assert_eq!(json["breakout_level"], serde_json::json!(100.0));
        assert_eq!(json["invalidation"], serde_json::json!(110.5));
        assert_eq!(json["status"], "target_hit");
        assert!(json["target"].is_number());
        assert!(json.get("levels").is_none(), "flattened, not nested");
        let back: ChartPattern = serde_json::from_value(json).unwrap();
        assert_eq!(back.levels, p.levels);

        let mut plain = p.clone();
        plain.levels = PatternLevels::default();
        let json = serde_json::to_value(&plain).unwrap();
        for key in ["breakout_level", "target", "invalidation", "status"] {
            assert!(json.get(key).is_none(), "{key} is absent, not null");
        }
        let back: ChartPattern = serde_json::from_value(json).unwrap();
        assert_eq!(back.levels, PatternLevels::default());
    }

    /// A positive random walk with wicks and volumes.
    fn walk() -> impl Strategy<Value = Vec<Candle>> {
        proptest::collection::vec((-3.0f64..3.0, 0.0f64..1.5, 0.0f64..1.5, 1.0f64..100.0), 20..200).prop_map(|steps| {
            let mut prev = 100.0_f64;
            steps
                .into_iter()
                .map(|(step, up, dn, volume)| {
                    let open = prev;
                    let close = (prev + step).max(1.0);
                    prev = close;
                    Candle {
                        open,
                        high: open.max(close) + up,
                        low: (open.min(close) - dn).max(0.5),
                        close,
                        volume,
                        taker_buy_volume: None,
                    }
                })
                .collect()
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn levels_put_target_and_invalidation_on_opposite_sides(candles in walk()) {
            for p in ChartPatternEngine::analyze(&candles) {
                prop_assert!((0.0..=1.0).contains(&p.confidence), "{} confidence {}", p.pattern_type, p.confidence);
                if let Some(s) = &p.levels.status {
                    prop_assert!(["pending", "confirmed", "failed", "target_hit"].contains(&s.as_str()), "{s}");
                }
                if let (Some(b), Some(t), Some(i)) = (p.levels.breakout_level, p.levels.target, p.levels.invalidation) {
                    prop_assert!(
                        (t > b && i < b) || (t < b && i > b),
                        "{}: breakout {b}, target {t}, invalidation {i}", p.pattern_type
                    );
                }
            }
        }
    }
}
