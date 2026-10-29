// quant-core/src/structure.rs — Market-structure engine.
//
// Reads a candle series the way a discretionary trader reads a chart, in five steps:
//   1. swing highs / lows: fractal pivots with `left` / `right` bars on each side
//      (the same detector `chart_patterns` uses — see `find_swings_with`);
//   2. a cleaned skeleton: strictly alternating high → low → high, with every leg
//      shorter than `min_leg_atr × ATR(14)` removed, because a swing smaller than one
//      bar of normal movement is noise, not structure;
//   3. HH / LH / HL / LL labels, each swing against the previous swing of its kind;
//   4. breaks of structure, walked bar by bar: a CLOSE beyond the most recent
//      confirmed swing high (low) is a bullish (bearish) break. It is a CHoCH (change
//      of character) when it reverses the direction of the previous break and a BOS
//      (break of structure) otherwise. A wick through the level is not a break;
//   5. the dealing range (the last swing high and low), where price sits in it, and
//      the EMA 21 / 50 / 200 stack.
//
// Pure: no I/O, no clock, deterministic. Every index is into the input slice; the
// caller maps indices to timestamps. Swings are judged with the whole series in hand,
// so this describes structure AS OF the last bar — it is not a bar-by-bar replay of
// what a trader could have known at each earlier bar.

use crate::chart_patterns::{ChartPatternEngine, SwingKind, SwingPoint};
use crate::patterns::Candle;
use crate::IndicatorState;

/// Tunables for one structure read.
#[derive(Debug, Clone, Copy)]
pub struct StructureConfig {
    /// Bars that must sit on each side of a pivot. Clamped to at least 1.
    pub left: usize,
    pub right: usize,
    /// Minimum leg, in ATR(14) multiples, for a swing to count as structure.
    pub min_leg_atr: f64,
}

impl Default for StructureConfig {
    /// 3/3 fractals: tight enough that the newest swing is only three bars old, with
    /// the 1×ATR leg filter doing the noise rejection a wider window would otherwise
    /// have to do by lagging.
    fn default() -> Self {
        Self { left: 3, right: 3, min_leg_atr: 1.0 }
    }
}

/// A swing in the cleaned skeleton, labelled against the previous swing of its kind.
#[derive(Debug, Clone, PartialEq)]
pub struct StructureSwing {
    pub idx: usize,
    pub price: f64,
    pub is_high: bool,
    /// `HH` / `LH` for highs, `HL` / `LL` for lows; `None` for the first of its kind
    /// (nothing to compare it with). An EQUAL high reads `LH` and an equal low `HL`:
    /// a level that failed to extend is not a higher high.
    pub label: Option<&'static str>,
}

/// One close beyond a confirmed swing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructureBreak {
    /// `BOS` or `CHoCH`.
    pub kind: &'static str,
    /// `bullish` (closed above a swing high) or `bearish` (below a swing low).
    pub direction: &'static str,
    /// The broken swing's price.
    pub level: f64,
    pub swing_idx: usize,
    /// The bar whose close broke it.
    pub break_idx: usize,
}

#[derive(Debug, Clone)]
pub struct MarketStructure {
    /// `up` / `down` = direction of the most recent break; `range` = no break in the
    /// window. `last_bos` / `last_choch` carry how old it is.
    pub trend: &'static str,
    /// The full cleaned skeleton, oldest first. Callers trim it.
    pub swings: Vec<StructureSwing>,
    pub last_bos: Option<StructureBreak>,
    pub last_choch: Option<StructureBreak>,
    /// The dealing range: the most recent swing high and swing low.
    pub range_high: Option<f64>,
    pub range_low: Option<f64>,
    pub ema_21: Option<f64>,
    pub ema_50: Option<f64>,
    pub ema_200: Option<f64>,
    /// EMA-21 change over the last 5 bars, in percent.
    pub ema_21_slope_pct: Option<f64>,
    /// `bullish` (close > 21 > 50 [> 200]), `bearish` (the mirror), `mixed`, or
    /// `unavailable` when EMA-21/50 cannot be computed.
    pub ema_stack: &'static str,
    pub atr_14: Option<f64>,
    pub close: f64,
}

/// Fractal swing detection: bar `i` is a high when no bar within `left` before or
/// `right` after has a high `>=` it (ties disqualify both), and a low symmetrically.
/// The last `right` bars can never be confirmed, by construction.
///
/// With `left == right == 5` this is, line for line, the loop `ChartPatternEngine`
/// used to inline (pinned by `tests::five_bar_window_matches_the_original_detector`).
pub(crate) fn find_swings_with(candles: &[Candle], left: usize, right: usize) -> Vec<SwingPoint> {
    let mut swings = Vec::new();
    let n = candles.len();
    if n < left + right + 1 {
        return swings;
    }

    for i in left..(n - right) {
        let mut is_high = true;
        let mut is_low = true;

        for j in (i - left)..=(i + right) {
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

/// Remove legs shorter than `min_leg` from an ALTERNATING skeleton, smallest first,
/// keeping it alternating.
///
/// An interior minimum leg is removed with both its endpoints (a high and a low, so
/// the neighbours on either side are still opposite kinds). An edge leg loses only its
/// outer point: the oldest swing at the start, the newest at the end — a fresh leg too
/// small to matter yet is not structure yet. A non-finite or non-positive `min_leg`
/// disables the filter (ATR unavailable).
///
/// ponytail: O(n²) in the swing count. A 500-bar window yields well under a hundred
/// swings; switch to a heap keyed on leg size if windows ever grow by 10×.
fn filter_minor_legs(mut swings: Vec<SwingPoint>, min_leg: f64) -> Vec<SwingPoint> {
    if !(min_leg.is_finite() && min_leg > 0.0) {
        return swings;
    }
    while swings.len() >= 2 {
        let mut smallest = 0usize;
        let mut smallest_leg = f64::INFINITY;
        for i in 0..swings.len() - 1 {
            let leg = (swings[i + 1].price - swings[i].price).abs();
            if leg < smallest_leg {
                smallest = i;
                smallest_leg = leg;
            }
        }
        if !(smallest_leg < min_leg) {
            break;
        }
        if smallest == 0 {
            swings.remove(0);
        } else if smallest + 2 == swings.len() {
            swings.pop();
        } else {
            swings.drain(smallest..smallest + 2);
        }
    }
    swings
}

/// The cleaned, alternating skeleton for `candles` under `cfg`, given its ATR.
/// `pub(crate)` so `levels` finds order blocks behind the same breaks reported here.
pub(crate) fn skeleton(candles: &[Candle], cfg: &StructureConfig, atr: Option<f64>) -> Vec<SwingPoint> {
    let raw = find_swings_with(candles, cfg.left.max(1), cfg.right.max(1));
    let alternating = ChartPatternEngine::alternate_swings(&raw);
    let min_leg = atr.map_or(f64::NAN, |a| a * cfg.min_leg_atr);
    filter_minor_legs(alternating, min_leg)
}

fn label_swings(skel: &[SwingPoint]) -> Vec<Option<&'static str>> {
    let mut last_high: Option<f64> = None;
    let mut last_low: Option<f64> = None;
    skel.iter()
        .map(|s| match s.kind {
            SwingKind::Peak => {
                let label = last_high.map(|p| if s.price > p { "HH" } else { "LH" });
                last_high = Some(s.price);
                label
            }
            SwingKind::Trough => {
                let label = last_low.map(|p| if s.price < p { "LL" } else { "HL" });
                last_low = Some(s.price);
                label
            }
        })
        .collect()
}

/// Every break of structure in the window, oldest first.
///
/// A swing becomes breakable once it is CONFIRMED — `right` bars after it — which is
/// the earliest bar a trader could have drawn it. The most recent confirmed high and
/// low are the live levels; each can be broken once, after which it is consumed until
/// a newer swing of that kind confirms. The first break of the window is a BOS: with
/// no prior break there is no character to change.
/// `pub(crate)` so `levels` takes its order blocks from these same breaks.
pub(crate) fn structure_breaks(candles: &[Candle], skel: &[SwingPoint], right: usize) -> Vec<StructureBreak> {
    let right = right.max(1);
    let mut out: Vec<StructureBreak> = Vec::new();
    let mut next = 0usize;
    let mut high: Option<SwingPoint> = None;
    let mut low: Option<SwingPoint> = None;
    let mut trend = 0i8;

    for (bar, candle) in candles.iter().enumerate() {
        while next < skel.len() && skel[next].idx + right <= bar {
            match skel[next].kind {
                SwingKind::Peak => high = Some(skel[next]),
                SwingKind::Trough => low = Some(skel[next]),
            }
            next += 1;
        }
        if let Some(h) = high {
            if candle.close > h.price {
                out.push(StructureBreak {
                    kind: if trend < 0 { "CHoCH" } else { "BOS" },
                    direction: "bullish",
                    level: h.price,
                    swing_idx: h.idx,
                    break_idx: bar,
                });
                trend = 1;
                high = None;
            }
        }
        if let Some(l) = low {
            if candle.close < l.price {
                out.push(StructureBreak {
                    kind: if trend > 0 { "CHoCH" } else { "BOS" },
                    direction: "bearish",
                    level: l.price,
                    swing_idx: l.idx,
                    break_idx: bar,
                });
                trend = -1;
                low = None;
            }
        }
    }
    out
}

/// Where `price` sits in the range `[low, high]`, in percent, and its zone:
/// `discount` below 45%, `equilibrium` 45–55%, `premium` above 55%, and
/// `below_range` / `above_range` outside it. `None` for a degenerate range.
pub fn range_position(high: f64, low: f64, price: f64) -> Option<(f64, &'static str)> {
    if !(high.is_finite() && low.is_finite() && price.is_finite()) || high <= low {
        return None;
    }
    let pct = (price - low) / (high - low) * 100.0;
    let zone = if pct > 100.0 {
        "above_range"
    } else if pct < 0.0 {
        "below_range"
    } else if (45.0..=55.0).contains(&pct) {
        "equilibrium"
    } else if pct > 55.0 {
        "premium"
    } else {
        "discount"
    };
    Some((pct, zone))
}

fn finite(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}

fn ema_stack(close: f64, e21: Option<f64>, e50: Option<f64>, e200: Option<f64>) -> &'static str {
    let (Some(a), Some(b)) = (e21, e50) else {
        return "unavailable";
    };
    // EMA-200 joins the stack only when there is enough history for it; a young
    // listing is judged on 21/50 rather than reported as having no stack.
    let bullish = close > a && a > b && e200.map_or(true, |z| b > z);
    let bearish = close < a && a < b && e200.map_or(true, |z| b < z);
    if bullish {
        "bullish"
    } else if bearish {
        "bearish"
    } else {
        "mixed"
    }
}

/// Analyze one candle series. `None` only for an empty series or a non-finite last
/// close; every other input (short, flat, gappy) yields a result whose fields are
/// `None` / empty where they could not be measured.
pub fn analyze_structure(candles: &[Candle], cfg: &StructureConfig) -> Option<MarketStructure> {
    let close = candles.last()?.close;
    if !close.is_finite() {
        return None;
    }

    let atr = finite(IndicatorState::compute_atr(candles, 14));
    let skel = skeleton(candles, cfg, atr);
    let labels = label_swings(&skel);
    let breaks = structure_breaks(candles, &skel, cfg.right);

    let trend = match breaks.last() {
        Some(b) if b.direction == "bullish" => "up",
        Some(_) => "down",
        None => "range",
    };

    let ema_21 = finite(IndicatorState::compute_ema(candles, 21));
    let ema_50 = finite(IndicatorState::compute_ema(candles, 50));
    let ema_200 = finite(IndicatorState::compute_ema(candles, 200));
    const SLOPE_BARS: usize = 5;
    let ema_21_slope_pct = if candles.len() >= 21 + SLOPE_BARS {
        let prior = finite(IndicatorState::compute_ema(&candles[..candles.len() - SLOPE_BARS], 21));
        match (ema_21, prior) {
            (Some(now), Some(then)) if then != 0.0 => Some((now - then) / then.abs() * 100.0),
            _ => None,
        }
    } else {
        None
    };

    Some(MarketStructure {
        trend,
        swings: skel
            .iter()
            .zip(labels)
            .map(|(s, label)| StructureSwing {
                idx: s.idx,
                price: s.price,
                is_high: s.kind == SwingKind::Peak,
                label,
            })
            .collect(),
        last_bos: breaks.iter().rev().find(|b| b.kind == "BOS").copied(),
        last_choch: breaks.iter().rev().find(|b| b.kind == "CHoCH").copied(),
        range_high: skel.iter().rev().find(|s| s.kind == SwingKind::Peak).map(|s| s.price),
        range_low: skel.iter().rev().find(|s| s.kind == SwingKind::Trough).map(|s| s.price),
        ema_21,
        ema_50,
        ema_200,
        ema_21_slope_pct,
        ema_stack: ema_stack(close, ema_21, ema_50, ema_200),
        atr_14: atr,
        close,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn candle(open: f64, high: f64, low: f64, close: f64) -> Candle {
        Candle { open, high, low, close, volume: 10.0, taker_buy_volume: None }
    }

    /// Candles that walk linearly between `pivots`, `per_leg` bars per leg. Each bar
    /// gets a wick that grows by 0.001 per bar index, so two bars meeting at a pivot
    /// never tie (a tie disqualifies both from being a swing) and the later one wins.
    fn path(pivots: &[f64], per_leg: usize) -> Vec<Candle> {
        let mut points = vec![pivots[0]];
        for w in pivots.windows(2) {
            for step in 1..=per_leg {
                points.push(w[0] + (w[1] - w[0]) * step as f64 / per_leg as f64);
            }
        }
        points
            .windows(2)
            .enumerate()
            .map(|(k, p)| {
                let eps = 0.1 + 0.001 * k as f64;
                candle(p[0], p[0].max(p[1]) + eps, p[0].min(p[1]) - eps, p[1])
            })
            .collect()
    }

    /// The pre-refactor body of `ChartPatternEngine::find_swings`, verbatim.
    fn reference_find_swings(candles: &[Candle]) -> Vec<SwingPoint> {
        const SWING_WINDOW: usize = 5;
        let mut swings = Vec::new();
        let n = candles.len();
        if n < SWING_WINDOW * 2 + 1 {
            return swings;
        }
        for i in SWING_WINDOW..(n - SWING_WINDOW) {
            let mut is_high = true;
            let mut is_low = true;
            for j in (i.saturating_sub(SWING_WINDOW))..=(i + SWING_WINDOW).min(n - 1) {
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

    #[test]
    fn uptrend_reads_up_with_higher_highs_and_higher_lows() {
        let candles = path(&[100.0, 110.0, 104.0, 116.0, 109.0, 122.0, 115.0, 125.0], 6);
        let ms = analyze_structure(&candles, &StructureConfig::default()).unwrap();
        assert_eq!(ms.trend, "up");
        let bos = ms.last_bos.expect("a bullish break");
        assert_eq!(bos.direction, "bullish");
        assert!((bos.level - 122.0).abs() < 0.5, "broke the 122 high, got {}", bos.level);
        assert!(ms.last_choch.is_none());
        let labels: Vec<_> = ms.swings.iter().filter_map(|s| s.label).collect();
        assert!(labels.iter().all(|l| *l == "HH" || *l == "HL"), "{labels:?}");
        assert!(labels.contains(&"HH") && labels.contains(&"HL"));
    }

    #[test]
    fn losing_the_last_higher_low_is_a_bearish_choch() {
        let candles = path(&[100.0, 110.0, 104.0, 116.0, 109.0, 122.0, 113.0, 118.0, 104.0], 6);
        let ms = analyze_structure(&candles, &StructureConfig::default()).unwrap();
        assert_eq!(ms.trend, "down");
        let choch = ms.last_choch.expect("a change of character");
        assert_eq!(choch.direction, "bearish");
        assert!((choch.level - 113.0).abs() < 0.5, "broke the 113 low, got {}", choch.level);
        assert!(ms.swings.iter().any(|s| s.label == Some("LH")), "118 is a lower high");
    }

    #[test]
    fn flat_and_tiny_series_degrade_without_panicking() {
        let flat: Vec<Candle> = (0..60).map(|_| candle(50.0, 50.0, 50.0, 50.0)).collect();
        let ms = analyze_structure(&flat, &StructureConfig::default()).unwrap();
        assert_eq!(ms.trend, "range");
        assert!(ms.swings.is_empty(), "ties disqualify every bar");
        assert!(ms.range_high.is_none() && ms.range_low.is_none());
        assert_eq!(ms.ema_stack, "mixed");

        let one = vec![candle(1.0, 1.1, 0.9, 1.0)];
        let ms = analyze_structure(&one, &StructureConfig::default()).unwrap();
        assert_eq!(ms.trend, "range");
        assert_eq!(ms.ema_stack, "unavailable");
        assert!(ms.atr_14.is_none());

        assert!(analyze_structure(&[], &StructureConfig::default()).is_none());
    }

    #[test]
    fn range_position_zones() {
        assert_eq!(range_position(110.0, 100.0, 105.0), Some((50.0, "equilibrium")));
        assert_eq!(range_position(110.0, 100.0, 108.0).map(|p| p.1), Some("premium"));
        assert_eq!(range_position(110.0, 100.0, 101.0).map(|p| p.1), Some("discount"));
        assert_eq!(range_position(110.0, 100.0, 111.0).map(|p| p.1), Some("above_range"));
        assert_eq!(range_position(110.0, 100.0, 99.0).map(|p| p.1), Some("below_range"));
        assert_eq!(range_position(100.0, 100.0, 100.0), None);
        assert_eq!(range_position(100.0, 110.0, 105.0), None);
    }

    /// A bounded random walk: some bars flat, wicks of random size, never below 0.5.
    fn walk() -> impl Strategy<Value = Vec<Candle>> {
        prop::collection::vec((-3.0f64..3.0, 0.0f64..2.0, 0.0f64..2.0, 0u8..5), 0..260).prop_map(
            |steps| {
                let mut price = 1000.0f64;
                steps
                    .into_iter()
                    .map(|(delta, up, down, flat)| {
                        let open = price;
                        let close = if flat == 0 { open } else { (open + delta).max(1.0) };
                        price = close;
                        candle(open, open.max(close) + up, (open.min(close) - down).max(0.5), close)
                    })
                    .collect()
            },
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn five_bar_window_matches_the_original_detector(candles in walk()) {
            let new = find_swings_with(&candles, 5, 5);
            let old = reference_find_swings(&candles);
            prop_assert_eq!(new.len(), old.len());
            for (a, b) in new.iter().zip(old.iter()) {
                prop_assert_eq!(a.idx, b.idx);
                prop_assert_eq!(a.price.to_bits(), b.price.to_bits());
                prop_assert_eq!(a.kind, b.kind);
            }
        }

        #[test]
        fn skeleton_alternates_and_every_leg_clears_the_atr_floor(candles in walk()) {
            let cfg = StructureConfig::default();
            let atr = finite(IndicatorState::compute_atr(&candles, 14));
            let skel = skeleton(&candles, &cfg, atr);
            for w in skel.windows(2) {
                prop_assert!(w[0].kind != w[1].kind, "skeleton must alternate");
                prop_assert!(w[0].idx <= w[1].idx);
                if let Some(a) = atr {
                    prop_assert!((w[1].price - w[0].price).abs() >= a * cfg.min_leg_atr);
                }
            }
        }

        #[test]
        fn labels_agree_with_swing_prices(candles in walk()) {
            let ms = analyze_structure(&candles, &StructureConfig::default());
            prop_assert_eq!(ms.is_none(), candles.is_empty());
            let Some(ms) = ms else { return Ok(()); };
            let mut prev_high: Option<f64> = None;
            let mut prev_low: Option<f64> = None;
            for s in &ms.swings {
                let prev = if s.is_high { prev_high } else { prev_low };
                match (prev, s.label) {
                    (None, None) => {}
                    (Some(p), Some("HH")) => prop_assert!(s.is_high && s.price > p),
                    (Some(p), Some("LH")) => prop_assert!(s.is_high && s.price <= p),
                    (Some(p), Some("LL")) => prop_assert!(!s.is_high && s.price < p),
                    (Some(p), Some("HL")) => prop_assert!(!s.is_high && s.price >= p),
                    other => prop_assert!(false, "inconsistent label {:?}", other),
                }
                if s.is_high { prev_high = Some(s.price) } else { prev_low = Some(s.price) }
            }
        }

        #[test]
        fn breaks_are_close_confirmed_beyond_a_confirmed_swing(candles in walk()) {
            let cfg = StructureConfig::default();
            let atr = finite(IndicatorState::compute_atr(&candles, 14));
            let skel = skeleton(&candles, &cfg, atr);
            let breaks = structure_breaks(&candles, &skel, cfg.right);
            let mut prev_dir: Option<&str> = None;
            for b in &breaks {
                let close = candles[b.break_idx].close;
                match b.direction {
                    "bullish" => prop_assert!(close > b.level),
                    "bearish" => prop_assert!(close < b.level),
                    other => prop_assert!(false, "bad direction {}", other),
                }
                prop_assert!(b.break_idx >= b.swing_idx + cfg.right, "broken before it was confirmed");
                let reversal = prev_dir.is_some_and(|d| d != b.direction);
                prop_assert_eq!(b.kind == "CHoCH", reversal);
                prev_dir = Some(b.direction);
            }
            let ms = analyze_structure(&candles, &cfg);
            if let Some(ms) = ms {
                let expected = match breaks.last() {
                    Some(b) if b.direction == "bullish" => "up",
                    Some(_) => "down",
                    None => "range",
                };
                prop_assert_eq!(ms.trend, expected);
            }
        }
    }
}
