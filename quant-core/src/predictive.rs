// quant/predictive.rs — Dual Predictive Math Engine
//
// Strat Ai — Phase 1: Dual-Engine Predictive System
//
// Bundles two complementary regression models:
//
//   1. **Linear OLS** — Standard Ordinary Least Squares regression over
//      array indices (not timestamps, to avoid float overflow). Provides
//      a clean baseline trend direction. Anchored to the last close price.
//
//   2. **Curved VWEPR** — Volume-Weighted Exponential Polynomial Regression.
//      Delegates to the existing `vwepr` module. Captures price curvature
//      (acceleration/deceleration) that linear regression misses.
//
// Both produce `ProjectedPoint` arrays with identical anchor semantics
// (first point = last actual close price & time), making them directly
// overlayable on the same Lightweight Charts instance.
//
// The `ProjectionPayload` struct bundles both datasets plus the VWEPR
// acceleration coefficient for downstream AI analysis (e.g., DeepSeek
// conviction scoring can weight its narrative based on whether price
// is accelerating or decelerating).

use super::vwepr::{self, OhlcCandle, ProjectedPoint};

// ── Dual-Engine Output ──────────────────────────────────────────────────────

/// Combined output of both predictive engines, ready for IPC serialization.
///
/// The frontend can render both lines simultaneously:
///   - `linear_points` as a subtle baseline trend indicator
///   - `curved_points` as the primary Ghost Line (VWEPR)
///
/// The `acceleration_coefficient` is the quadratic term `a` from the VWEPR
/// polynomial `y = ax² + bx + c`:
///   - `a > 0` → price is accelerating upward (bullish momentum)
///   - `a < 0` → price is accelerating downward (bearish momentum)
///   - `a ≈ 0` → trend is approximately linear
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct ProjectionPayload {
    /// OLS linear regression projection points (unweighted baseline).
    pub linear_points: Vec<ProjectedPoint>,
    /// VWLR — Volume-Weighted Linear Regression points. An "advanced OLS":
    /// still a straight-line fit, but each bar is weighted by its traded
    /// volume so high-participation bars dominate the trend estimate.
    pub volume_points: Vec<ProjectedPoint>,
    /// VWEPR curved polynomial projection points.
    pub curved_points: Vec<ProjectedPoint>,
    /// The quadratic coefficient `a` from the VWEPR fit.
    /// Injected into DeepSeek prompts as a momentum signal.
    pub acceleration_coefficient: f64,
}

// ── Constants ───────────────────────────────────────────────────────────────

/// Maximum lookback window for OLS regression. Kept identical to VWEPR's
/// `MAX_WINDOW` (50) so every engine — OLS, VWLR and VWEPR — fits over the
/// exact same set of bars and the three lines stay directly comparable.
const OLS_MAX_WINDOW: usize = 50;

// ── Linear OLS Regression ───────────────────────────────────────────────────

/// Calculate a linear OLS regression over recent candles and project forward.
///
/// Uses **array indices** (0, 1, 2, …) as the x-axis — NOT raw Unix
/// timestamps — to prevent float accumulator overflow that causes
/// NaN/Infinity slopes (the exact bug that plagued the old React OLS).
///
/// # Arguments
/// * `candles`           — Slice of OHLCV candles (chronological order).
/// * `projection_length` — Number of future bars to project.
/// * `interval_sec`      — Duration of one bar in seconds.
///
/// # Returns
/// `Vec<ProjectedPoint>` with `projection_length + 1` entries:
///   - Index 0 = anchor (last candle's actual close + time).
///   - Index 1..=projection_length = future projected points.
///
/// Returns an empty `Vec` when insufficient data prevents a meaningful fit.
pub fn calculate_ols(
    candles: &[OhlcCandle],
    projection_length: usize,
    interval_sec: i64,
) -> Vec<ProjectedPoint> {
    if candles.is_empty() || projection_length == 0 {
        return Vec::new();
    }

    // ── 1. Window selection ─────────────────────────────────────────────────
    let window_size = candles.len().min(OLS_MAX_WINDOW);
    let window = &candles[candles.len() - window_size..];
    let n = window_size as f64;

    // ── 2. Accumulate OLS sums (x = array index) ────────────────────────────
    let mut sum_x: f64 = 0.0;
    let mut sum_y: f64 = 0.0;
    let mut sum_xy: f64 = 0.0;
    let mut sum_xx: f64 = 0.0;

    for (i, candle) in window.iter().enumerate() {
        let x = i as f64;
        let y = candle.close;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    // ── 3. Solve for slope and intercept ─────────────────────────────────────
    let denom = n * sum_xx - sum_x * sum_x;

    if denom.abs() < 1e-30 {
        // Degenerate case (e.g., single candle or all same x).
        // Return only the anchor point.
        let last = &window[window.len() - 1];
        return vec![ProjectedPoint {
            time: last.time,
            value: last.close,
        }];
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;

    // Guard: slope must be finite.
    if !slope.is_finite() {
        let last = &window[window.len() - 1];
        return vec![ProjectedPoint {
            time: last.time,
            value: last.close,
        }];
    }

    // ── 4. Anchored Projection ──────────────────────────────────────────────
    //
    // Like VWEPR, we anchor to the last actual close price (NOT the
    // regression's fitted value at the last index). This prevents the
    // visible vertical detachment that occurs when the regression line
    // doesn't pass exactly through the last data point.
    let last_candle = &window[window.len() - 1];
    let last_close = last_candle.close;
    let last_time = last_candle.time;

    let mut result = Vec::with_capacity(projection_length + 1);

    // Anchor point: exact last close
    result.push(ProjectedPoint {
        time: last_time,
        value: last_close,
    });

    // Project forward: each step adds slope * k to the anchor price
    for k in 1..=projection_length {
        let future_time = last_time + (k as i64 * interval_sec);
        let projected_value = last_close + slope * k as f64;

        result.push(ProjectedPoint {
            time: future_time,
            value: projected_value,
        });
    }

    result
}

/// The coefficient of determination (R²) of the SAME linear fit `calculate_ols`
/// projects — i.e. the fraction of close-price variance the straight line explains,
/// over the identical trailing `OLS_MAX_WINDOW` window. `None` when the fit is
/// degenerate and no honest number exists.
///
/// This is the missing half of the projection. `calculate_ols` returns WHERE the line
/// points; nothing returned HOW WELL it fits, so the only thing a caller could reach for
/// as a "confidence" was the size of the projected step. `tool-server`'s
/// `build_projection` did exactly that: `|slope| / price / 1%`, which is a move
/// MAGNITUDE, not a confidence. Because a regression slope is an average per-bar drift,
/// realistic values sat two to four orders of magnitude under that 1% normaliser — a
/// perfectly ordinary 7-points-per-bar trend on a 23,000 index reported as "0.03
/// confidence". Measured across three live sessions: 0.006, 0.00064, 0.03. The agent
/// correctly discounted all of them to zero weight, so `get_prediction` contributed
/// nothing to any analysis. R² is bounded [0, 1] by construction and actually answers
/// the question the field name asks.
///
/// # Numerics
///
/// Deliberately the CENTERED two-pass form (subtract the means, then accumulate second
/// moments) rather than the algebraically equivalent one-pass `n·Σy² − (Σy)²`. Index
/// closes are large relative to their variance — NIFTY near 23,000 over 50 bars makes
/// `n·Σy²` about `1.3e12` while the quantity being extracted from it is of order `1e5` —
/// so the one-pass form differences two nearly-equal large sums. At these magnitudes the
/// loss is real but not fatal (f64 leaves roughly `3e-4` of absolute error against a
/// `1e5` target); it degrades as the market gets quieter, and the two-pass form is
/// unconditionally stable for two extra passes over at most 50 candles. Cheap insurance
/// rather than a fix for a measured failure.
///
/// # Returns
/// * `Some(r2)` in `[0.0, 1.0]` for a well-posed fit over 3 or more candles.
/// * `None` when there are fewer than 3 candles (a line through 2 points fits
///   perfectly, so R² would be 1.0 — maximum confidence from no evidence), when the
///   closes are all identical (no variance to explain), or when any close is non-finite.
pub fn ols_fit_quality(candles: &[OhlcCandle]) -> Option<f64> {
    if candles.is_empty() {
        return None;
    }
    // The same window `calculate_ols` fits, so this R² describes THAT line and cannot
    // drift from it.
    let window_size = candles.len().min(OLS_MAX_WINDOW);
    if window_size < 3 {
        return None;
    }
    let window = &candles[candles.len() - window_size..];
    let n = window_size as f64;

    // Pass 1: means.
    let mut sum_x: f64 = 0.0;
    let mut sum_y: f64 = 0.0;
    for (i, candle) in window.iter().enumerate() {
        if !candle.close.is_finite() {
            return None;
        }
        sum_x += i as f64;
        sum_y += candle.close;
    }
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    // Pass 2: centered second moments.
    let mut sxx: f64 = 0.0;
    let mut syy: f64 = 0.0;
    let mut sxy: f64 = 0.0;
    for (i, candle) in window.iter().enumerate() {
        let dx = i as f64 - mean_x;
        let dy = candle.close - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    // `sxx` is fixed by the window length so it can only vanish on a degenerate window;
    // `syy` vanishes on a perfectly flat series, where "how much variance does the trend
    // explain" has no answer rather than the answer 0.
    if sxx < 1e-30 || syy < 1e-30 {
        return None;
    }

    let r2 = (sxy * sxy) / (sxx * syy);
    if !r2.is_finite() {
        return None;
    }
    // Algebraically already in [0, 1]; clamped against float drift at the boundaries.
    Some(r2.clamp(0.0, 1.0))
}

// ── Volume-Weighted Linear Regression (VWLR) ────────────────────────────────

/// Calculate a **volume-weighted** linear regression over recent candles and
/// project forward. This is the "advanced OLS": identical straight-line model,
/// but each observation is weighted by its traded volume (weighted least
/// squares), so bars with heavy participation pull the trend line more than
/// thin, low-conviction bars. Sits between plain OLS (unweighted line) and
/// VWEPR (volume-weighted curve).
///
/// Uses array indices as the x-axis (like OLS) to avoid float overflow, and
/// anchors the projection to the last actual close price.
///
/// Returns `Vec<ProjectedPoint>` with `projection_length + 1` entries (index 0
/// = anchor), or an empty `Vec` when the fit is degenerate.
pub fn calculate_vwlr(
    candles: &[OhlcCandle],
    projection_length: usize,
    interval_sec: i64,
) -> Vec<ProjectedPoint> {
    if candles.is_empty() || projection_length == 0 {
        return Vec::new();
    }

    let window_size = candles.len().min(OLS_MAX_WINDOW);
    let window = &candles[candles.len() - window_size..];

    // Weighted sums (weight = volume, floored at 1.0 so zero-volume bars still
    // contribute a minimal amount instead of vanishing).
    let mut sw: f64 = 0.0;
    let mut swx: f64 = 0.0;
    let mut swy: f64 = 0.0;
    let mut swxx: f64 = 0.0;
    let mut swxy: f64 = 0.0;

    for (i, candle) in window.iter().enumerate() {
        let x = i as f64;
        let y = candle.close;
        let w = candle.volume.max(1.0);
        sw += w;
        swx += w * x;
        swy += w * y;
        swxx += w * x * x;
        swxy += w * x * y;
    }

    let last_candle = &window[window.len() - 1];
    let last_close = last_candle.close;
    let last_time = last_candle.time;

    let denom = sw * swxx - swx * swx;
    if denom.abs() < 1e-30 {
        return vec![ProjectedPoint { time: last_time, value: last_close }];
    }

    let slope = (sw * swxy - swx * swy) / denom;
    if !slope.is_finite() {
        return vec![ProjectedPoint { time: last_time, value: last_close }];
    }

    let mut result = Vec::with_capacity(projection_length + 1);
    result.push(ProjectedPoint { time: last_time, value: last_close });
    for k in 1..=projection_length {
        result.push(ProjectedPoint {
            time: last_time + (k as i64 * interval_sec),
            value: last_close + slope * k as f64,
        });
    }
    result
}

// ── Dual-Engine Unified API ─────────────────────────────────────────────────

/// Run both predictive engines and return the bundled `ProjectionPayload`.
///
/// This is the primary entry point for the IPC command layer. It runs
/// OLS and VWEPR in sequence (both are pure CPU, sub-microsecond),
/// bundles the results, and returns them in a single IPC response.
///
/// # Arguments
/// * `candles`           — Slice of OHLCV candles (chronological order).
/// * `projection_length` — Number of future bars to project.
/// * `interval_sec`      — Duration of one bar in seconds.
pub fn calculate_dual_projection(
    candles: &[OhlcCandle],
    projection_length: usize,
    interval_sec: i64,
) -> ProjectionPayload {
    let linear_points = calculate_ols(candles, projection_length, interval_sec);
    let volume_points = calculate_vwlr(candles, projection_length, interval_sec);
    let (curved_points, acceleration_coefficient) =
        vwepr::calculate_vwepr_with_accel(candles, projection_length, interval_sec);

    ProjectionPayload {
        linear_points,
        volume_points,
        curved_points,
        acceleration_coefficient,
    }
}

// ── Unit Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a candle with minimal fields.
    fn candle(time: i64, close: f64, volume: f64) -> OhlcCandle {
        OhlcCandle {
            time,
            open: close,
            high: close,
            low: close,
            close,
            volume,
        }
    }

    // ── OLS Tests ───────────────────────────────────────────────────────────

    #[test]
    fn ols_empty_returns_empty() {
        let result = calculate_ols(&[], 5, 60);
        assert!(result.is_empty());
    }

    #[test]
    fn ols_zero_projection_returns_empty() {
        let candles = vec![candle(1000, 100.0, 500.0)];
        let result = calculate_ols(&candles, 0, 60);
        assert!(result.is_empty());
    }

    #[test]
    fn ols_anchor_matches_last_close() {
        let candles: Vec<OhlcCandle> = (0..25)
            .map(|i| candle(1000 + i * 60, 100.0 + i as f64 * 0.5, 1000.0))
            .collect();

        let result = calculate_ols(&candles, 5, 60);
        assert!(!result.is_empty());
        assert_eq!(result[0].time, candles.last().unwrap().time);
        assert!(
            (result[0].value - candles.last().unwrap().close).abs() < 1e-10,
            "OLS anchor must equal last close"
        );
    }

    #[test]
    fn ols_linear_data_extrapolates_perfectly() {
        // Perfectly linear data: close = 100 + 2*i
        let candles: Vec<OhlcCandle> = (0..30)
            .map(|i| candle(1000 + i * 60, 100.0 + 2.0 * i as f64, 500.0))
            .collect();

        let result = calculate_ols(&candles, 5, 60);
        assert_eq!(result.len(), 6); // 1 anchor + 5 projected

        let last_close = candles.last().unwrap().close; // 100 + 2*29 = 158.0
        for k in 1..=5 {
            let expected = last_close + 2.0 * k as f64;
            assert!(
                (result[k].value - expected).abs() < 1e-8,
                "OLS on linear data should extrapolate perfectly: step {}, got {}, expected {}",
                k, result[k].value, expected
            );
        }
    }

    #[test]
    fn ols_flat_projects_flat() {
        let candles: Vec<OhlcCandle> = (0..20)
            .map(|i| candle(1000 + i * 60, 50.0, 1000.0))
            .collect();

        let result = calculate_ols(&candles, 5, 60);
        for point in &result {
            assert!(
                (point.value - 50.0).abs() < 1e-8,
                "Flat data should project flat: got {}",
                point.value
            );
        }
    }

    #[test]
    fn ols_timestamps_are_sequential() {
        let candles: Vec<OhlcCandle> = (0..20)
            .map(|i| candle(5000 + i * 300, 200.0 + i as f64, 500.0))
            .collect();

        let interval = 300_i64;
        let proj_len = 8;
        let result = calculate_ols(&candles, proj_len, interval);

        assert_eq!(result.len(), proj_len + 1);
        let last_time = candles.last().unwrap().time;
        for (idx, point) in result.iter().enumerate() {
            let expected = last_time + (idx as i64 * interval);
            assert_eq!(point.time, expected, "OLS point {} wrong timestamp", idx);
        }
    }

    #[test]
    fn ols_single_candle_returns_anchor() {
        let candles = vec![candle(3000, 75.0, 100.0)];
        let result = calculate_ols(&candles, 5, 60);
        assert!(!result.is_empty());
        assert_eq!(result[0].time, 3000);
        assert!((result[0].value - 75.0).abs() < 1e-10);
    }

    // ── OLS fit-quality (R²) Tests ──────────────────────────────────────────

    #[test]
    fn fit_quality_of_a_perfect_line_is_one() {
        let candles: Vec<OhlcCandle> = (0..30)
            .map(|i| candle(1000 + i * 60, 100.0 + 2.0 * i as f64, 500.0))
            .collect();

        let r2 = ols_fit_quality(&candles).expect("a clean line has a measurable fit");
        assert!(
            (r2 - 1.0).abs() < 1e-9,
            "a perfectly linear series must explain all variance, got {r2}"
        );
    }

    #[test]
    fn fit_quality_of_a_trendless_zigzag_is_near_zero() {
        // Alternating 100 / 101: real variance, no trend to explain.
        let candles: Vec<OhlcCandle> = (0..20)
            .map(|i| candle(1000 + i * 60, if i % 2 == 0 { 100.0 } else { 101.0 }, 500.0))
            .collect();

        let r2 = ols_fit_quality(&candles).expect("a zigzag still has y-variance");
        assert!(
            r2 < 0.05,
            "chop must not read as a confident trend, got {r2}"
        );
    }

    #[test]
    fn fit_quality_survives_index_price_scale() {
        // THE case the old magnitude formula scored at 0.03: a gentle, clearly real trend
        // on a large base. 0.5 points per bar on 23,000 with a small deterministic wobble
        // so the fit is not trivially perfect. This is also the numerically awkward shape
        // for R² — tiny variance sitting on a huge mean — so it guards the centered
        // accumulation at the same time.
        let candles: Vec<OhlcCandle> = (0..50)
            .map(|i| {
                let wobble = (i % 3) as f64 - 1.0; // -1, 0, +1 repeating
                candle(1000 + i * 600, 23_000.0 + 0.5 * i as f64 + wobble * 0.2, 0.0)
            })
            .collect();

        let r2 = ols_fit_quality(&candles).expect("a trending index has a measurable fit");
        assert!(
            r2 > 0.9,
            "a clean trend on an index-scale base must report high confidence, got {r2}"
        );
    }

    #[test]
    fn fit_quality_of_a_flat_series_is_none() {
        // No variance to explain: R² is 0/0, so there is no honest number to return.
        let candles: Vec<OhlcCandle> = (0..20)
            .map(|i| candle(1000 + i * 60, 50.0, 1000.0))
            .collect();

        assert!(ols_fit_quality(&candles).is_none());
    }

    #[test]
    fn fit_quality_needs_at_least_three_candles() {
        // A line through 2 points fits perfectly, so R² would be 1.0 — maximum
        // confidence from no evidence. That must be refused, not reported.
        assert!(ols_fit_quality(&[]).is_none());
        assert!(ols_fit_quality(&[candle(0, 10.0, 1.0)]).is_none());
        assert!(ols_fit_quality(&[candle(0, 10.0, 1.0), candle(60, 11.0, 1.0)]).is_none());

        let three = vec![
            candle(0, 10.0, 1.0),
            candle(60, 11.0, 1.0),
            candle(120, 12.5, 1.0),
        ];
        assert!(ols_fit_quality(&three).is_some(), "three candles is enough");
    }

    #[test]
    fn fit_quality_rejects_a_non_finite_close() {
        let mut candles: Vec<OhlcCandle> = (0..10)
            .map(|i| candle(1000 + i * 60, 100.0 + i as f64, 500.0))
            .collect();
        candles[4].close = f64::NAN;

        assert!(
            ols_fit_quality(&candles).is_none(),
            "a NaN close must not silently produce a fit quality"
        );
    }

    #[test]
    fn fit_quality_reads_only_the_trailing_window() {
        // Wild history before the last OLS_MAX_WINDOW bars must not drag the score of a
        // clean recent trend, since this must describe the SAME window calculate_ols fits.
        let mut candles: Vec<OhlcCandle> = (0..40)
            .map(|i| candle(i * 60, if i % 2 == 0 { 10.0 } else { 900.0 }, 1.0))
            .collect();
        candles.extend(
            (0..OLS_MAX_WINDOW).map(|i| candle((40 + i as i64) * 60, 500.0 + i as f64, 1.0)),
        );

        let r2 = ols_fit_quality(&candles).expect("the recent window is well posed");
        assert!(
            (r2 - 1.0).abs() < 1e-9,
            "only the trailing window may count, got {r2}"
        );
    }

    #[test]
    fn fit_quality_is_direction_agnostic() {
        // R² measures fit, not sign: a clean downtrend is as confident as a clean uptrend.
        let up: Vec<OhlcCandle> = (0..30)
            .map(|i| candle(i * 60, 100.0 + 1.5 * i as f64, 1.0))
            .collect();
        let down: Vec<OhlcCandle> = (0..30)
            .map(|i| candle(i * 60, 100.0 - 1.5 * i as f64, 1.0))
            .collect();

        let r_up = ols_fit_quality(&up).unwrap();
        let r_down = ols_fit_quality(&down).unwrap();
        assert!((r_up - r_down).abs() < 1e-12, "{r_up} vs {r_down}");
    }

    // ── Dual-Engine Tests ───────────────────────────────────────────────────

    #[test]
    fn dual_produces_both_outputs() {
        let candles: Vec<OhlcCandle> = (0..30)
            .map(|i| candle(1000 + i * 60, 100.0 + i as f64 * 0.3, 500.0))
            .collect();

        let payload = calculate_dual_projection(&candles, 6, 60);

        // Both engines should produce output
        assert!(!payload.linear_points.is_empty());
        assert!(!payload.curved_points.is_empty());

        // Both should have 7 points (1 anchor + 6 projected)
        assert_eq!(payload.linear_points.len(), 7);
        assert_eq!(payload.curved_points.len(), 7);

        // Acceleration coefficient should be finite
        assert!(
            payload.acceleration_coefficient.is_finite(),
            "Acceleration coefficient must be finite"
        );
    }

    #[test]
    fn dual_anchors_match() {
        let candles: Vec<OhlcCandle> = (0..25)
            .map(|i| candle(1000 + i * 60, 150.0 + i as f64, 800.0))
            .collect();

        let payload = calculate_dual_projection(&candles, 5, 60);
        let last_close = candles.last().unwrap().close;
        let last_time = candles.last().unwrap().time;

        // Both engines must anchor to the same point
        assert_eq!(payload.linear_points[0].time, last_time);
        assert_eq!(payload.curved_points[0].time, last_time);
        assert!(
            (payload.linear_points[0].value - last_close).abs() < 1e-10,
            "OLS anchor mismatch"
        );
        assert!(
            (payload.curved_points[0].value - last_close).abs() < 1e-10,
            "VWEPR anchor mismatch"
        );
    }

    #[test]
    fn dual_linear_data_produces_near_zero_acceleration() {
        // Perfectly linear data: VWEPR's quadratic term `a` should be ≈ 0.
        let candles: Vec<OhlcCandle> = (0..30)
            .map(|i| candle(1000 + i * 60, 100.0 + i as f64, 500.0))
            .collect();

        let payload = calculate_dual_projection(&candles, 5, 60);

        assert!(
            payload.acceleration_coefficient.abs() < 1e-6,
            "Linear data should produce near-zero acceleration: got {}",
            payload.acceleration_coefficient
        );
    }

    #[test]
    fn dual_quadratic_data_produces_positive_acceleration() {
        // Upward-curving data: close = 100 + 0.1*i²
        let candles: Vec<OhlcCandle> = (0..30)
            .map(|i| candle(1000 + i * 60, 100.0 + 0.1 * (i as f64).powi(2), 500.0))
            .collect();

        let payload = calculate_dual_projection(&candles, 5, 60);

        assert!(
            payload.acceleration_coefficient > 0.0,
            "Upward-curving data should produce positive acceleration: got {}",
            payload.acceleration_coefficient
        );
    }

    #[test]
    fn dual_curves_diverge_on_nonlinear_data() {
        // Quadratic data: the linear OLS and curved VWEPR should diverge.
        let candles: Vec<OhlcCandle> = (0..30)
            .map(|i| candle(1000 + i * 60, 100.0 + 0.05 * (i as f64).powi(2), 500.0))
            .collect();

        let payload = calculate_dual_projection(&candles, 10, 60);

        // At projection step 10, linear extrapolation should differ
        // from the quadratic curve.
        let linear_terminal = payload.linear_points.last().unwrap().value;
        let curved_terminal = payload.curved_points.last().unwrap().value;
        let divergence = (curved_terminal - linear_terminal).abs();

        assert!(
            divergence > 0.01,
            "OLS and VWEPR should diverge on quadratic data (divergence={})",
            divergence
        );
    }

    #[test]
    fn dual_empty_input() {
        let payload = calculate_dual_projection(&[], 5, 60);
        assert!(payload.linear_points.is_empty());
        assert!(payload.curved_points.is_empty());
        assert!((payload.acceleration_coefficient - 0.0).abs() < 1e-15);
    }

    #[test]
    fn payload_serializes_to_json() {
        let candles: Vec<OhlcCandle> = (0..20)
            .map(|i| candle(1000 + i * 60, 100.0 + i as f64, 500.0))
            .collect();

        let payload = calculate_dual_projection(&candles, 3, 60);
        let json = serde_json::to_string(&payload);
        assert!(json.is_ok(), "ProjectionPayload must serialize to JSON");

        let json_str = json.unwrap();
        assert!(json_str.contains("linear_points"));
        assert!(json_str.contains("curved_points"));
        assert!(json_str.contains("acceleration_coefficient"));
    }
}
