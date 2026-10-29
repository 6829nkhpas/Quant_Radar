// quant/vwepr.rs — Volume-Weighted Exponential Polynomial Regression (VWEPR)
//
// Strat Ai — Phase 1: Rust Math Engine
//
// Fits a SECOND-degree polynomial (y = a₂·x² + a₁·x + a₀) to recent OHLCV
// candles, where each data point is weighted by:
//
//     w_i = volume_i × α^(window_size − 1 − i)
//
// A quadratic (not cubic) is used deliberately:
//   • It matches the pure-JS fallback engine (`vweprProjection` in
//     `ghostLineComputation.ts`), so the Tauri and browser paths agree.
//   • It matches the documented `acceleration_coefficient` semantics (the
//     quadratic term `a` of `y = ax² + bx + c`).
//   • A cubic term makes the forward extrapolation "fly off" the screen once
//     it leaves the fitted window — the exact broken-ghost symptom we're
//     fixing. Quadratic curvature extrapolates far more stably.
//
// The 3×3 normal-equation system is solved via Gaussian elimination with
// partial pivoting. No external linear-algebra crate required.
//
// The projection anchors to the last actual close price and extrapolates the
// *change* (delta) driven by the polynomial's slope and acceleration, avoiding
// "floating detachments" caused by using the raw intercept `a₀`.

// ── Output Struct ───────────────────────────────────────────────────────────

/// A single projected point on the Ghost Line curve.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct ProjectedPoint {
    /// UNIX timestamp in seconds.
    pub time: i64,
    /// Projected price value.
    pub value: f64,
}

// ── Input Struct ────────────────────────────────────────────────────────────

/// An OHLCV candle enriched with a UNIX timestamp.
///
/// This is the input contract for the VWEPR engine. It mirrors the data shape
/// returned by `BinaryCandle` (in `commands/charts.rs`) but uses seconds
/// for the timestamp and `f64` for volume, matching the quant module's
/// internal conventions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OhlcCandle {
    /// UNIX timestamp in seconds for this candle's open time.
    pub time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    /// Traded volume for this bar.
    pub volume: f64,
}

// ── Constants ───────────────────────────────────────────────────────────────

/// Maximum number of trailing candles to consider for the regression window.
/// 50 periods — identical to `predictive::OLS_MAX_WINDOW` so OLS, VWLR and
/// VWEPR all fit over the same bars and remain directly comparable.
pub const MAX_WINDOW: usize = 50;

/// Exponential decay factor. Closer to 1.0 = slower decay (more uniform
/// weighting). 0.90 over a 50-bar window means the oldest bar keeps a small
/// but non-zero pull (0.90^49 ≈ 0.0057), so recent volume spikes dominate the
/// fit while the tail still anchors the curve.
const ALPHA: f64 = 0.90;

// ── Public API ──────────────────────────────────────────────────────────────

/// Calculate the VWEPR curve: fit a weighted quadratic polynomial to recent
/// candles and project `projection_length` bars into the future.
///
/// # Arguments
/// * `candles`           — Slice of OHLCV candles (chronological order).
/// * `projection_length` — Number of future bars to project.
/// * `interval_sec`      — Duration of one bar in seconds (e.g., 300 for 5m).
///
/// # Returns
/// A `Vec<ProjectedPoint>` with length `projection_length + 1`:
///   - Index 0 = the anchor point (last candle's close + time).
///   - Index 1..=projection_length = future projected points.
///
/// Returns an empty `Vec` when insufficient data prevents a meaningful fit.
pub fn calculate_vwepr_curve(
    candles: &[OhlcCandle],
    projection_length: usize,
    interval_sec: i64,
) -> Vec<ProjectedPoint> {
    vwepr_core(candles, projection_length, interval_sec).0
}

/// Calculate the VWEPR curve and also return the acceleration coefficient `a`.
///
/// This is used by the dual-engine predictive module to inject the curvature
/// signal into downstream AI analysis (e.g., DeepSeek conviction scoring).
///
/// # Returns
/// `(projected_points, acceleration_coefficient)` — `a` is the quadratic
/// coefficient `a₂` from the polynomial fit.
/// Positive = accelerating upward, negative = accelerating downward, near-zero = linear trend.
///
/// Returns `(points, 0.0)` when the matrix is singular and no fit is possible.
pub fn calculate_vwepr_with_accel(
    candles: &[OhlcCandle],
    projection_length: usize,
    interval_sec: i64,
) -> (Vec<ProjectedPoint>, f64) {
    vwepr_core(candles, projection_length, interval_sec)
}

/// Internal core: fits the weighted quadratic polynomial and projects forward.
///
/// Returns `(projected_points, acceleration_coefficient)`.
fn vwepr_core(
    candles: &[OhlcCandle],
    projection_length: usize,
    interval_sec: i64,
) -> (Vec<ProjectedPoint>, f64) {
    if candles.is_empty() || projection_length == 0 {
        return (Vec::new(), 0.0);
    }

    // ── 1. Determine the regression window ──────────────────────────────────
    let window_size = candles.len().min(MAX_WINDOW);
    let window = &candles[candles.len() - window_size..];

    // ── 2. Accumulate weighted sums for the normal equations ────────────────
    //
    // We fit a quadratic polynomial: y = a₂·x² + a₁·x + a₀
    //
    // We solve:
    //   | s0  s1  s2 |   | a_0 |   | sy0 |
    //   | s1  s2  s3 | × | a_1 | = | sy1 |
    //   | s2  s3  s4 |   | a_2 |   | sy2 |
    //
    // where s_k = Σ w_i · x_i^k  and  sy_k = Σ w_i · y_i · x_i^k.

    let mut s0: f64 = 0.0; // Σ w
    let mut s1: f64 = 0.0; // Σ w·x
    let mut s2: f64 = 0.0; // Σ w·x²
    let mut s3: f64 = 0.0; // Σ w·x³
    let mut s4: f64 = 0.0; // Σ w·x⁴

    let mut sy0: f64 = 0.0; // Σ w·y
    let mut sy1: f64 = 0.0; // Σ w·x·y
    let mut sy2: f64 = 0.0; // Σ w·x²·y

    for (i, candle) in window.iter().enumerate() {
        // Weight = volume × exponential decay (recent bars weighted more).
        // Volume floor of 1.0 keeps zero-volume bars contributing a minimal
        // amount (so the fit degrades gracefully to an unweighted one when a
        // whole window has no volume) instead of collapsing the weights to 0.
        let vol = candle.volume.max(1.0);
        let decay = ALPHA.powi((window_size as i32) - 1 - (i as i32));
        let w = vol * decay;

        let x = i as f64;
        let y = candle.close;

        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x3 * x;

        s0 += w;
        s1 += w * x;
        s2 += w * x2;
        s3 += w * x3;
        s4 += w * x4;

        sy0 += w * y;
        sy1 += w * x * y;
        sy2 += w * x2 * y;
    }

    // ── 3. Solve via 3x3 Gaussian Elimination with Partial Pivoting ─────────
    let mat = [
        [s0, s1, s2],
        [s1, s2, s3],
        [s2, s3, s4],
    ];
    let rhs = [sy0, sy1, sy2];

    let coeffs = match solve_system_3x3(mat, rhs) {
        Some(x) => x,
        None => {
            // Singular or near-singular matrix — cannot solve. Return only the
            // anchor point so the UI always has at least the last known price.
            let last = &window[window.len() - 1];
            return (vec![ProjectedPoint {
                time: last.time,
                value: last.close,
            }], 0.0);
        }
    };

    let a_0 = coeffs[0]; // intercept
    let a_1 = coeffs[1]; // linear coefficient
    let a_2 = coeffs[2]; // quadratic coefficient (acceleration)

    // ── 4. Anchored Projection ──────────────────────────────────────────────
    //
    // CRITICAL: We do NOT use the raw polynomial value at future x. Instead,
    // we compute the *delta* from the polynomial's value at the last known
    // data index (n_index) and add it to the actual last close price. This
    // anchors the curve to reality and prevents "floating detachments".

    let last_candle = &window[window.len() - 1];
    let last_close = last_candle.close;
    let last_time = last_candle.time;
    let n_index = (window_size - 1) as f64; // x-coordinate of the last data point

    // Value of the fitted polynomial at the anchor point.
    let fitted_at_anchor = a_2 * n_index * n_index + a_1 * n_index + a_0;

    let mut result = Vec::with_capacity(projection_length + 1);

    // Push the anchor point — must exactly equal the last candle's actual close.
    result.push(ProjectedPoint {
        time: last_time,
        value: last_close,
    });

    // Project future bars.
    for i in 1..=projection_length {
        let future_x = n_index + i as f64;
        let future_time = last_time + (i as i64 * interval_sec);

        // Polynomial value at future_x.
        let fitted_at_future = a_2 * future_x * future_x + a_1 * future_x + a_0;

        // Delta = change in fitted value from the anchor to this future point.
        let delta = fitted_at_future - fitted_at_anchor;

        result.push(ProjectedPoint {
            time: future_time,
            value: last_close + delta,
        });
    }

    // For y = a₂·x² + a₁·x + a₀ the second derivative is a constant 2·a₂, so
    // the "acceleration" coefficient (analogous to a parabola's `a`) is a₂.
    (result, a_2)
}

// ── Linear Solver: 3×3 Gaussian Elimination with Partial Pivoting ───────────

/// Solves a 3x3 system of linear equations using Gaussian elimination with
/// partial pivoting.
///
/// Returns `Some([a_0, a_1, a_2])` on success, or `None` if the matrix is
/// singular.
fn solve_system_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    const N: usize = 3;
    for i in 0..N {
        // Find pivot.
        let mut max_row = i;
        for r in (i + 1)..N {
            if a[r][i].abs() > a[max_row][i].abs() {
                max_row = r;
            }
        }
        if a[max_row][i].abs() < 1e-30 {
            return None; // Singular matrix.
        }
        // Swap rows.
        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }

        // Eliminate column elements below pivot.
        for r in (i + 1)..N {
            let factor = a[r][i] / a[i][i];
            b[r] -= factor * b[i];
            for c in i..N {
                a[r][c] -= factor * a[i][c];
            }
        }
    }

    // Back substitution.
    let mut x = [0.0; N];
    for i in (0..N).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..N {
            sum += a[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / a[i][i];
    }
    Some(x)
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

    #[test]
    fn empty_input_returns_empty() {
        let result = calculate_vwepr_curve(&[], 10, 60);
        assert!(result.is_empty());
    }

    #[test]
    fn zero_projection_returns_empty() {
        let candles = vec![candle(1000, 100.0, 500.0)];
        let result = calculate_vwepr_curve(&candles, 0, 60);
        assert!(result.is_empty());
    }

    #[test]
    fn anchor_point_matches_last_close() {
        let candles: Vec<OhlcCandle> = (0..20)
            .map(|i| candle(1000 + i * 60, 100.0 + i as f64 * 0.5, 1000.0))
            .collect();

        let result = calculate_vwepr_curve(&candles, 5, 60);

        assert!(!result.is_empty());
        // First point must be the anchor.
        assert_eq!(result[0].time, candles.last().unwrap().time);
        assert!(
            (result[0].value - candles.last().unwrap().close).abs() < 1e-10,
            "Anchor value must exactly equal last close"
        );
    }

    #[test]
    fn projection_timestamps_are_sequential() {
        let candles: Vec<OhlcCandle> = (0..30)
            .map(|i| candle(1000 + i * 300, 50.0 + i as f64, 500.0))
            .collect();

        let interval = 300_i64;
        let proj_len = 10;
        let result = calculate_vwepr_curve(&candles, proj_len, interval);

        assert_eq!(result.len(), proj_len + 1);

        let last_time = candles.last().unwrap().time;
        for (idx, point) in result.iter().enumerate() {
            let expected_time = last_time + (idx as i64 * interval);
            assert_eq!(
                point.time, expected_time,
                "Point {} has wrong timestamp", idx
            );
        }
    }

    #[test]
    fn volume_weighting_affects_curve() {
        // Regime-change price series: flat for the first half, then a sharp
        // ramp in the second half. Volume weighting the flat region heavily
        // vs. weighting the ramp region heavily should produce materially
        // different polynomial fits — and therefore different projections.
        let n = 30_i64;

        let price = |i: i64| -> f64 {
            if i < 15 { 100.0 } else { 100.0 + (i - 15) as f64 * 3.0 }
        };

        // Volume concentrated at the END (ramp region → steeper extrapolation).
        let candles_end_heavy: Vec<OhlcCandle> = (0..n)
            .map(|i| {
                let vol = if i >= 15 { 10_000.0 } else { 10.0 };
                candle(1000 + i * 60, price(i), vol)
            })
            .collect();

        // Volume concentrated at the START (flat region → flatter extrapolation).
        let candles_start_heavy: Vec<OhlcCandle> = (0..n)
            .map(|i| {
                let vol = if i < 15 { 10_000.0 } else { 10.0 };
                candle(1000 + i * 60, price(i), vol)
            })
            .collect();

        let r1 = calculate_vwepr_curve(&candles_end_heavy, 5, 60);
        let r2 = calculate_vwepr_curve(&candles_start_heavy, 5, 60);

        assert_eq!(r1.len(), 6);
        assert_eq!(r2.len(), 6);

        // The curves should differ meaningfully.
        let diff: f64 = r1.iter().zip(r2.iter())
            .map(|(a, b)| (a.value - b.value).abs())
            .sum();
        assert!(
            diff > 0.01,
            "Volume weighting should cause different projection curves (diff={})",
            diff
        );
    }

    #[test]
    fn flat_prices_project_flat() {
        let candles: Vec<OhlcCandle> = (0..20)
            .map(|i| candle(1000 + i * 60, 100.0, 1000.0))
            .collect();

        let result = calculate_vwepr_curve(&candles, 5, 60);

        // All projected values should be ≈ 100.0 for perfectly flat input.
        for point in &result {
            assert!(
                (point.value - 100.0).abs() < 1e-8,
                "Flat price input should project flat: got {}",
                point.value
            );
        }
    }

    #[test]
    fn linear_prices_have_near_zero_acceleration() {
        // Perfectly linear data → quadratic term a₂ ≈ 0.
        let candles: Vec<OhlcCandle> = (0..30)
            .map(|i| candle(1000 + i * 60, 100.0 + i as f64, 500.0))
            .collect();

        let (_, accel) = calculate_vwepr_with_accel(&candles, 5, 60);
        assert!(
            accel.abs() < 1e-6,
            "Linear data should produce near-zero acceleration: got {}",
            accel
        );
    }

    #[test]
    fn upward_curve_has_positive_acceleration() {
        // close = 100 + 0.1·i² → a₂ ≈ 0.1 > 0.
        let candles: Vec<OhlcCandle> = (0..30)
            .map(|i| candle(1000 + i * 60, 100.0 + 0.1 * (i as f64).powi(2), 500.0))
            .collect();

        let (_, accel) = calculate_vwepr_with_accel(&candles, 5, 60);
        assert!(
            accel > 0.0,
            "Upward-curving data should produce positive acceleration: got {}",
            accel
        );
    }

    #[test]
    fn single_candle_returns_anchor_only() {
        // With only 1 candle, the matrix is degenerate (s1=s2=s3=s4=0).
        // Should return the anchor point gracefully.
        let candles = vec![candle(5000, 42.0, 100.0)];
        let result = calculate_vwepr_curve(&candles, 5, 60);

        // At minimum we get the anchor point.
        assert!(!result.is_empty());
        assert_eq!(result[0].time, 5000);
        assert!((result[0].value - 42.0).abs() < 1e-10);
    }

    #[test]
    fn solver_3x3_identity() {
        let mat = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let rhs = [5.0, 6.0, 7.0];
        let res = solve_system_3x3(mat, rhs).unwrap();
        assert_eq!(res, [5.0, 6.0, 7.0]);
    }

    #[test]
    fn solver_3x3_known_system() {
        // 2x_0 +  x_1 -  x_2 =  8
        // -3x_0 - x_1 + 2x_2 = -11
        // -2x_0 + x_1 + 2x_2 = -3
        // Solution: x_0 = 2, x_1 = 3, x_2 = -1.
        let mat = [
            [2.0, 1.0, -1.0],
            [-3.0, -1.0, 2.0],
            [-2.0, 1.0, 2.0],
        ];
        let rhs = [8.0, -11.0, -3.0];
        let res = solve_system_3x3(mat, rhs).unwrap();
        assert!((res[0] - 2.0).abs() < 1e-10);
        assert!((res[1] - 3.0).abs() < 1e-10);
        assert!((res[2] - -1.0).abs() < 1e-10);
    }

    #[test]
    fn solver_3x3_singular_returns_none() {
        // Two identical rows → singular.
        let mat = [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ];
        let rhs = [1.0, 1.0, 2.0];
        assert!(solve_system_3x3(mat, rhs).is_none());
    }

    #[test]
    fn division_by_zero_guard() {
        // All zeros → det = 0 → should not panic, return anchor only.
        let candles = vec![
            OhlcCandle { time: 100, open: 0.0, high: 0.0, low: 0.0, close: 0.0, volume: 0.0 },
        ];
        let result = calculate_vwepr_curve(&candles, 5, 60);
        // Should not panic; returns at least the anchor.
        assert!(!result.is_empty());
    }

    #[test]
    fn large_window_capped_at_max() {
        // More than MAX_WINDOW candles — window should be capped.
        let candles: Vec<OhlcCandle> = (0..200)
            .map(|i| candle(1000 + i * 60, 100.0 + (i as f64 * 0.1), 500.0))
            .collect();

        let result = calculate_vwepr_curve(&candles, 10, 60);
        assert_eq!(result.len(), 11); // 1 anchor + 10 projected.

        // Anchor must match the LAST candle (index 199), not candle at MAX_WINDOW.
        assert_eq!(result[0].time, candles[199].time);
    }
}
