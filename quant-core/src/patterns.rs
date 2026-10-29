// quant/patterns.rs — Candlestick Pattern Matcher (Tauri-local copy).
//
// Mirrors aggregator/src/quant/patterns.rs for local Tauri execution.
// Detects: Bullish Engulfing, Bearish Engulfing, Doji, Hammer, Shooting Star.
//
// `PatternEngine::analyze` (those five) is unchanged on purpose: it is consumed BY NAME
// by consensus `active_patterns` (get_consensus_report), scanner `classify_bias` (radar)
// and tool-server `pattern_context::candle_events` (get_chart_patterns context). Adding
// names there would change all three — `classify_bias` reads "Morning Star" or "Pin Bar"
// as neutral, and `candle_events` guards bar adjacency for Engulfing only and caps the
// context at 12 events. So the price-action detectors get_price_action reads (pin bar,
// inside / outside bar, momentum candle, harami, morning / evening star, tweezers) are a
// separate `PatternEngine::analyze_extended`, which re-uses `analyze` for the five so
// the two can never disagree on them. `candle_anatomy` and `prior_volume_sma` describe
// one bar's shape and its volume against the bars before it.

/// A single OHLCV candle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Candle {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    /// Base volume bought by takers (Binance kline field 9), when known. The
    /// exact bar delta is `2 * taker_buy_volume - volume`; the agent's order-flow
    /// tool uses it instead of the close-location proxy. `None` for history
    /// stored before the column existed and for any source that cannot say —
    /// never a guessed 0.0, which would read as an all-sell bar. No pattern or
    /// indicator here reads it; it rides along so `get_candles` can return it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub taker_buy_volume: Option<f64>,
}

impl Candle {
    #[inline]
    pub fn body(&self) -> f64 { (self.close - self.open).abs() }
    #[inline]
    pub fn range(&self) -> f64 { self.high - self.low }
    #[inline]
    pub fn is_bullish(&self) -> bool { self.close > self.open }
    #[inline]
    pub fn is_bearish(&self) -> bool { self.open > self.close }
    #[inline]
    pub fn body_top(&self) -> f64 { self.open.max(self.close) }
    #[inline]
    pub fn body_bottom(&self) -> f64 { self.open.min(self.close) }
    #[inline]
    pub fn upper_shadow(&self) -> f64 { self.high - self.body_top() }
    #[inline]
    pub fn lower_shadow(&self) -> f64 { self.body_bottom() - self.low }
}

pub struct PatternEngine;

const MIN_RANGE: f64 = 1e-9;
const DOJI_BODY_RATIO: f64 = 0.10;
const SHADOW_BODY_MULTIPLIER: f64 = 2.0;
const BODY_POSITION_RATIO: f64 = 0.33;

impl PatternEngine {
    pub fn analyze(history: &[Candle]) -> Vec<String> {
        let mut signals: Vec<String> = Vec::new();
        if history.is_empty() { return signals; }

        let current = &history[history.len() - 1];
        let range = current.range();

        // Doji
        if range > MIN_RANGE && (current.body() / range) < DOJI_BODY_RATIO {
            signals.push("Doji".to_string());
        }

        // Hammer
        if range > MIN_RANGE && current.body() > MIN_RANGE {
            let body = current.body();
            if current.lower_shadow() >= body * SHADOW_BODY_MULTIPLIER
                && current.upper_shadow() <= range * BODY_POSITION_RATIO
            {
                signals.push("Hammer".to_string());
            }
        }

        // Shooting Star
        if range > MIN_RANGE && current.body() > MIN_RANGE {
            let body = current.body();
            if current.upper_shadow() >= body * SHADOW_BODY_MULTIPLIER
                && current.lower_shadow() <= range * BODY_POSITION_RATIO
            {
                signals.push("Shooting Star".to_string());
            }
        }

        // Two-candle patterns
        if history.len() >= 2 {
            let prev = &history[history.len() - 2];

            // Bullish Engulfing
            if prev.is_bearish() && current.is_bullish()
                && prev.body() > MIN_RANGE && current.body() > MIN_RANGE
                && current.body_bottom() <= prev.body_bottom()
                && current.body_top() >= prev.body_top()
            {
                signals.push("Bullish Engulfing".to_string());
            }

            // Bearish Engulfing
            if prev.is_bullish() && current.is_bearish()
                && prev.body() > MIN_RANGE && current.body() > MIN_RANGE
                && current.body_top() >= prev.body_top()
                && current.body_bottom() <= prev.body_bottom()
            {
                signals.push("Bearish Engulfing".to_string());
            }
        }

        signals
    }
}

/// One candlestick pattern on the newest bar of a window and the side it favours.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandlePattern {
    pub name: &'static str,
    /// `bullish`, `bearish` or `neutral`.
    pub direction: &'static str,
}

/// The shape of one bar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandleAnatomy {
    /// Body, upper wick and lower wick as % of the bar's range; the three sum to 100.
    pub body_pct: f64,
    pub upper_wick_pct: f64,
    pub lower_wick_pct: f64,
    /// Where the close sits in the range: 0 at the low, 1 at the high.
    pub close_location: f64,
    /// Range over ATR; `None` without a positive finite ATR.
    pub range_atr: Option<f64>,
    /// Volume over the prior bars' mean volume; `None` without that baseline.
    pub volume_ratio: Option<f64>,
}

/// Bars in the volume baseline: the 20-bar volume SMA (RVOL's denominator) a trader's
/// volume pane shows.
pub const VOLUME_SMA_BARS: usize = 20;
/// A pin bar's wick is at least two thirds of its range: the rejection IS the bar.
const PIN_WICK_RATIO: f64 = 2.0 / 3.0;
/// A body this many ATRs long is a momentum candle whatever its wicks: one bar
/// covering a bar and a half of normal range is displacement, not noise.
const MOMENTUM_BODY_ATR: f64 = 1.5;
/// ...or a near-marubozu (body at least 85% of the range) at least one ATR tall: the
/// bar closed where it was going all bar, with nothing given back.
const MARUBOZU_BODY_RATIO: f64 = 0.85;
const MARUBOZU_RANGE_ATR: f64 = 1.0;
/// A star's first bar is a real move (body at least half its range) and its middle
/// bar a pause (body at most 30% of the first one's) — the indecision the third bar
/// resolves.
const STAR_FIRST_BODY_RATIO: f64 = 0.5;
const STAR_BODY_RATIO: f64 = 0.3;
/// Tweezer lows (highs) match within a tenth of the smaller bar's range: the same
/// price defended twice, not two bars that happen to be close.
const TWEEZER_RANGE_RATIO: f64 = 0.1;

impl PatternEngine {
    /// The five `analyze` patterns with their direction, then the price-action
    /// patterns, all on the newest bar of `history`, in a fixed order. `atr` sizes the
    /// momentum candle and is ignored unless positive and finite (no momentum candle
    /// without it). Never panics: short, flat or non-finite input just matches less.
    pub fn analyze_extended(history: &[Candle], atr: Option<f64>) -> Vec<CandlePattern> {
        let pat = |name: &'static str, direction: &'static str| CandlePattern { name, direction };
        // The five, from `analyze` itself so the two can never disagree on them.
        let mut out: Vec<CandlePattern> = Self::analyze(history)
            .iter()
            .filter_map(|name| match name.as_str() {
                "Doji" => Some(pat("Doji", "neutral")),
                "Hammer" => Some(pat("Hammer", "bullish")),
                "Shooting Star" => Some(pat("Shooting Star", "bearish")),
                "Bullish Engulfing" => Some(pat("Bullish Engulfing", "bullish")),
                "Bearish Engulfing" => Some(pat("Bearish Engulfing", "bearish")),
                _ => None,
            })
            .collect();
        let Some(c) = history.last() else {
            return out;
        };
        let atr = atr.filter(|a| a.is_finite() && *a > 0.0);
        let (range, body) = (c.range(), c.body());

        // Pin bar. A wick of at least 2/3 of the range leaves at most 1/3 for the body
        // and the other wick, so the body sits in the opposite third by construction —
        // no second test needed.
        if range > MIN_RANGE {
            if c.lower_shadow() >= range * PIN_WICK_RATIO {
                out.push(pat("Bullish Pin Bar", "bullish"));
            } else if c.upper_shadow() >= range * PIN_WICK_RATIO {
                out.push(pat("Bearish Pin Bar", "bearish"));
            }
        }

        // Momentum / marubozu candle.
        if let Some(a) = atr {
            if body > MIN_RANGE
                && (body >= MOMENTUM_BODY_ATR * a
                    || (body / range >= MARUBOZU_BODY_RATIO && range >= MARUBOZU_RANGE_ATR * a))
            {
                if c.is_bullish() {
                    out.push(pat("Bullish Momentum Candle", "bullish"));
                } else {
                    out.push(pat("Bearish Momentum Candle", "bearish"));
                }
            }
        }

        if history.len() >= 2 {
            let p = &history[history.len() - 2];
            if c.high < p.high && c.low > p.low {
                out.push(pat("Inside Bar", "neutral"));
            }
            if c.high > p.high && c.low < p.low {
                let direction = if c.is_bullish() {
                    "bullish"
                } else if c.is_bearish() {
                    "bearish"
                } else {
                    "neutral"
                };
                out.push(pat("Outside Bar", direction));
            }
            // Harami: a smaller body inside the previous one, of the other colour.
            let contained =
                c.body_top() <= p.body_top() && c.body_bottom() >= p.body_bottom() && c.body() < p.body();
            if contained && p.is_bearish() && c.is_bullish() {
                out.push(pat("Bullish Harami", "bullish"));
            }
            if contained && p.is_bullish() && c.is_bearish() {
                out.push(pat("Bearish Harami", "bearish"));
            }
            let tol = TWEEZER_RANGE_RATIO * p.range().min(c.range());
            if tol > 0.0 {
                if p.is_bearish() && c.is_bullish() && (p.low - c.low).abs() <= tol {
                    out.push(pat("Tweezer Bottom", "bullish"));
                }
                if p.is_bullish() && c.is_bearish() && (p.high - c.high).abs() <= tol {
                    out.push(pat("Tweezer Top", "bearish"));
                }
            }
        }

        // ponytail: no gap requirement (a 24/7 market has no session gaps to open
        // across) and no prior-trend test; the context is the key level get_price_action
        // pairs a pattern with.
        if history.len() >= 3 {
            let (pp, p) = (&history[history.len() - 3], &history[history.len() - 2]);
            let mid = |x: &Candle| (x.open + x.close) / 2.0;
            let star = pp.body() > MIN_RANGE
                && pp.body() >= STAR_FIRST_BODY_RATIO * pp.range()
                && p.body() <= STAR_BODY_RATIO * pp.body();
            if star && pp.is_bearish() && mid(p) < mid(pp) && c.is_bullish() && c.close > mid(pp) {
                out.push(pat("Morning Star", "bullish"));
            }
            if star && pp.is_bullish() && mid(p) > mid(pp) && c.is_bearish() && c.close < mid(pp) {
                out.push(pat("Evening Star", "bearish"));
            }
        }
        out
    }
}

/// One bar's shape: body and wicks as % of its range, where it closed in that range,
/// the range in ATRs and the volume against `volume_sma`. `None` for a flat bar (no
/// range to divide by) or a non-finite one; `range_atr` / `volume_ratio` are `None`
/// without a positive finite ATR / baseline.
pub fn candle_anatomy(c: &Candle, atr: Option<f64>, volume_sma: Option<f64>) -> Option<CandleAnatomy> {
    let range = c.range();
    if !(range.is_finite() && range > MIN_RANGE && c.open.is_finite() && c.close.is_finite()) {
        return None;
    }
    let pct = |v: f64| v / range * 100.0;
    Some(CandleAnatomy {
        body_pct: pct(c.body()),
        upper_wick_pct: pct(c.upper_shadow()),
        lower_wick_pct: pct(c.lower_shadow()),
        close_location: ((c.close - c.low) / range).clamp(0.0, 1.0),
        range_atr: atr.filter(|a| a.is_finite() && *a > 0.0).map(|a| range / a),
        volume_ratio: volume_sma
            .filter(|s| s.is_finite() && *s > 0.0)
            .map(|s| c.volume / s)
            .filter(|r| r.is_finite()),
    })
}

/// Mean volume of the `period` bars BEFORE bar `idx` (`idx == len` is the bar after the
/// series). `None` when fewer than `period` bars precede it, `period` is 0, or the mean
/// is not a positive finite number — no baseline, so no ratio, never a fabricated one.
pub fn prior_volume_sma(candles: &[Candle], idx: usize, period: usize) -> Option<f64> {
    if period == 0 || idx < period || idx > candles.len() {
        return None;
    }
    let mean = candles[idx - period..idx].iter().map(|c| c.volume).sum::<f64>() / period as f64;
    (mean.is_finite() && mean > 0.0).then_some(mean)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn vbar(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Candle {
        Candle { open, high, low, close, volume, taker_buy_volume: None }
    }

    fn candle(open: f64, high: f64, low: f64, close: f64) -> Candle {
        vbar(open, high, low, close, 10.0)
    }

    fn bars(ohlc: &[(f64, f64, f64, f64)]) -> Vec<Candle> {
        ohlc.iter().map(|&(o, h, l, c)| candle(o, h, l, c)).collect()
    }

    fn has(history: &[Candle], atr: Option<f64>, name: &'static str, direction: &'static str) -> bool {
        PatternEngine::analyze_extended(history, atr).contains(&CandlePattern { name, direction })
    }

    const FIVE: [&str; 5] = ["Doji", "Hammer", "Shooting Star", "Bullish Engulfing", "Bearish Engulfing"];

    fn five_of(history: &[Candle], atr: Option<f64>) -> Vec<CandlePattern> {
        PatternEngine::analyze_extended(history, atr).into_iter().filter(|p| FIVE.contains(&p.name)).collect()
    }

    #[test]
    fn the_original_five_read_as_before() {
        // The aggregator copy's fixtures. Doji: body 0.1 of a 10 range (< 10%); the 5 and
        // 4.9 wicks are both over a third of the range, so no hammer / star.
        let doji = bars(&[(100.0, 105.0, 95.0, 100.1)]);
        assert_eq!(PatternEngine::analyze(&doji), vec!["Doji"]);
        assert_eq!(five_of(&doji, None), vec![CandlePattern { name: "Doji", direction: "neutral" }]);
        // Hammer: lower wick 5 >= 2 × body 1, upper wick 0.2 <= 0.33 × 6.2.
        let hammer = bars(&[(100.0, 101.2, 95.0, 101.0)]);
        assert_eq!(PatternEngine::analyze(&hammer), vec!["Hammer"]);
        assert_eq!(five_of(&hammer, None), vec![CandlePattern { name: "Hammer", direction: "bullish" }]);
        // Shooting Star: upper wick 5 >= 2 × body 1, lower wick 0.2 <= 0.33 × 6.2.
        let star = bars(&[(101.0, 106.0, 99.8, 100.0)]);
        assert_eq!(PatternEngine::analyze(&star), vec!["Shooting Star"]);
        assert_eq!(five_of(&star, None), vec![CandlePattern { name: "Shooting Star", direction: "bearish" }]);
        // Engulfing: body 99..105 covers 100..104, and the mirror.
        let bull = bars(&[(104.0, 105.0, 99.0, 100.0), (99.0, 106.0, 98.0, 105.0)]);
        assert_eq!(PatternEngine::analyze(&bull), vec!["Bullish Engulfing"]);
        assert_eq!(five_of(&bull, None), vec![CandlePattern { name: "Bullish Engulfing", direction: "bullish" }]);
        let bear = bars(&[(100.0, 106.0, 99.0, 105.0), (106.0, 107.0, 98.0, 99.0)]);
        assert_eq!(PatternEngine::analyze(&bear), vec!["Bearish Engulfing"]);
        assert_eq!(five_of(&bear, None), vec![CandlePattern { name: "Bearish Engulfing", direction: "bearish" }]);
        assert!(PatternEngine::analyze(&[]).is_empty());
        assert!(PatternEngine::analyze_extended(&[], Some(1.0)).is_empty());
    }

    #[test]
    fn pin_bars() {
        // Lower wick 100.8 - 97 = 3.8 >= 2/3 × 4; upper wick 0.1.
        assert!(has(&bars(&[(100.8, 101.0, 97.0, 100.9)]), None, "Bullish Pin Bar", "bullish"));
        // Upper wick 103 - 99.2 = 3.8 >= 2/3 × 4; lower wick 0.1.
        assert!(has(&bars(&[(99.2, 103.0, 99.0, 99.1)]), None, "Bearish Pin Bar", "bearish"));
        // Wicks 1 and 0.5 of a 2 range: neither reaches 1.333.
        let plain = PatternEngine::analyze_extended(&bars(&[(100.0, 101.0, 99.0, 100.5)]), None);
        assert!(plain.iter().all(|p| !p.name.contains("Pin Bar")), "{plain:?}");
    }

    #[test]
    fn inside_and_outside_bars() {
        // 104 < 105 and 96 > 95.
        assert!(has(&bars(&[(100.0, 105.0, 95.0, 102.0), (101.0, 104.0, 96.0, 103.0)]), None, "Inside Bar", "neutral"));
        // An equal high is not inside.
        let equal = bars(&[(100.0, 105.0, 95.0, 102.0), (101.0, 105.0, 96.0, 103.0)]);
        assert!(!has(&equal, None, "Inside Bar", "neutral"));
        // 103 > 102 and 98 < 99, closing up.
        assert!(has(&bars(&[(100.0, 102.0, 99.0, 101.0), (100.5, 103.0, 98.0, 102.5)]), None, "Outside Bar", "bullish"));
        // The high only matches the previous one: not outside.
        let matched = PatternEngine::analyze_extended(&bars(&[(100.0, 102.0, 99.0, 101.0), (100.5, 102.0, 98.0, 101.5)]), None);
        assert!(matched.iter().all(|p| p.name != "Outside Bar"), "{matched:?}");
    }

    #[test]
    fn momentum_candles_need_an_atr() {
        // Body 3.1 >= 1.5 × ATR 2.
        let long_body = bars(&[(100.0, 103.2, 99.9, 103.1)]);
        assert!(has(&long_body, Some(2.0), "Bullish Momentum Candle", "bullish"));
        // Marubozu: body 2.0 of a 2.15 range (0.93 >= 0.85), and 2.15 >= 1 ATR.
        let marubozu = bars(&[(100.0, 102.1, 99.95, 102.0)]);
        assert!(has(&marubozu, Some(2.0), "Bullish Momentum Candle", "bullish"));
        assert!(!has(&marubozu, None, "Bullish Momentum Candle", "bullish"), "no ATR, no momentum candle");
        assert!(!has(&marubozu, Some(f64::NAN), "Bullish Momentum Candle", "bullish"));
        // Body 0.5 and 25% of the range: neither rule.
        let small = PatternEngine::analyze_extended(&bars(&[(100.0, 101.0, 99.0, 100.5)]), Some(2.0));
        assert!(small.iter().all(|p| !p.name.contains("Momentum")), "{small:?}");
        // Mirror: body 3.1 down.
        assert!(has(&bars(&[(103.1, 103.2, 99.9, 100.0)]), Some(2.0), "Bearish Momentum Candle", "bearish"));
    }

    #[test]
    fn harami() {
        // Bearish 105 -> 100, then bullish 101 -> 102.5: body 101..102.5 inside 100..105.
        assert!(has(&bars(&[(105.0, 105.5, 99.5, 100.0), (101.0, 103.0, 100.5, 102.5)]), None, "Bullish Harami", "bullish"));
        // Bullish 100 -> 105, then bearish 104 -> 102.5: body 102.5..104 inside 100..105.
        assert!(has(&bars(&[(100.0, 105.5, 99.5, 105.0), (104.0, 104.5, 102.0, 102.5)]), None, "Bearish Harami", "bearish"));
        // Two bullish bars: no harami either way.
        let same = PatternEngine::analyze_extended(&bars(&[(100.0, 105.5, 99.5, 105.0), (101.0, 103.0, 100.5, 102.5)]), None);
        assert!(same.iter().all(|p| !p.name.contains("Harami")), "{same:?}");
    }

    #[test]
    fn morning_and_evening_stars() {
        // Bar 1 bearish 110 -> 104 (body 6 of 7), bar 2 body 0.3 <= 1.8 with mid 103.35
        // below 107, bar 3 bullish closing 108.5 above bar 1's mid 107.
        let morning = [(110.0, 110.5, 103.5, 104.0), (103.5, 104.0, 102.0, 103.2), (103.5, 109.0, 103.0, 108.5)];
        assert!(has(&bars(&morning), None, "Morning Star", "bullish"));
        // Closing at 106 does not get back above 107.
        let short = [morning[0], morning[1], (103.5, 109.0, 103.0, 106.0)];
        assert!(!has(&bars(&short), None, "Morning Star", "bullish"));
        // Mirror: bullish 104 -> 110, pause at mid 110.65 above 107, bearish close 105.5.
        let evening = [(104.0, 110.5, 103.5, 110.0), (110.5, 112.0, 110.0, 110.8), (110.5, 111.0, 105.0, 105.5)];
        assert!(has(&bars(&evening), None, "Evening Star", "bearish"));
        let short = [evening[0], evening[1], (110.5, 111.0, 105.0, 108.0)];
        assert!(!has(&bars(&short), None, "Evening Star", "bearish"));
    }

    #[test]
    fn tweezers() {
        // Lows 99 and 99.05: 0.05 apart, within 0.1 × min(3.5, 2.95) = 0.295.
        assert!(has(&bars(&[(102.0, 102.5, 99.0, 99.5), (99.6, 102.0, 99.05, 101.8)]), None, "Tweezer Bottom", "bullish"));
        // Highs 102 and 101.95.
        assert!(has(&bars(&[(99.0, 102.0, 98.5, 101.5), (101.4, 101.95, 99.0, 99.2)]), None, "Tweezer Top", "bearish"));
        // Lows 99 and 99.5: 0.5 apart, beyond 0.1 × 2.5.
        assert!(!has(&bars(&[(102.0, 102.5, 99.0, 99.5), (99.6, 102.0, 99.5, 101.8)]), None, "Tweezer Bottom", "bullish"));
    }

    #[test]
    fn anatomy_of_a_bar() {
        // Range 6: body 3 (50%), upper wick 1, lower wick 2; closes 5/6 of the way up.
        let a = candle_anatomy(&candle(100.0, 104.0, 98.0, 103.0), Some(3.0), Some(5.0)).unwrap();
        assert!((a.body_pct - 50.0).abs() < 1e-9);
        assert!((a.upper_wick_pct - 100.0 / 6.0).abs() < 1e-9);
        assert!((a.lower_wick_pct - 200.0 / 6.0).abs() < 1e-9);
        assert!((a.close_location - 5.0 / 6.0).abs() < 1e-9);
        assert_eq!((a.range_atr, a.volume_ratio), (Some(2.0), Some(2.0)));
        assert!(candle_anatomy(&candle(100.0, 100.0, 100.0, 100.0), Some(3.0), Some(5.0)).is_none(), "flat");
        assert!(candle_anatomy(&candle(100.0, 104.0, 98.0, f64::NAN), None, None).is_none());
        let bare = candle_anatomy(&candle(100.0, 104.0, 98.0, 103.0), None, Some(0.0)).unwrap();
        assert_eq!((bare.range_atr, bare.volume_ratio), (None, None));
    }

    #[test]
    fn prior_volume_sma_needs_a_full_window() {
        let c: Vec<Candle> = (0..25).map(|_| candle(1.0, 1.1, 0.9, 1.0)).collect();
        assert_eq!(prior_volume_sma(&c, 20, 20), Some(10.0));
        assert_eq!(prior_volume_sma(&c, 19, 20), None);
        assert_eq!(prior_volume_sma(&c, 20, 0), None);
        assert_eq!(prior_volume_sma(&c, 25, 20), Some(10.0), "the bar after the series");
        assert_eq!(prior_volume_sma(&c, 26, 20), None);
        let silent: Vec<Candle> = (0..25).map(|_| vbar(1.0, 1.1, 0.9, 1.0, 0.0)).collect();
        assert_eq!(prior_volume_sma(&silent, 20, 20), None);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn the_five_are_unchanged_and_nothing_panics(
            raw in prop::collection::vec((90.0f64..110.0, 0.0f64..5.0, 0.0f64..5.0, 90.0f64..110.0, 0u8..4), 0..6),
            atr in prop::option::of(0.0f64..5.0)
        ) {
            let history: Vec<Candle> = raw
                .iter()
                .map(|&(o, up, down, c, kind)| {
                    if kind == 0 {
                        candle(o, o, o, o)
                    } else {
                        candle(o, o.max(c) + up, o.min(c) - down, c)
                    }
                })
                .collect();
            let extended = PatternEngine::analyze_extended(&history, atr);
            let five: Vec<String> =
                extended.iter().filter(|p| FIVE.contains(&p.name)).map(|p| p.name.to_string()).collect();
            prop_assert_eq!(five, PatternEngine::analyze(&history));
            for p in &extended {
                prop_assert!(["bullish", "bearish", "neutral"].contains(&p.direction), "{:?}", p);
            }
            prop_assert_eq!(extended, PatternEngine::analyze_extended(&history, atr));
        }
    }
}
