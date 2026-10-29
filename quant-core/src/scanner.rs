// quant/scanner.rs — Located Pattern & Strategy Scanner (FEAT-037).
//
// The classic ConsensusEngine returns patterns/strategies as bare name
// strings (`Vec<String>`) — enough for an LLM prompt, but useless for
// drawing on a chart. This module produces **located** detections: every
// pattern and strategy carries the exact candle index, UNIX timestamp,
// timeframe, and the price geometry needed to render a marker / zone /
// level on the front-end chart.
//
// ── Design ────────────────────────────────────────────────────────────────
//   • Reuses the existing `PatternEngine` and `StrategyEngine` verbatim —
//     this module is purely a *locator* layer, not a new detector. It walks
//     the candle history and records WHERE each engine fired.
//   • A `lookback` window bounds the scan to the most recent N candles so
//     the chart isn't cluttered with months-old signals.
//   • Deduplicates by name, keeping the **most recent** occurrence of each
//     pattern / strategy (the one a trader actually cares about).
//   • Stateless and pure — driven on-demand by the React UI for any
//     user-chosen symbol, and by the background radar worker for live alerts.

use serde::{Deserialize, Serialize};

use crate::patterns::{Candle, PatternEngine};
use crate::strategies::{IndicatorSnapshot, StrategyEngine};
use crate::{ConsensusEngine, IndicatorState};

// ── Input Contract ──────────────────────────────────────────────────────

/// An OHLCV candle with a UNIX-seconds timestamp.
///
/// This is the IPC input contract from the React chart, whose candles
/// already carry a `time` field. The timestamp is what lets the front-end
/// place a marker on the exact bar where a signal fired.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimedCandle {
    /// UNIX timestamp in seconds (candle open time).
    pub time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    #[serde(default)]
    pub volume: f64,
}

impl TimedCandle {
    #[inline]
    fn to_plain(&self) -> Candle {
        Candle {
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            taker_buy_volume: None,
        }
    }
}

// ── Bias Classification ─────────────────────────────────────────────────

/// Directional bias of a detection, used to colour markers on the chart.
pub const BIAS_BULLISH: &str = "BULLISH";
pub const BIAS_BEARISH: &str = "BEARISH";
pub const BIAS_NEUTRAL: &str = "NEUTRAL";

/// Infer directional bias from a pattern / strategy name.
///
/// Keyword-based so it stays correct as new detectors are added to the
/// pattern / strategy engines without needing a parallel enum.
pub fn classify_bias(name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    let bullish = n.contains("bull")
        || n.contains("hammer")
        || n.contains("golden")
        || n.contains("breakout")
        || n.contains("bounce")
        || n.contains("accumulation")
        || n.contains("inverse head")
        || n.contains("falling wedge")
        || n.contains("double bottom")
        || n.contains("triple bottom")
        || n.contains("ascending tri")
        || n.contains("cup and handle");
    let bearish = n.contains("bear")
        || n.contains("shooting")
        || n.contains("death")
        || n.contains("breakdown")
        || n.contains("distribution")
        || n.contains("head & shoulders top")
        || n.contains("rising wedge")
        || n.contains("double top")
        || n.contains("triple top")
        || n.contains("descending tri")
        || n.contains("inverse cup");

    if bullish && !bearish {
        BIAS_BULLISH
    } else if bearish && !bullish {
        BIAS_BEARISH
    } else {
        BIAS_NEUTRAL
    }
}

// ── Output Contracts ────────────────────────────────────────────────────

/// A candlestick pattern located at a specific bar, with the geometry of
/// that candle so the UI can draw a highlight box + marker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocatedPattern {
    pub name: String,
    pub bias: String,
    pub candle_index: usize,
    pub time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_time: Option<i64>,
}

/// An institutional strategy located at a specific bar.
///
/// `level` carries the strategy's key price line when meaningful
/// (ORB high/low, VWAP) so the UI can draw a horizontal reference line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocatedStrategy {
    pub name: String,
    pub bias: String,
    pub candle_index: usize,
    pub time: i64,
    pub price: f64,
    pub level: Option<f64>,
}

/// Full per-symbol scan result returned to the UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadarScan {
    pub symbol: String,
    pub timeframe: String,
    pub candle_count: usize,
    pub last_close: f64,
    pub last_time: i64,
    pub trend_score: i32,
    pub momentum_state: String,
    pub volatility_state: String,
    pub volume_flow_state: String,
    pub patterns: Vec<LocatedPattern>,
    pub strategies: Vec<LocatedStrategy>,
}

// ── Tuning Constants ────────────────────────────────────────────────────

/// Default number of most-recent candles to scan for located signals.
pub const DEFAULT_LOOKBACK: usize = 60;

/// Minimum candles required before the consensus summary is meaningful.
const MIN_CONSENSUS_CANDLES: usize = 30;

// ── Scanner ─────────────────────────────────────────────────────────────

/// Scan a timed candle series for located patterns and strategies.
///
/// Returns a `RadarScan` containing both the consensus summary (trend /
/// momentum / volatility / volume flow) and the located detections, ready
/// to be rendered on the front-end chart.
pub fn scan(symbol: &str, timed: &[TimedCandle], timeframe: &str, lookback: usize) -> RadarScan {
    let plain: Vec<Candle> = timed.iter().map(TimedCandle::to_plain).collect();
    let last_close = plain.last().map(|c| c.close).unwrap_or(0.0);
    let last_time = timed.last().map(|c| c.time).unwrap_or(0);

    // ── Consensus summary (reuse the canonical engine) ───────────────────
    let (trend_score, momentum_state, volatility_state, volume_flow_state) =
        if plain.len() >= MIN_CONSENSUS_CANDLES {
            let indicators = IndicatorState::from_candles_basic(&plain);
            let report = ConsensusEngine::compile_consensus(symbol, &plain, &indicators, timeframe);
            (
                report.trend_score,
                report.momentum_state,
                report.volatility_state,
                report.volume_flow_state,
            )
        } else {
            (0, "NEUTRAL".to_string(), "NORMAL".to_string(), "NEUTRAL".to_string())
        };

    let mut patterns: Vec<LocatedPattern> = Vec::new();
    let mut strategies: Vec<LocatedStrategy> = Vec::new();

    if !plain.is_empty() {
        use std::collections::HashMap;
        let lb = lookback.max(1);
        let scan_start = plain.len().saturating_sub(lb);

        // Track the position of each named detection so a later occurrence
        // overwrites the earlier one (keep the most recent).
        let mut pattern_pos: HashMap<String, usize> = HashMap::new();
        let mut strategy_pos: HashMap<String, usize> = HashMap::new();

        for i in scan_start..plain.len() {
            let window = &plain[..=i];
            let bar = &timed[i];

            // ── Patterns located at bar i ────────────────────────────────
            for name in PatternEngine::analyze(window) {
                let located = LocatedPattern {
                    bias: classify_bias(&name).to_string(),
                    candle_index: i,
                    time: bar.time,
                    open: bar.open,
                    high: bar.high,
                    low: bar.low,
                    close: bar.close,
                    name: name.clone(),
                    start_time: None,
                };
                match pattern_pos.get(&name) {
                    Some(&pos) => patterns[pos] = located,
                    None => {
                        pattern_pos.insert(name, patterns.len());
                        patterns.push(located);
                    }
                }
            }

            // ── Strategies located at bar i ──────────────────────────────
            let snapshot = IndicatorState::from_candles_basic(window).to_snapshot();
            for name in StrategyEngine::evaluate(window, &snapshot) {
                let located = LocatedStrategy {
                    bias: classify_bias(&name).to_string(),
                    candle_index: i,
                    time: bar.time,
                    price: bar.close,
                    level: strategy_level(&name, &snapshot),
                    name: name.clone(),
                };
                match strategy_pos.get(&name) {
                    Some(&pos) => strategies[pos] = located,
                    None => {
                        strategy_pos.insert(name, strategies.len());
                        strategies.push(located);
                    }
                }
            }
        }

        // ── Structural Chart Patterns located in lookback window ──────────
        let structural_patterns = crate::chart_patterns::ChartPatternEngine::analyze(&plain);
        for p in structural_patterns {
            if p.end_idx >= scan_start && p.end_idx < timed.len() && p.start_idx < timed.len() {
                let mut max_high = timed[p.start_idx].high;
                let mut min_low = timed[p.start_idx].low;
                for j in p.start_idx..=p.end_idx {
                    if timed[j].high > max_high {
                        max_high = timed[j].high;
                    }
                    if timed[j].low < min_low {
                        min_low = timed[j].low;
                    }
                }

                let located = LocatedPattern {
                    name: p.pattern_type.clone(),
                    bias: classify_bias(&p.pattern_type).to_string(),
                    candle_index: p.end_idx,
                    time: timed[p.end_idx].time,
                    open: timed[p.end_idx].open,
                    high: max_high,
                    low: min_low,
                    close: timed[p.end_idx].close,
                    start_time: Some(timed[p.start_idx].time),
                };

                match pattern_pos.get(&p.pattern_type) {
                    Some(&pos) => {
                        if located.candle_index > patterns[pos].candle_index {
                            patterns[pos] = located;
                        }
                    }
                    None => {
                        pattern_pos.insert(p.pattern_type.clone(), patterns.len());
                        patterns.push(located);
                    }
                }
            }
        }
    }

    // Present detections most-recent first.
    patterns.sort_by(|a, b| b.candle_index.cmp(&a.candle_index));
    strategies.sort_by(|a, b| b.candle_index.cmp(&a.candle_index));

    RadarScan {
        symbol: symbol.to_string(),
        timeframe: timeframe.to_string(),
        candle_count: plain.len(),
        last_close,
        last_time,
        trend_score,
        momentum_state,
        volatility_state,
        volume_flow_state,
        patterns,
        strategies,
    }
}

/// Resolve the meaningful price level for a strategy, when one exists.
fn strategy_level(name: &str, snap: &IndicatorSnapshot) -> Option<f64> {
    let n = name.to_ascii_lowercase();
    let finite = |v: f64| if v.is_finite() { Some(v) } else { None };

    if n.contains("orb breakout") {
        finite(snap.orb_high)
    } else if n.contains("orb breakdown") {
        finite(snap.orb_low)
    } else if n.contains("vwap") {
        finite(snap.vwap)
    } else if n.contains("golden") || n.contains("death") {
        finite(snap.sma_50)
    } else {
        None
    }
}

// ── Unit Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a rising series long enough for indicators to be finite.
    fn rising(count: usize) -> Vec<TimedCandle> {
        (0..count)
            .map(|i| {
                let base = 100.0 + i as f64 * 0.5;
                TimedCandle {
                    time: 1_000 + i as i64 * 600,
                    open: base,
                    high: base + 1.0,
                    low: base - 1.0,
                    close: base + 0.4,
                    volume: 1_000.0,
                }
            })
            .collect()
    }

    #[test]
    fn bias_classification_is_correct() {
        assert_eq!(classify_bias("Bullish Engulfing"), BIAS_BULLISH);
        assert_eq!(classify_bias("Bearish Engulfing"), BIAS_BEARISH);
        assert_eq!(classify_bias("Golden Cross"), BIAS_BULLISH);
        assert_eq!(classify_bias("Death Cross"), BIAS_BEARISH);
        assert_eq!(classify_bias("ORB Breakout (Bullish)"), BIAS_BULLISH);
        assert_eq!(classify_bias("ORB Breakdown (Bearish)"), BIAS_BEARISH);
        assert_eq!(classify_bias("Doji"), BIAS_NEUTRAL);
    }

    #[test]
    fn scan_returns_summary_for_sufficient_data() {
        let candles = rising(60);
        let report = scan("TEST", &candles, "10m", DEFAULT_LOOKBACK);
        assert_eq!(report.symbol, "TEST");
        assert_eq!(report.timeframe, "10m");
        assert_eq!(report.candle_count, 60);
        assert!((report.last_close - candles.last().unwrap().close).abs() < 1e-9);
        assert_eq!(report.last_time, candles.last().unwrap().time);
    }

    #[test]
    fn detects_and_locates_a_hammer() {
        // A clean hammer: tiny body at the top, long lower shadow.
        let mut candles = rising(40);
        let last = candles.len() - 1;
        candles[last] = TimedCandle {
            time: candles[last].time,
            open: 130.0,
            high: 130.5,
            low: 122.0,
            close: 130.2,
            volume: 1_500.0,
        };

        let report = scan("TEST", &candles, "10m", DEFAULT_LOOKBACK);
        let hammer = report.patterns.iter().find(|p| p.name == "Hammer");
        assert!(hammer.is_some(), "expected a Hammer to be located");
        let hammer = hammer.unwrap();
        assert_eq!(hammer.candle_index, last);
        assert_eq!(hammer.time, candles[last].time);
        assert_eq!(hammer.bias, BIAS_BULLISH);
    }

    #[test]
    fn empty_input_is_safe() {
        let report = scan("TEST", &[], "10m", DEFAULT_LOOKBACK);
        assert_eq!(report.candle_count, 0);
        assert!(report.patterns.is_empty());
        assert!(report.strategies.is_empty());
    }

    #[test]
    fn scan_serializes_for_ipc() {
        let candles = rising(35);
        let report = scan("TEST", &candles, "10m", DEFAULT_LOOKBACK);
        let json = serde_json::to_string(&report).expect("RadarScan must serialize");
        assert!(json.contains("patterns"));
        assert!(json.contains("strategies"));
        assert!(json.contains("timeframe"));
    }
}
