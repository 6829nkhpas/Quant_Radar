// quant/strategies.rs — Institutional Strategy Engine (Tauri-local copy).
//
// Mirrors aggregator/src/quant/strategies.rs for local Tauri execution.
// Detects: Golden Cross, Death Cross, VWAP Bounce, ORB Breakout/Breakdown.

use crate::patterns::Candle;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndicatorSnapshot {
    pub sma_50: f64,
    pub sma_200: f64,
    pub prev_sma_50: f64,
    pub prev_sma_200: f64,
    pub vwap: f64,
    pub average_volume: f64,
    pub orb_high: f64,
    pub orb_low: f64,
}

pub struct StrategyEngine;

const VWAP_VOLUME_SURGE: f64 = 1.5;
const ORB_VOLUME_SURGE: f64 = 1.2;
const MIN_SMA_SEPARATION: f64 = 1e-9;

impl StrategyEngine {
    pub fn evaluate(history: &[Candle], indicators: &IndicatorSnapshot) -> Vec<String> {
        let mut signals: Vec<String> = Vec::new();
        if history.is_empty() { return signals; }
        let current = &history[history.len() - 1];

        // Golden / Death Cross
        if indicators.prev_sma_50.is_finite() && indicators.prev_sma_200.is_finite()
            && indicators.sma_50.is_finite() && indicators.sma_200.is_finite()
        {
            if indicators.prev_sma_50 <= indicators.prev_sma_200
                && indicators.sma_50 > indicators.sma_200
            {
                signals.push("Golden Cross".to_string());
            }
            if indicators.prev_sma_50 >= indicators.prev_sma_200
                && indicators.sma_50 < indicators.sma_200
            {
                signals.push("Death Cross".to_string());
            }
        }

        // VWAP Bounce (Bullish)
        if history.len() >= 2 && indicators.vwap.is_finite() {
            let prev = &history[history.len() - 2];
            if current.low <= indicators.vwap && current.close > indicators.vwap
                && prev.is_bearish() && indicators.average_volume > MIN_SMA_SEPARATION
                && current.volume > indicators.average_volume * VWAP_VOLUME_SURGE
            {
                signals.push("VWAP Bounce (Bullish)".to_string());
            }
        }

        // ORB Breakout / Breakdown
        if indicators.orb_high.is_finite() && indicators.average_volume > MIN_SMA_SEPARATION {
            if current.close > indicators.orb_high
                && current.volume > indicators.average_volume * ORB_VOLUME_SURGE
            {
                signals.push("ORB Breakout (Bullish)".to_string());
            }
        }
        if indicators.orb_low.is_finite() && indicators.average_volume > MIN_SMA_SEPARATION {
            if current.close < indicators.orb_low
                && current.volume > indicators.average_volume * ORB_VOLUME_SURGE
            {
                signals.push("ORB Breakdown (Bearish)".to_string());
            }
        }

        signals
    }
}
