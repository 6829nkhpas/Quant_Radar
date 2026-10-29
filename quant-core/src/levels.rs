// quant-core/src/levels.rs — Key-level engine.
//
// The levels a discretionary trader marks on a chart before taking a trade, read from
// one candle series:
//   1. swing zones: fractal swing highs and lows (the same 3/3 detector `structure`
//      uses) clustered within an ATR tolerance. A price that turned the market twice is
//      a zone, not a line, and its touch count is what makes it strong;
//   2. liquidity pools: two or more swing highs (lows) at the same price — the "equal
//      highs / lows" stops rest behind — and whether a later bar has swept them (a wick
//      through, the close back inside) or broken them (a close through);
//   3. fair value gaps: three-bar imbalances (bar i-2's high below bar i's low, or the
//      mirror) and how much of each later bars have traded back into;
//   4. order blocks: the last opposite-colour candle before the move that broke
//      structure (the same close-confirmed breaks `structure` reports as BOS / CHoCH),
//      and whether price has come back into it since;
//   5. trendlines: lines through two pivots of a kind that at least three pivots touch
//      and that no close crossed from the first touch to the last one — reported
//      `broken` when a close crossed it after that;
//   6. round numbers near price and a VWAP anchored at any bar, with its ±1σ band.
//
// Pure: no I/O, no clock, deterministic. Every index is into the input slice; the
// caller maps indices to timestamps and merges these with the levels only timestamps
// can give (previous day, week, month, sessions — tool-server `get_key_levels`). Like
// `structure`, this describes the chart AS OF the last bar, with the whole series in
// hand; it is not a bar-by-bar replay of what was knowable at each earlier bar.

use crate::chart_patterns::{SwingKind, SwingPoint};
use crate::patterns::Candle;
use crate::structure::{self, StructureConfig};
use crate::IndicatorState;

/// Swing prices within this many ATR(14) are one zone. A quarter of a bar's normal
/// range is the width a trader draws a zone box at; wider and two separate shelves
/// merge, narrower and a retest that undercut by a few ticks counts as a new level.
pub const ZONE_ATR: f64 = 0.25;
/// Swing highs (lows) within this many ATRs are "equal" — the double top / bottom whose
/// stops are the liquidity. Tighter than a zone: equal means the same price.
pub const EQUAL_ATR: f64 = 0.1;
/// A pivot within this many ATRs of a line touches it, and a close more than this far
/// on the wrong side crosses it. A fifth of a bar's range absorbs the wick noise a hand
/// drawn line also tolerates.
pub const TRENDLINE_ATR: f64 = 0.2;
/// Two points define any line; a third touch is what makes it a trendline.
pub const MIN_TRENDLINE_TOUCHES: usize = 3;
/// Gaps narrower than this many ATRs are ignored: a few-tick imbalance on a quiet bar
/// is not a level anyone trades.
pub const FVG_MIN_ATR: f64 = 0.1;
/// Order blocks are taken from the newest breaks only. A block behind a break ten
/// structure legs ago has long been traded through or forgotten.
const ORDER_BLOCK_BREAKS: usize = 10;
/// Trendlines kept per kind (support / resistance) and series. Two is what a trader
/// draws; every further pivot pair mostly redraws one of them.
const TRENDLINES_PER_KIND: usize = 2;

/// Swing prices clustered into one zone.
#[derive(Debug, Clone, PartialEq)]
pub struct Zone {
    pub low: f64,
    pub high: f64,
    /// The zone's midpoint.
    pub mid: f64,
    /// Swings in the zone — highs and lows alike: old resistance retested as support
    /// is the same level.
    pub touches: usize,
    pub last_touch_idx: usize,
    /// 0-1: touches (up to four) weigh 0.6, how recent the last touch is 0.4.
    pub strength: f64,
}

/// Equal highs or equal lows.
#[derive(Debug, Clone, PartialEq)]
pub struct LiquidityPool {
    /// `equal_highs` or `equal_lows`.
    pub side: &'static str,
    /// The pool's outer edge: the highest of the equal highs, the lowest of the lows.
    pub price: f64,
    pub touches: usize,
    pub last_touch_idx: usize,
    /// `broken` (a later close beyond the price), `swept` (a later wick beyond it and
    /// no close) or `intact`.
    pub status: &'static str,
}

/// A three-bar imbalance.
#[derive(Debug, Clone, PartialEq)]
pub struct FairValueGap {
    /// `bullish` (bar i-2's high below bar i's low) or `bearish` (the mirror).
    pub direction: &'static str,
    pub low: f64,
    pub high: f64,
    /// The middle (displacement) bar.
    pub idx: usize,
    /// 0-1: the deepest any later bar traded back into the gap, as a fraction of it.
    pub filled: f64,
}

/// The last opposite-colour candle before a break of structure.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderBlock {
    /// The break's direction: a bullish block is a bearish candle before a bullish break.
    pub direction: &'static str,
    pub low: f64,
    pub high: f64,
    pub idx: usize,
    pub break_idx: usize,
    /// A bar after the break traded back into the block.
    pub mitigated: bool,
}

/// A line through pivots of one kind.
#[derive(Debug, Clone, PartialEq)]
pub struct Trendline {
    /// `support` (through swing lows) or `resistance` (through swing highs).
    pub kind: &'static str,
    pub slope_per_bar: f64,
    pub touches: usize,
    /// The line is `first_touch_price + slope_per_bar × (x − first_touch_idx)`.
    pub first_touch_idx: usize,
    pub first_touch_price: f64,
    pub last_touch_idx: usize,
    /// The line projected to the last bar of the series.
    pub price_at_last: f64,
    /// `respected` (no close through it since the first touch) or `broken` (a close
    /// through it after the last touch).
    pub status: &'static str,
}

/// Every key level one series yields.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyLevels {
    /// ATR(14), `None` when it cannot be measured or is zero; every tolerance above is a
    /// multiple of it, and without it only exact matches cluster.
    pub atr: Option<f64>,
    /// Zones with two or more touches, plus single swings at or after the dealing-range
    /// edges (the newest swing high and low), ascending by price.
    pub zones: Vec<Zone>,
    pub pools: Vec<LiquidityPool>,
    /// Not fully filled gaps only, oldest first.
    pub fvgs: Vec<FairValueGap>,
    pub order_blocks: Vec<OrderBlock>,
    pub trendlines: Vec<Trendline>,
    /// VWAP anchored at the newest confirmed structure swing: `(swing idx, vwap)`.
    pub swing_vwap: Option<(usize, f64)>,
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn atr_14(candles: &[Candle]) -> Option<f64> {
    Some(IndicatorState::compute_atr(candles, 14)).filter(|a| a.is_finite() && *a > 0.0)
}

/// Cluster `(bar idx, price)` points into zones no wider than `tol`, ascending.
///
/// Greedy from the lowest price: a zone takes every next point within `tol` of its
/// lowest one, so a zone never spans more than `tol` however many points chain.
/// Non-finite prices are dropped; a non-finite or negative `tol` is 0 (only equal
/// prices merge). `n_bars` scales recency: a touch on the newest bar scores 1.
pub fn cluster_zones(points: &[(usize, f64)], tol: f64, n_bars: usize) -> Vec<Zone> {
    let tol = if tol.is_finite() && tol > 0.0 { tol } else { 0.0 };
    let mut pts: Vec<(usize, f64)> = points.iter().copied().filter(|(_, p)| p.is_finite()).collect();
    pts.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut zones = Vec::new();
    let mut i = 0;
    while i < pts.len() {
        let low = pts[i].1;
        let mut j = i;
        let mut last = 0usize;
        while j < pts.len() && pts[j].1 - low <= tol {
            last = last.max(pts[j].0);
            j += 1;
        }
        let high = pts[j - 1].1;
        let touches = j - i;
        let recency = if n_bars > 1 {
            last.min(n_bars - 1) as f64 / (n_bars - 1) as f64
        } else {
            1.0
        };
        zones.push(Zone {
            low,
            high,
            mid: (low + high) / 2.0,
            touches,
            last_touch_idx: last,
            strength: round2(0.6 * touches.min(4) as f64 / 4.0 + 0.4 * recency),
        });
        i = j;
    }
    zones
}

/// What the bars after a pool did to it: a close beyond `price` broke it, a wick beyond
/// it with no such close swept it (the stops were taken and price came back).
fn pool_status(after: &[Candle], price: f64, above: bool) -> &'static str {
    let beyond = |v: f64| if above { v > price } else { v < price };
    if after.iter().any(|c| beyond(c.close)) {
        "broken"
    } else if after.iter().any(|c| beyond(if above { c.high } else { c.low })) {
        "swept"
    } else {
        "intact"
    }
}

/// Equal highs and equal lows: clusters of two or more swings of one kind within `tol`.
///
/// ponytail: the status reads only the bars after the LAST touch; a sweep between the
/// first and last touch is not looked for. Check the run between touches if pools that
/// were already raided start reading `intact`.
fn liquidity_pools(candles: &[Candle], swings: &[SwingPoint], tol: f64) -> Vec<LiquidityPool> {
    let mut out = Vec::new();
    for (kind, side) in [(SwingKind::Peak, "equal_highs"), (SwingKind::Trough, "equal_lows")] {
        let above = kind == SwingKind::Peak;
        let points: Vec<(usize, f64)> =
            swings.iter().filter(|s| s.kind == kind).map(|s| (s.idx, s.price)).collect();
        for z in cluster_zones(&points, tol, candles.len()) {
            if z.touches < 2 {
                continue;
            }
            let price = if above { z.high } else { z.low };
            let after = &candles[(z.last_touch_idx + 1).min(candles.len())..];
            out.push(LiquidityPool {
                side,
                price,
                touches: z.touches,
                last_touch_idx: z.last_touch_idx,
                status: pool_status(after, price, above),
            });
        }
    }
    out
}

/// Every three-bar fair value gap at least `min_size` wide, oldest first, with how much
/// of it later bars filled — filled ones included (the caller drops those): a later bar
/// can only fill more, never less, which is what makes the fraction comparable between
/// two reads.
pub fn fair_value_gaps(candles: &[Candle], min_size: f64) -> Vec<FairValueGap> {
    let min_size = if min_size.is_finite() && min_size > 0.0 { min_size } else { 0.0 };
    let mut out = Vec::new();
    for i in 2..candles.len() {
        let (first, third) = (&candles[i - 2], &candles[i]);
        let (direction, low, high) = if first.high < third.low {
            ("bullish", first.high, third.low)
        } else if first.low > third.high {
            ("bearish", third.high, first.low)
        } else {
            continue;
        };
        let size = high - low;
        if !(low.is_finite() && high.is_finite() && size > 0.0) || size < min_size {
            continue;
        }
        // How deep later bars traded back in: a bullish gap fills from its top down (price
        // returns from above), a bearish one from its bottom up.
        let depth = candles[i + 1..].iter().fold(0.0f64, |deepest, later| {
            deepest.max(if direction == "bullish" { high - later.low } else { later.high - low })
        });
        out.push(FairValueGap {
            direction,
            low,
            high,
            idx: i - 1,
            filled: round2((depth / size).clamp(0.0, 1.0)),
        });
    }
    out
}

/// The order block of each of the newest `ORDER_BLOCK_BREAKS` breaks of `skel`: the
/// last opposite-colour candle between the broken swing and the breaking bar.
///
/// ponytail: "last opposite candle before the break" — a small counter candle inside the
/// displacement leg is picked over the one at the turn. Walk back to the leg's extreme
/// first if blocks start sitting mid-move.
fn order_blocks(candles: &[Candle], skel: &[SwingPoint], right: usize) -> Vec<OrderBlock> {
    let breaks = structure::structure_breaks(candles, skel, right);
    let newest = &breaks[breaks.len().saturating_sub(ORDER_BLOCK_BREAKS)..];
    let mut out: Vec<OrderBlock> = Vec::new();
    for b in newest {
        let bullish = b.direction == "bullish";
        let opposite = |c: &Candle| if bullish { c.is_bearish() } else { c.is_bullish() };
        let Some(idx) = (b.swing_idx..b.break_idx).rev().find(|&k| opposite(&candles[k])) else {
            continue;
        };
        if out.iter().any(|o| o.idx == idx) {
            continue;
        }
        let block = &candles[idx];
        let mitigated = candles[b.break_idx + 1..]
            .iter()
            .any(|c| if bullish { c.low <= block.high } else { c.high >= block.low });
        out.push(OrderBlock {
            direction: b.direction,
            low: block.low,
            high: block.high,
            idx,
            break_idx: b.break_idx,
            mitigated,
        });
    }
    out
}

/// Support lines through swing lows and resistance lines through swing highs.
///
/// Every pair of pivots of a kind defines a candidate line. It is kept when at least
/// `MIN_TRENDLINE_TOUCHES` pivots from the first one on sit within `tol` of it, no
/// close crossed it (by more than `tol`) from the first touch to the last, and it still
/// projects to a positive price at the last bar; a close across it after the last touch
/// makes it `broken`. Per kind, the most-touched lines
/// win (then the most recently touched, then respected over broken), skipping any that
/// project to within `tol` of one already kept — most pairs redraw the same line.
///
/// ponytail: O(m²·(m+n)) over the m pivots of a kind — about a million steps per frame
/// at 500 bars. Index the closes by line if frames ever grow by 10×.
fn trendlines(candles: &[Candle], swings: &[SwingPoint], tol: f64) -> Vec<Trendline> {
    let n = candles.len();
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    for (kind, name) in [(SwingKind::Trough, "support"), (SwingKind::Peak, "resistance")] {
        let support = kind == SwingKind::Trough;
        let pivots: Vec<SwingPoint> = swings.iter().filter(|s| s.kind == kind).copied().collect();
        let mut lines: Vec<Trendline> = Vec::new();
        for a in 0..pivots.len() {
            for b in a + 1..pivots.len() {
                let (pa, pb) = (pivots[a], pivots[b]);
                if pb.idx <= pa.idx {
                    continue;
                }
                let slope = (pb.price - pa.price) / (pb.idx - pa.idx) as f64;
                let at = |x: usize| pa.price + slope * (x as f64 - pa.idx as f64);
                let touches: Vec<usize> = pivots[a..]
                    .iter()
                    .filter(|p| (p.price - at(p.idx)).abs() <= tol)
                    .map(|p| p.idx)
                    .collect();
                if touches.len() < MIN_TRENDLINE_TOUCHES {
                    continue;
                }
                let last_touch = touches.iter().copied().max().unwrap_or(pa.idx);
                let through = |x: usize| {
                    let close = candles[x].close;
                    if support { close < at(x) - tol } else { close > at(x) + tol }
                };
                if (pa.idx..=last_touch).any(|x| through(x)) {
                    continue;
                }
                // A line that projects to no positive price at the last bar is not a
                // level anyone can trade. Measured live 2026-10-25 (merge-yash tool test):
                // on BTCUSDT spot 1W (SWING / INVESTOR key levels) a 5-touch support line
                // falling 169.6 a week projected to -18,005 and was listed to the model as
                // "respected" (the perp's shorter 1W history drew no such line). Dropping
                // it here lets the next-best line take its slot.
                if !(at(n - 1) > 0.0) {
                    continue;
                }
                let broken = (last_touch + 1..n).any(|x| through(x));
                lines.push(Trendline {
                    kind: name,
                    slope_per_bar: slope,
                    touches: touches.len(),
                    first_touch_idx: pa.idx,
                    first_touch_price: pa.price,
                    last_touch_idx: last_touch,
                    price_at_last: at(n - 1),
                    status: if broken { "broken" } else { "respected" },
                });
            }
        }
        lines.sort_by(|x, y| {
            y.touches
                .cmp(&x.touches)
                .then(y.last_touch_idx.cmp(&x.last_touch_idx))
                .then((x.status != "respected").cmp(&(y.status != "respected")))
        });
        let mut kept: Vec<Trendline> = Vec::new();
        for line in lines {
            if kept.len() == TRENDLINES_PER_KIND {
                break;
            }
            if kept.iter().all(|k| (k.price_at_last - line.price_at_last).abs() > tol) {
                kept.push(line);
            }
        }
        out.extend(kept);
    }
    out
}

/// The round numbers around `price`: the step is one tenth of its order of magnitude
/// (1,000 at 65,010; 0.1 at 2.34; 0.00001 at 0.0001234), and the result is the two steps
/// at or below price and the two above. Empty for a non-finite or non-positive price.
pub fn round_numbers(price: f64) -> Vec<f64> {
    if !(price.is_finite() && price > 0.0) {
        return Vec::new();
    }
    let step = 10f64.powi(price.log10().floor() as i32 - 1);
    let base = (price / step).floor();
    [base - 1.0, base, base + 1.0, base + 2.0]
        .iter()
        .map(|k| k * step)
        .filter(|v| *v > 0.0)
        .collect()
}

/// VWAP from bar `anchor` to the end: Σ(typical price × volume) / Σ volume, with the
/// typical price (H+L+C)/3 that tool-server's `candles::session_vwap` uses. `None` when
/// `anchor` is past the end or the bars carry no volume.
///
/// New and separate on purpose: `IndicatorState::compute_vwap` (the whole window's VWAP)
/// is unchanged. A thin wrapper over `anchored_vwap_bands`, whose first pass is this
/// function's old loop, so the VWAP it returns is the same number to the bit.
pub fn anchored_vwap(candles: &[Candle], anchor: usize) -> Option<f64> {
    anchored_vwap_bands(candles, anchor).map(|(vwap, _)| vwap)
}

/// `anchored_vwap` and its volume-weighted standard deviation, `(vwap, sigma)`, with
/// sigma = sqrt(Σ volume × (typical − vwap)² / Σ volume): the ±1σ band a charting
/// platform draws around an anchored VWAP (pro-trader Task 4, get_levels_ladder's
/// `vwap_bands`). `None` exactly when `anchored_vwap` is.
///
/// Two passes on purpose: the one-pass form Σv·tp²/Σv − vwap² subtracts two numbers
/// near 4.5e9 at a 67,000 BTC price to get a variance that is often only 1e2-1e5, so
/// the subtraction throws away about 5-8 of f64's 16 digits, and on a flat run (variance ~0)
/// it can come out negative, whose square root is NaN. Subtracting the VWAP first keeps
/// every term small and non-negative.
pub fn anchored_vwap_bands(candles: &[Candle], anchor: usize) -> Option<(f64, f64)> {
    let bars = candles.get(anchor..)?;
    let typical = |c: &Candle| (c.high + c.low + c.close) / 3.0;
    let (mut pv, mut vol) = (0.0, 0.0);
    for c in bars {
        pv += typical(c) * c.volume;
        vol += c.volume;
    }
    let vwap = pv / vol;
    if !(vol > 0.0 && vwap.is_finite()) {
        return None;
    }
    let spread: f64 = bars.iter().map(|c| c.volume * (typical(c) - vwap).powi(2)).sum();
    Some((vwap, (spread / vol).sqrt()))
}

/// Every key level of one candle series. `None` only for an empty series or a
/// non-finite last close; short, flat or gappy input yields empty lists.
pub fn analyze_levels(candles: &[Candle]) -> Option<KeyLevels> {
    let close = candles.last()?.close;
    if !close.is_finite() {
        return None;
    }
    let atr = atr_14(candles);
    let tol = |k: f64| atr.map_or(0.0, |a| a * k);
    let cfg = StructureConfig::default();
    let n = candles.len();

    // Raw 3/3 fractals for zones, pools and trendlines: a retest smaller than the ATR
    // leg filter is still a touch. Order blocks use the filtered skeleton, so their
    // breaks are exactly the BOS / CHoCH get_market_structure reports.
    let swings = structure::find_swings_with(candles, cfg.left, cfg.right);
    let points: Vec<(usize, f64)> = swings.iter().map(|s| (s.idx, s.price)).collect();
    let mut zones = cluster_zones(&points, tol(ZONE_ATR), n);
    // A single-touch zone is noise unless it is a dealing-range edge.
    let edge = [SwingKind::Peak, SwingKind::Trough]
        .iter()
        .filter_map(|k| swings.iter().rev().find(|s| s.kind == *k).map(|s| s.idx))
        .min();
    zones.retain(|z| z.touches >= 2 || edge.is_some_and(|e| z.last_touch_idx >= e));

    let mut fvgs = fair_value_gaps(candles, tol(FVG_MIN_ATR));
    fvgs.retain(|g| g.filled < 1.0);

    let skel = structure::skeleton(candles, &cfg, atr);
    let swing_vwap = skel.last().and_then(|s| anchored_vwap(candles, s.idx).map(|v| (s.idx, v)));

    Some(KeyLevels {
        atr,
        zones,
        pools: liquidity_pools(candles, &swings, tol(EQUAL_ATR)),
        fvgs,
        order_blocks: order_blocks(candles, &skel, cfg.right),
        trendlines: trendlines(candles, &swings, tol(TRENDLINE_ATR)),
        swing_vwap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // `candle`, `path` and `walk` are copies of the private helpers in
    // `structure::tests`, so the fixtures here are the bars Task 1 traced by hand.
    fn candle(open: f64, high: f64, low: f64, close: f64) -> Candle {
        Candle { open, high, low, close, volume: 10.0, taker_buy_volume: None }
    }

    /// Candles that walk linearly between `pivots`, `per_leg` bars per leg, with a wick
    /// that grows by 0.001 per bar so two bars meeting at a pivot never tie.
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

    #[test]
    fn zones_merge_within_the_tolerance_and_split_beyond_it() {
        let zones = cluster_zones(&[(1, 100.0), (5, 100.2), (9, 100.4), (12, 101.0)], 0.5, 20);
        assert_eq!(zones.len(), 2, "{zones:?}");
        assert_eq!((zones[0].low, zones[0].high, zones[0].touches, zones[0].last_touch_idx), (100.0, 100.4, 3, 9));
        assert!((zones[0].mid - 100.2).abs() < 1e-9);
        // 0.6 × 3/4 + 0.4 × 9/19 = 0.639…
        assert_eq!(zones[0].strength, 0.64);
        assert_eq!((zones[1].low, zones[1].high, zones[1].touches, zones[1].last_touch_idx), (101.0, 101.0, 1, 12));

        // No tolerance: only equal prices merge, and a NaN is not a touch.
        let exact = cluster_zones(&[(1, 5.0), (2, 5.0), (3, 6.0), (4, f64::NAN)], f64::NAN, 4);
        assert_eq!(exact.iter().map(|z| z.touches).collect::<Vec<_>>(), vec![2, 1]);
        assert!(cluster_zones(&[], 1.0, 0).is_empty());
    }

    #[test]
    fn a_wick_beyond_a_pool_sweeps_it_and_a_close_beyond_breaks_it() {
        let after = vec![candle(100.0, 101.0, 99.0, 100.5), candle(100.5, 102.5, 100.0, 101.0)];
        assert_eq!(pool_status(&after, 102.0, true), "swept");
        assert_eq!(pool_status(&after, 100.8, true), "broken");
        assert_eq!(pool_status(&after, 103.0, true), "intact");
        assert_eq!(pool_status(&after, 99.5, false), "swept");
        assert_eq!(pool_status(&after, 100.7, false), "broken");
        assert_eq!(pool_status(&after, 98.0, false), "intact");
        assert_eq!(pool_status(&[], 1.0, true), "intact");
    }

    #[test]
    fn a_double_top_is_an_intact_equal_highs_pool() {
        // Highs at bars 6 (110.106) and 18 (110.118): 0.012 apart, inside 0.1 ATR.
        let candles = path(&[100.0, 110.0, 104.0, 110.0, 104.0], 6);
        let kl = analyze_levels(&candles).unwrap();
        assert_eq!(kl.pools.len(), 1, "{:?}", kl.pools);
        let pool = &kl.pools[0];
        assert_eq!((pool.side, pool.touches, pool.last_touch_idx, pool.status), ("equal_highs", 2, 18, "intact"));
        assert!((pool.price - 110.118).abs() < 1e-9);
    }

    #[test]
    fn fair_value_gaps_and_their_fill() {
        // Bullish: bar 0's high 101 sits below bar 2's low 102 — a gap 101..102.
        let mut c = vec![
            candle(100.0, 101.0, 99.0, 100.8),
            candle(100.8, 103.5, 100.6, 103.2),
            candle(103.2, 104.0, 102.0, 103.8),
        ];
        let g = fair_value_gaps(&c, 0.0);
        assert_eq!(g.len(), 1);
        assert_eq!((g[0].direction, g[0].low, g[0].high, g[0].idx, g[0].filled), ("bullish", 101.0, 102.0, 1, 0.0));
        assert!(fair_value_gaps(&c, 1.5).is_empty(), "narrower than the minimum");
        // A later low at 101.5 fills half of it, one at 100.9 all of it.
        c.push(candle(103.8, 104.0, 101.5, 103.0));
        assert_eq!(fair_value_gaps(&c, 0.0)[0].filled, 0.5);
        c.push(candle(103.0, 103.2, 100.9, 101.0));
        assert_eq!(fair_value_gaps(&c, 0.0)[0].filled, 1.0);

        // Bearish: bar 0's low 99 sits above bar 2's high 98; bar 3's high 98.5 fills half.
        let c = vec![
            candle(100.0, 101.0, 99.0, 99.2),
            candle(99.2, 99.4, 96.5, 96.8),
            candle(96.8, 98.0, 96.0, 96.2),
            candle(96.2, 98.5, 96.1, 98.3),
        ];
        let g = fair_value_gaps(&c, 0.0);
        assert_eq!(g.len(), 1);
        assert_eq!((g[0].direction, g[0].low, g[0].high, g[0].idx, g[0].filled), ("bearish", 98.0, 99.0, 1, 0.5));
    }

    #[test]
    fn order_blocks_are_the_last_down_candle_before_each_bullish_break() {
        // Task 1's uptrend: bullish BOS at bars 15, 27 and 40, through the highs at 6,
        // 18 and 30. The last bearish candle before each is the bottom of the pullback.
        let candles = path(&[100.0, 110.0, 104.0, 116.0, 109.0, 122.0, 115.0, 125.0], 6);
        let kl = analyze_levels(&candles).unwrap();
        let blocks: Vec<(usize, &str, usize, bool)> =
            kl.order_blocks.iter().map(|o| (o.idx, o.direction, o.break_idx, o.mitigated)).collect();
        assert_eq!(blocks, vec![(11, "bullish", 15, false), (23, "bullish", 27, false), (35, "bullish", 40, false)]);
        let newest = kl.order_blocks.last().unwrap();
        assert!((newest.low - 114.865).abs() < 1e-3 && (newest.high - 116.302).abs() < 1e-3, "{newest:?}");

        // Falling back to 113 trades into the newest block and closes below the 114.864
        // low: a bearish CHoCH at bar 47, whose block is the last up candle (bar 41).
        let candles = path(&[100.0, 110.0, 104.0, 116.0, 109.0, 122.0, 115.0, 125.0, 113.0], 6);
        let kl = analyze_levels(&candles).unwrap();
        let blocks: Vec<(usize, &str, bool)> =
            kl.order_blocks.iter().map(|o| (o.idx, o.direction, o.mitigated)).collect();
        assert_eq!(
            blocks,
            vec![(11, "bullish", false), (23, "bullish", false), (35, "bullish", true), (41, "bearish", false)]
        );
    }

    #[test]
    fn three_rising_lows_draw_a_support_line_that_a_crash_breaks() {
        // Lows at bars 12 (102.888), 24 (105.876) and 36 (108.864): exactly collinear,
        // 0.249 a bar, and every close sits above the line.
        let candles = path(&[100.0, 106.0, 103.0, 109.0, 106.0, 112.0, 109.0, 115.0], 6);
        let kl = analyze_levels(&candles).unwrap();
        let support: Vec<&Trendline> = kl.trendlines.iter().filter(|t| t.kind == "support").collect();
        assert_eq!(support.len(), 1, "the three pairs redraw one line: {:?}", kl.trendlines);
        let s = support[0];
        assert_eq!((s.touches, s.first_touch_idx, s.last_touch_idx, s.status), (3, 12, 36, "respected"));
        assert!((s.slope_per_bar - 0.249).abs() < 1e-9, "{s:?}");
        assert!((s.price_at_last - 110.109).abs() < 1e-6, "projected to bar 41: {s:?}");

        // A leg down to 100 closes through it (110 against 110.6 at bar 43).
        let crashed = path(&[100.0, 106.0, 103.0, 109.0, 106.0, 112.0, 109.0, 115.0, 100.0], 6);
        let kl = analyze_levels(&crashed).unwrap();
        let s = kl.trendlines.iter().find(|t| t.kind == "support").expect("still drawn, now broken");
        assert_eq!((s.touches, s.first_touch_idx, s.status), (3, 12, "broken"));
    }

    #[test]
    fn a_line_projecting_below_zero_is_not_drawn() {
        // The highs at bars 6 (46), 18 (36), 30 (26) and 42 (16) fall 0.832 a bar: a
        // 4-touch resistance line (broken by the closes above it after bar 42) that
        // projects to about -3.0 at bar 65, the last, and was drawn before the guard
        // (checked by disabling it). The rising lows at bars 36 (10), 48 (13) and
        // 60 (16) still draw their support line.
        let candles = path(&[40.0, 46.0, 30.0, 36.0, 20.0, 26.0, 10.0, 16.0, 13.0, 19.0, 16.0, 22.0], 6);
        let kl = analyze_levels(&candles).unwrap();
        assert!(kl.trendlines.iter().all(|t| t.price_at_last > 0.0), "{:?}", kl.trendlines);
        assert!(kl.trendlines.iter().all(|t| t.kind == "support"), "{:?}", kl.trendlines);
        assert_eq!(kl.trendlines.len(), 1, "{:?}", kl.trendlines);
        let s = &kl.trendlines[0];
        assert_eq!((s.touches, s.first_touch_idx, s.last_touch_idx, s.status), (3, 36, 60, "respected"));
        assert!(s.slope_per_bar > 0.0, "{s:?}");
    }

    #[test]
    fn round_numbers_step_with_the_price() {
        assert_eq!(round_numbers(65010.0), vec![64000.0, 65000.0, 66000.0, 67000.0]);
        let small = round_numbers(1.234e-4);
        assert_eq!(small.len(), 4);
        for (got, want) in small.iter().zip([1.1e-4, 1.2e-4, 1.3e-4, 1.4e-4]) {
            assert!((got - want).abs() < 1e-12, "{small:?}");
        }
        assert!(round_numbers(0.0).is_empty() && round_numbers(-5.0).is_empty() && round_numbers(f64::NAN).is_empty());
    }

    #[test]
    fn anchored_vwap_weights_typical_price_by_volume() {
        let c = vec![
            Candle { open: 10.0, high: 11.0, low: 9.0, close: 10.0, volume: 1.0, taker_buy_volume: None },
            Candle { open: 10.0, high: 13.0, low: 11.0, close: 12.0, volume: 3.0, taker_buy_volume: None },
        ];
        // Typical prices 10 and 12: (10 × 1 + 12 × 3) / 4.
        assert_eq!(anchored_vwap(&c, 0), Some(11.5));
        assert_eq!(anchored_vwap(&c, 1), Some(12.0));
        assert_eq!(anchored_vwap(&c, 2), None);
        assert_eq!(anchored_vwap(&c, 3), None);
        let silent = vec![Candle { volume: 0.0, ..c[0].clone() }];
        assert_eq!(anchored_vwap(&silent, 0), None);
    }

    #[test]
    fn anchored_vwap_bands_are_the_volume_weighted_spread_around_the_vwap() {
        let c = vec![
            Candle { open: 10.0, high: 11.0, low: 9.0, close: 10.0, volume: 1.0, taker_buy_volume: None },
            Candle { open: 10.0, high: 13.0, low: 11.0, close: 12.0, volume: 3.0, taker_buy_volume: None },
        ];
        // Typical prices 10 and 12 around the 11.5 VWAP: (1 × 1.5² + 3 × 0.5²) / 4 = 0.75.
        let (vwap, sigma) = anchored_vwap_bands(&c, 0).unwrap();
        assert_eq!(vwap, 11.5);
        assert!((sigma - 0.75f64.sqrt()).abs() < 1e-12, "{sigma}");
        assert_eq!(anchored_vwap(&c, 0), Some(vwap), "the wrapper serves the same VWAP");
        assert_eq!(anchored_vwap_bands(&c, 1), Some((12.0, 0.0)), "one bar has no spread");
        assert_eq!(anchored_vwap_bands(&c, 2), None);
        assert_eq!(anchored_vwap_bands(&[Candle { volume: 0.0, ..c[0].clone() }], 0), None);

        // BTC-sized: typical prices 67,000 / 67,002 / 67,004 on volumes 1 / 2 / 1 give a
        // 67,002 VWAP and a variance of (1 × 2² + 2 × 0 + 1 × 2²) / 4 = 2.
        let at = |tp: f64, volume: f64| Candle { open: tp, high: tp, low: tp, close: tp, volume, taker_buy_volume: None };
        let (vwap, sigma) = anchored_vwap_bands(&[at(67_000.0, 1.0), at(67_002.0, 2.0), at(67_004.0, 1.0)], 0).unwrap();
        assert_eq!(vwap, 67_002.0);
        assert!((sigma - 2f64.sqrt()).abs() < 1e-9, "{sigma}");

        // A flat run at a price no binary fraction holds exactly: the spread is ~0, never NaN.
        let flat: Vec<Candle> = [0.3, 0.7, 1.1, 0.01].iter().map(|v| at(67_000.1, *v)).collect();
        let (vwap, sigma) = anchored_vwap_bands(&flat, 0).unwrap();
        assert!((vwap - 67_000.1).abs() < 1e-6, "{vwap}");
        assert!(sigma.is_finite() && (0.0..1e-6).contains(&sigma), "{sigma}");
    }

    #[test]
    fn short_and_flat_series_degrade_to_empty_lists() {
        assert!(analyze_levels(&[]).is_none());
        assert!(analyze_levels(&[candle(1.0, 1.0, 1.0, f64::NAN)]).is_none());

        let one = analyze_levels(&[candle(1.0, 1.1, 0.9, 1.0)]).unwrap();
        assert!(one.atr.is_none());
        assert!(one.zones.is_empty() && one.pools.is_empty() && one.fvgs.is_empty());
        assert!(one.order_blocks.is_empty() && one.trendlines.is_empty() && one.swing_vwap.is_none());

        let flat: Vec<Candle> = (0..60).map(|_| candle(50.0, 50.0, 50.0, 50.0)).collect();
        let kl = analyze_levels(&flat).unwrap();
        assert!(kl.atr.is_none(), "a zero ATR is no ATR");
        assert!(kl.zones.is_empty() && kl.pools.is_empty() && kl.fvgs.is_empty(), "ties disqualify every swing");
        assert!(kl.order_blocks.is_empty() && kl.trendlines.is_empty() && kl.swing_vwap.is_none());
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
        fn zones_partition_the_points_and_never_exceed_the_tolerance(
            points in prop::collection::vec((0usize..500, -1000.0f64..1000.0), 0..40),
            tol in 0.0f64..50.0
        ) {
            let zones = cluster_zones(&points, tol, 500);
            prop_assert_eq!(zones.iter().map(|z| z.touches).sum::<usize>(), points.len());
            for z in &zones {
                prop_assert!(z.high - z.low <= tol);
                prop_assert!(z.low <= z.mid && z.mid <= z.high);
                prop_assert!((0.0..=1.0).contains(&z.strength));
                prop_assert!(points.iter().any(|p| p.0 == z.last_touch_idx));
            }
            for w in zones.windows(2) {
                prop_assert!(w[0].high < w[1].low, "zones overlap: {:?}", w);
            }
        }

        #[test]
        fn pools_have_two_touches_and_the_status_the_bars_after_them_give(candles in walk()) {
            let swings = structure::find_swings_with(&candles, 3, 3);
            let tol = atr_14(&candles).map_or(0.0, |a| a * EQUAL_ATR);
            for p in liquidity_pools(&candles, &swings, tol) {
                prop_assert!(p.touches >= 2);
                prop_assert!(swings.iter().any(|s| s.price == p.price && s.idx <= p.last_touch_idx));
                let after = &candles[p.last_touch_idx + 1..];
                let above = p.side == "equal_highs";
                let closed = after.iter().any(|c| if above { c.close > p.price } else { c.close < p.price });
                let wicked = after.iter().any(|c| if above { c.high > p.price } else { c.low < p.price });
                let want = if closed { "broken" } else if wicked { "swept" } else { "intact" };
                prop_assert_eq!(p.status, want);
            }
        }

        #[test]
        fn a_gap_only_fills_further_as_bars_arrive(candles in walk(), split in 0usize..300) {
            let k = split.min(candles.len());
            let full = fair_value_gaps(&candles, 0.0);
            for g in fair_value_gaps(&candles[..k], 0.0) {
                let later = full.iter().find(|f| f.idx == g.idx);
                prop_assert!(later.is_some(), "a gap vanished: {:?}", g);
                let later = later.unwrap();
                prop_assert_eq!((later.direction, later.low, later.high), (g.direction, g.low, g.high));
                prop_assert!(later.filled >= g.filled);
            }
            for g in &full {
                prop_assert!((0.0..=1.0).contains(&g.filled));
                prop_assert!(g.low < g.high);
            }
        }

        #[test]
        fn trendlines_have_three_touches_and_no_close_through_while_respected(candles in walk()) {
            let Some(kl) = analyze_levels(&candles) else { return Ok(()); };
            let tol = kl.atr.map_or(0.0, |a| a * TRENDLINE_ATR);
            let n = candles.len();
            for t in &kl.trendlines {
                prop_assert!(t.touches >= MIN_TRENDLINE_TOUCHES);
                prop_assert!(t.price_at_last > 0.0, "a line projecting to no positive price: {:?}", t);
                prop_assert!(t.first_touch_idx < t.last_touch_idx && t.last_touch_idx < n);
                // The line exactly as `trendlines` evaluates it.
                let at = |x: usize| t.first_touch_price + t.slope_per_bar * (x as f64 - t.first_touch_idx as f64);
                let through = |x: usize| {
                    if t.kind == "support" { candles[x].close < at(x) - tol } else { candles[x].close > at(x) + tol }
                };
                prop_assert!(!(t.first_touch_idx..=t.last_touch_idx).any(|x| through(x)));
                let crossed_after = (t.last_touch_idx + 1..n).any(|x| through(x));
                prop_assert_eq!(t.status == "broken", crossed_after);
                prop_assert_eq!(t.status == "respected", !crossed_after);
            }
        }

        #[test]
        fn analyze_levels_is_total_and_deterministic_and_blocks_are_the_last_opposite_candle(candles in walk()) {
            let a = analyze_levels(&candles);
            prop_assert_eq!(a.is_none(), candles.is_empty());
            prop_assert_eq!(&a, &analyze_levels(&candles));
            let Some(kl) = a else { return Ok(()); };
            for ob in &kl.order_blocks {
                let opposite = |c: &Candle| if ob.direction == "bullish" { c.is_bearish() } else { c.is_bullish() };
                prop_assert!(ob.idx < ob.break_idx && ob.break_idx < candles.len());
                prop_assert!(opposite(&candles[ob.idx]));
                prop_assert!(!candles[ob.idx + 1..ob.break_idx].iter().any(|c| opposite(c)));
            }
            for g in &kl.fvgs {
                prop_assert!(g.filled < 1.0);
            }
        }

        #[test]
        fn anchored_vwap_sits_inside_the_range_it_averages(candles in walk(), anchor in 0usize..300) {
            match anchored_vwap(&candles, anchor) {
                // walk() bars always carry volume, so only an anchor past the end is None.
                None => prop_assert!(anchor >= candles.len()),
                Some(v) => {
                    let bars = &candles[anchor..];
                    let lo = bars.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
                    let hi = bars.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
                    prop_assert!(v >= lo - 1e-9 && v <= hi + 1e-9, "{} outside {}..{}", v, lo, hi);
                }
            }
        }
    }
}
