// score.rs — the pure half of the radar: candles in, the ranked board frame out.
//
// No I/O, no clock (the caller passes `now_ms`), so every rule here is testable from
// fixtures. The detectors are crypto-app's quant-core, copied verbatim into
// `quant-core/`: `analyze_levels` (order blocks, fair value gaps, liquidity pools) and
// `analyze_structure` (BOS / CHoCH, EMA stack, ATR). This file only decides what those
// readings are worth and how a row is shown.

use quant_core::levels::{OrderBlock, analyze_levels};
use quant_core::patterns::Candle;
use quant_core::structure::{MarketStructure, StructureBreak, StructureConfig, analyze_structure};
use serde::Serialize;

pub const BAR_MS: i64 = 15 * 60 * 1000;
pub const HOUR_MS: i64 = 60 * 60 * 1000;

// ── Weights and thresholds ───────────────────────────────────────────────────
// ponytail: every number in this block is a first-guess heuristic, not a fitted
// parameter. Tune them after watching the live board for a few sessions (which rows
// moved, which were noise); the factor shape is what was agreed, the sizes are not.

/// Closed 15m bars fed to the 15m detectors (~5 days).
const WINDOW_15M: usize = 500;
/// Fewer closed 15m bars than this and the symbol is skipped (new listing, gap).
/// `pub` because the poller re-backfills any series shorter than this.
pub const MIN_BARS: usize = 300;
/// Last bar's open time older than this and the symbol is skipped (polls failing).
const STALE_MS: i64 = 30 * 60 * 1000;

const TREND_BREAK_PTS: f64 = 15.0;
const TREND_EMA_PTS: f64 = 10.0;
/// A 1h break older than this many 1h bars no longer counts as the trend (3 days).
const TREND_MAX_AGE: usize = 72;

const BOS_PTS: f64 = 20.0;
const CHOCH_PTS: f64 = 15.0;
/// A 15m break keeps full points up to this age (6 h) ...
const BREAK_FULL_AGE: usize = 24;
/// ... and fades linearly to 0 at this one (12 h).
const BREAK_ZERO_AGE: usize = 48;

const ZONE_INSIDE_PTS: f64 = 20.0;
const ZONE_1ATR_PTS: f64 = 15.0;
const ZONE_2ATR_PTS: f64 = 8.0;
/// An order block whose first retest is at most this many closed bars old still counts.
const OB_FRESH_TOUCH_BARS: usize = 3;

/// Momentum and OI compare against the close this many bars back (4 h).
const LOOKBACK_4H: usize = 16;
const MOMENTUM_MAX_ATR: f64 = 3.0;
const MOMENTUM_PTS_PER_ATR: f64 = 5.0;

const OI_PRICE_EDGE_PCT: f64 = 0.5;
const OI_EDGE_PCT: f64 = 1.0;
const OI_NEW_POSITIONS_PTS: f64 = 10.0;
const OI_COVERING_PTS: f64 = 4.0;

const SWEEP_PTS: f64 = 5.0;
const SWEEP_MAX_AGE: usize = 24;
const LIQ_PTS: f64 = 5.0;
/// Net liquidations in the last hour must reach this fraction of 24h quote volume.
const LIQ_NET_OF_VOLUME: f64 = 0.0005;

/// Choppy gate: efficiency ratio below this over `CHOP_BARS`, and no 15m break within
/// `BREAK_ZERO_AGE`, scores the symbol 0 on both sides.
const CHOP_BARS: usize = 32;
const CHOP_MAX_ER: f64 = 0.25;

const DAY_BARS: usize = 96;

/// Extended: a 4h move of at least this many ATR in the side's favour (where momentum
/// already scores in full) with NO live zone within 2 ATR (the zone factor scored 0).
/// Strong, but nothing to enter at now — listed apart so the main board is setups a
/// trader can act on, and these are the ones to wait on for a pullback (or a bounce).
const EXTENDED_MIN_MOVE_ATR: f64 = MOMENTUM_MAX_ATR;
/// Extended rows kept per side.
const EXTENDED_TOP_N: usize = 10;

// ── Inputs ───────────────────────────────────────────────────────────────────

/// One kline: open time, the candle, and its quote-asset (USDT) volume.
#[derive(Debug, Clone)]
pub struct Bar {
    pub t: i64,
    pub c: Candle,
    pub qv: f64,
}

/// USD liquidated in a window: `long_usd` = longs force-sold, `short_usd` = shorts
/// force-bought.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LiqWindow {
    pub long_usd: f64,
    pub short_usd: f64,
}

pub struct SymbolInputs<'a> {
    /// Bare Binance symbol, e.g. `SOLUSDT`.
    pub symbol: &'a str,
    /// 15m bars, oldest first; the last one may still be forming.
    pub bars: &'a [Bar],
    pub now_ms: i64,
    pub oi_4h_change_pct: Option<f64>,
    /// `None` until the liquidation feed has an hour of history.
    pub liq_1h: Option<LiqWindow>,
}

pub struct Cfg {
    pub top_n: usize,
    pub min_score: u32,
}

/// Parse a Binance `/fapi/v1/klines` body. Rows with a non-finite field are dropped
/// rather than turned into zeros, which would read as a real price.
pub fn parse_klines(body: &str) -> Result<Vec<Bar>, String> {
    let rows: Vec<Vec<serde_json::Value>> =
        serde_json::from_str(body).map_err(|e| format!("klines: {e}"))?;
    let num = |v: &serde_json::Value| -> Option<f64> {
        v.as_str().and_then(|s| s.parse::<f64>().ok()).or_else(|| v.as_f64()).filter(|x| x.is_finite())
    };
    Ok(rows
        .iter()
        .filter_map(|r| {
            let t = r.first()?.as_i64()?;
            let c = Candle {
                open: num(r.get(1)?)?,
                high: num(r.get(2)?)?,
                low: num(r.get(3)?)?,
                close: num(r.get(4)?)?,
                volume: num(r.get(5)?)?,
                taker_buy_volume: r.get(9).and_then(num),
            };
            Some(Bar { t, c, qv: num(r.get(7)?)? })
        })
        .collect())
}

/// Group 15m bars into hour buckets (`t / HOUR_MS`). A partial FIRST bucket (fewer than
/// four bars) is dropped, because its open is not the hour's open; the last bucket is
/// kept even when partial — whether it is complete is the caller's question.
pub fn to_hourly(bars: &[Bar]) -> Vec<Bar> {
    let mut out: Vec<(Bar, usize)> = Vec::new();
    for b in bars {
        let hour = b.t.div_euclid(HOUR_MS) * HOUR_MS;
        match out.last_mut() {
            Some((h, n)) if h.t == hour => {
                h.c.high = h.c.high.max(b.c.high);
                h.c.low = h.c.low.min(b.c.low);
                h.c.close = b.c.close;
                h.c.volume += b.c.volume;
                h.c.taker_buy_volume = match (h.c.taker_buy_volume, b.c.taker_buy_volume) {
                    (Some(x), Some(y)) => Some(x + y),
                    _ => None,
                };
                h.qv += b.qv;
                *n += 1;
            }
            _ => out.push((Bar { t: hour, c: b.c.clone(), qv: b.qv }, 1)),
        }
    }
    if out.first().is_some_and(|(_, n)| *n < 4) {
        out.remove(0);
    }
    out.into_iter().map(|(b, _)| b).collect()
}

/// 15m bars kept per symbol: one full backfill (~10 days, ~250 1h bars for EMA-200).
pub const KEEP_BARS: usize = 1000;

/// Fold a fresh poll into a symbol's series by open time: a bar already held is
/// replaced (the forming bar keeps updating), a newer one appended. When the poll
/// does not reach back to the series' last bar (missed polls), the poll replaces the
/// series outright rather than leaving a silent hole in it.
pub fn merge_bars(series: &mut Vec<Bar>, fresh: Vec<Bar>) {
    let Some(first) = fresh.first() else { return };
    if series.last().is_none_or(|l| first.t > l.t + BAR_MS) {
        *series = fresh;
    } else {
        for b in fresh {
            match series.binary_search_by_key(&b.t, |x| x.t) {
                Ok(i) => series[i] = b,
                Err(i) => series.insert(i, b),
            }
        }
    }
    let excess = series.len().saturating_sub(KEEP_BARS);
    series.drain(..excess);
}

// ── Output ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bull,
    Bear,
}

impl Side {
    fn word(self) -> &'static str {
        match self {
            Side::Bull => "bullish",
            Side::Bear => "bearish",
        }
    }
    fn trend(self) -> &'static str {
        match self {
            Side::Bull => "up",
            Side::Bear => "down",
        }
    }
    /// +1 for bullish, -1 for bearish: turns a signed move into "in this side's favour".
    fn sign(self) -> f64 {
        match self {
            Side::Bull => 1.0,
            Side::Bear => -1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BoardZone {
    /// `order_block` or `fvg`.
    pub kind: &'static str,
    pub low: f64,
    pub high: f64,
    /// From price to the zone's near edge, in percent: negative = below price, 0 = inside.
    pub distance_pct: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Row {
    /// Always the system symbol with the `.P` perp suffix, e.g. `SOLUSDT.P`.
    pub symbol: String,
    pub score: u32,
    pub price: f64,
    pub change_24h_pct: f64,
    pub zone: Option<BoardZone>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Scored {
    /// `None` when neither side is ahead (choppy, or both sides equal).
    pub side: Option<Side>,
    pub row: Row,
    pub quote_volume_24h: f64,
    /// See `EXTENDED_MIN_MOVE_ATR`. Listed under `extended`, not the main board.
    pub extended: bool,
}

/// The extended rows of each side (see `EXTENDED_MIN_MOVE_ATR`).
#[derive(Debug, Clone, Serialize, PartialEq, Default)]
pub struct Extended {
    pub bullish: Vec<Row>,
    pub bearish: Vec<Row>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Default)]
pub struct Tilt {
    pub bullish: usize,
    pub bearish: usize,
}

/// The v1 board frame. Field names are the wire contract the frontend validates.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Board {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub v: u32,
    /// `ready`, or `warming_up` (empty lists) until the first full scoring pass.
    pub status: &'static str,
    /// UNIX seconds.
    pub as_of: i64,
    pub timeframe: &'static str,
    pub trend_timeframe: &'static str,
    pub universe: usize,
    pub scored: usize,
    pub tilt: Tilt,
    pub liquidations_ready: bool,
    /// Setups with an entry zone near price (or not stretched): the board proper.
    pub bullish: Vec<Row>,
    pub bearish: Vec<Row>,
    /// Strong but stretched away from any zone. Added to v1 without a version bump: a
    /// client that does not know the field ignores it.
    pub extended: Extended,
}

impl Board {
    pub fn warming_up(as_of: i64, universe: usize) -> Self {
        Board {
            kind: "board",
            v: 1,
            status: "warming_up",
            as_of,
            timeframe: "15m",
            trend_timeframe: "1h",
            universe,
            scored: 0,
            tilt: Tilt::default(),
            liquidations_ready: false,
            bullish: Vec::new(),
            bearish: Vec::new(),
            extended: Extended::default(),
        }
    }
}

// ── Scoring ──────────────────────────────────────────────────────────────────

/// Two decimals; `+ 0.0` turns a `-0.0` into `0.0` so the wire never says "-0".
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0 + 0.0
}

/// The most recent break of either kind.
fn latest_break(ms: &MarketStructure) -> Option<StructureBreak> {
    match (ms.last_bos, ms.last_choch) {
        (Some(a), Some(b)) => Some(if a.break_idx >= b.break_idx { a } else { b }),
        (a, b) => a.or(b),
    }
}

/// `|close_now − close_k| / Σ|Δclose|` over the last `k` bars; 0 when nothing moved.
fn efficiency_ratio(c: &[Candle], k: usize) -> f64 {
    if c.len() <= k {
        return 0.0;
    }
    let n = c.len();
    let path: f64 = (n - k..n).map(|i| (c[i].close - c[i - 1].close).abs()).sum();
    if path > 0.0 { (c[n - 1].close - c[n - 1 - k].close).abs() / path } else { 0.0 }
}

/// An order block is still tradeable when price has not come back to it yet, or came
/// back only in the last `OB_FRESH_TOUCH_BARS` closed bars without closing through its
/// far edge.
///
/// quant-core's own `mitigated` flag turns true at the FIRST touch, so "unmitigated and
/// price inside it" can never happen; the radar needs "being retested right now", which
/// is why the first touch is found here instead.
fn order_block_live(c: &[Candle], ob: &OrderBlock) -> bool {
    let bull = ob.direction == "bullish";
    let n = c.len();
    let first_touch = (ob.break_idx + 1..n).find(|&k| if bull { c[k].low <= ob.high } else { c[k].high >= ob.low });
    match first_touch {
        None => true,
        Some(k) => {
            n - 1 - k < OB_FRESH_TOUCH_BARS
                && !c[k..n].iter().any(|b| if bull { b.close < ob.low } else { b.close > ob.high })
        }
    }
}

/// Points for one side plus the tag each factor would show, `(points, tag)`.
struct SideScore {
    total: f64,
    tags: Vec<(f64, String)>,
    zone: Option<BoardZone>,
    /// What the zone factor scored (0 = no live zone within 2 ATR).
    zone_pts: f64,
    /// The 4h move in ATR, in this side's favour (negative = against it).
    move_atr: f64,
}

struct Readings<'a> {
    closed: &'a [Candle],
    s15: &'a MarketStructure,
    s1h: Option<&'a MarketStructure>,
    n1h: usize,
    levels: Option<&'a quant_core::levels::KeyLevels>,
    price: f64,
    atr: f64,
    oi: Option<f64>,
    liq: Option<LiqWindow>,
    qv24: f64,
}

fn score_side(r: &Readings, side: Side) -> SideScore {
    let mut tags: Vec<(f64, String)> = Vec::new();
    let word = side.word();
    let n = r.closed.len();

    // 1h trend: the latest 1h break, if recent, plus the EMA stack.
    if let Some(s1h) = r.s1h {
        let brk = latest_break(s1h)
            .filter(|b| b.direction == word && r.n1h - 1 - b.break_idx <= TREND_MAX_AGE)
            .map_or(0.0, |_| TREND_BREAK_PTS);
        let ema = if s1h.ema_stack == word { TREND_EMA_PTS } else { 0.0 };
        if brk + ema > 0.0 {
            let label = if brk > 0.0 { format!("1h {}trend", side.trend()) } else { format!("1h EMAs {word}") };
            tags.push((brk + ema, label));
        }
    }

    // 15m structure: a fresh break in this side's direction.
    if let Some(b) = latest_break(r.s15).filter(|b| b.direction == word) {
        let age = n - 1 - b.break_idx;
        let base = if b.kind == "BOS" { BOS_PTS } else { CHOCH_PTS };
        let pts = if age <= BREAK_FULL_AGE {
            base
        } else if age < BREAK_ZERO_AGE {
            base * (BREAK_ZERO_AGE - age) as f64 / (BREAK_ZERO_AGE - BREAK_FULL_AGE) as f64
        } else {
            0.0
        };
        if pts > 0.0 {
            tags.push((pts, format!("15m {} {}", b.kind, side.trend())));
        }
    }

    // Zone: the nearest live order block or open FVG on this side of price.
    let mut zone: Option<(f64, &'static str, f64, f64)> = None; // (distance, kind, low, high)
    if let Some(lv) = r.levels {
        let obs = lv
            .order_blocks
            .iter()
            .filter(|o| o.direction == word && order_block_live(r.closed, o))
            .map(|o| ("order_block", o.low, o.high));
        let fvgs = lv.fvgs.iter().filter(|g| g.direction == word).map(|g| ("fvg", g.low, g.high));
        for (kind, low, high) in obs.chain(fvgs) {
            // Bullish zones must sit at or below price, bearish ones at or above.
            let dist = match side {
                Side::Bull if low <= r.price => (r.price - high).max(0.0),
                Side::Bear if high >= r.price => (low - r.price).max(0.0),
                _ => continue,
            };
            // Order blocks are listed first, so an equally near FVG never displaces one.
            if zone.is_none_or(|z| dist < z.0) {
                zone = Some((dist, kind, low, high));
            }
        }
    }
    let mut zone_pts = 0.0;
    let board_zone = zone.map(|(dist, kind, low, high)| {
        let pts = if dist <= 0.0 {
            ZONE_INSIDE_PTS
        } else if dist <= r.atr {
            ZONE_1ATR_PTS
        } else if dist <= 2.0 * r.atr {
            ZONE_2ATR_PTS
        } else {
            0.0
        };
        zone_pts = pts;
        if pts > 0.0 {
            let at = if dist <= 0.0 { "At" } else { "Near" };
            let name = if kind == "order_block" { "OB" } else { "FVG" };
            tags.push((pts, format!("{at} {word} {name}")));
        }
        BoardZone { kind, low, high, distance_pct: round2(-side.sign() * dist / r.price * 100.0) }
    });

    // Momentum: the 4h move in ATRs, in this side's favour.
    let ref_close = r.closed[n - LOOKBACK_4H].close;
    let move_atr = side.sign() * (r.price - ref_close) / r.atr;
    let mom = move_atr.clamp(0.0, MOMENTUM_MAX_ATR) * MOMENTUM_PTS_PER_ATR;
    if mom > 0.0 {
        let verb = if side == Side::Bull { "move" } else { "drop" };
        tags.push((mom, format!("Strong {verb} {:.1} ATR", move_atr)));
    }

    // OI against price over 4h: new positions behind the move, or the other side covering.
    if let Some(oi) = r.oi {
        let pc = side.sign() * (r.price - ref_close) / ref_close * 100.0;
        if pc >= OI_PRICE_EDGE_PCT && oi >= OI_EDGE_PCT {
            tags.push((OI_NEW_POSITIONS_PTS, format!("OI +{oi:.0}% 4h")));
        } else if pc > 0.0 && oi <= -OI_EDGE_PCT {
            let who = if side == Side::Bull { "Shorts covering" } else { "Longs closing" };
            tags.push((OI_COVERING_PTS, who.to_string()));
        }
    }

    // Liquidity: the opposite side's equal highs/lows swept, and forced liquidations.
    if let Some(lv) = r.levels {
        let pool = if side == Side::Bull { "equal_lows" } else { "equal_highs" };
        if lv.pools.iter().any(|p| p.side == pool && p.status == "swept" && n - 1 - p.last_touch_idx.min(n - 1) <= SWEEP_MAX_AGE) {
            let label = if side == Side::Bull { "Equal lows swept" } else { "Equal highs swept" };
            tags.push((SWEEP_PTS, label.to_string()));
        }
    }
    if let Some(liq) = r.liq {
        let net = match side {
            Side::Bull => liq.short_usd - liq.long_usd,
            Side::Bear => liq.long_usd - liq.short_usd,
        };
        if r.qv24 > 0.0 && net >= LIQ_NET_OF_VOLUME * r.qv24 {
            let who = if side == Side::Bull { "Shorts squeezed" } else { "Longs flushed" };
            tags.push((LIQ_PTS, format!("{who} {} 1h", usd_short(net))));
        }
    }

    let total = tags.iter().map(|(p, _)| p).sum();
    SideScore { total, tags, zone: board_zone, zone_pts, move_atr }
}

/// `$1.2M`, `$350K`, `$900`.
fn usd_short(v: f64) -> String {
    if v >= 1e6 {
        format!("${:.1}M", v / 1e6)
    } else if v >= 1e3 {
        format!("${:.0}K", v / 1e3)
    } else {
        format!("${v:.0}")
    }
}

/// Score one symbol. `None` only when it cannot be scored at all (too little history,
/// stale bars, no ATR); a choppy symbol is `Some` with `side: None` so it still counts
/// as scored.
pub fn score_symbol(inp: &SymbolInputs) -> Option<Scored> {
    let last = inp.bars.last()?;
    if inp.now_ms - last.t > STALE_MS {
        return None;
    }
    // The last REST kline is still forming while its 15 minutes are not over: its close
    // is the live price, but structure and zones read closed bars only.
    let forming = last.t + BAR_MS > inp.now_ms;
    let closed_bars = if forming { &inp.bars[..inp.bars.len() - 1] } else { inp.bars };
    if closed_bars.len() < MIN_BARS {
        return None;
    }
    let price = last.c.close;
    if !(price.is_finite() && price > 0.0) {
        return None;
    }

    let window = &closed_bars[closed_bars.len().saturating_sub(WINDOW_15M)..];
    let closed: Vec<Candle> = window.iter().map(|b| b.c.clone()).collect();
    let cfg = StructureConfig::default();
    let s15 = analyze_structure(&closed, &cfg)?;
    let atr = s15.atr_14.filter(|a| *a > 0.0)?;
    let levels = analyze_levels(&closed);

    // 1h from ALL closed 15m bars; the last hour counts only once all four are closed.
    let mut hourly = to_hourly(closed_bars);
    let closed_end = closed_bars.last().map_or(0, |b| b.t + BAR_MS);
    if hourly.last().is_some_and(|h| h.t + HOUR_MS > closed_end) {
        hourly.pop();
    }
    let hourly_c: Vec<Candle> = hourly.iter().map(|b| b.c.clone()).collect();
    let s1h = analyze_structure(&hourly_c, &cfg);

    let n = closed.len();
    let day = &closed_bars[closed_bars.len() - DAY_BARS..];
    let qv24: f64 = day.iter().map(|b| b.qv).sum();
    let change_24h_pct = round2((price - day[0].c.close) / day[0].c.close * 100.0);

    let recent_break = latest_break(&s15).is_some_and(|b| n - 1 - b.break_idx <= BREAK_ZERO_AGE);
    let choppy = efficiency_ratio(&closed, CHOP_BARS) < CHOP_MAX_ER && !recent_break;

    let readings = Readings {
        closed: &closed,
        s15: &s15,
        s1h: s1h.as_ref(),
        n1h: hourly_c.len(),
        levels: levels.as_ref(),
        price,
        atr,
        oi: inp.oi_4h_change_pct.filter(|v| v.is_finite()),
        liq: inp.liq_1h,
        qv24,
    };
    let bull = score_side(&readings, Side::Bull);
    let bear = score_side(&readings, Side::Bear);
    let (bull_pts, bear_pts) = if choppy { (0, 0) } else { (bull.total.round() as u32, bear.total.round() as u32) };

    let (side, chosen, score) = if bull_pts > bear_pts {
        (Some(Side::Bull), bull, bull_pts)
    } else if bear_pts > bull_pts {
        (Some(Side::Bear), bear, bear_pts)
    } else {
        (None, bull, 0)
    };
    let extended = side.is_some() && chosen.zone_pts == 0.0 && chosen.move_atr >= EXTENDED_MIN_MOVE_ATR;
    let mut tags = chosen.tags;
    tags.sort_by(|a, b| b.0.total_cmp(&a.0));
    Some(Scored {
        side,
        row: Row {
            symbol: format!("{}.P", inp.symbol),
            score: score.min(100),
            price,
            change_24h_pct,
            zone: chosen.zone,
            tags: tags.into_iter().take(3).map(|(_, t)| t).collect(),
        },
        quote_volume_24h: qv24,
        extended,
    })
}

/// Rank the scored symbols into the board: one side per symbol, at least `min_score`,
/// strongest first (ties to the higher 24h quote volume), `top_n` per side on the board
/// proper and `EXTENDED_TOP_N` per side under `extended`. The tilt counts every
/// qualifying symbol, extended or not, listed or not.
pub fn build_board(scored: Vec<Scored>, as_of: i64, universe: usize, liquidations_ready: bool, cfg: &Cfg) -> Board {
    let count = scored.len();
    let mut bull: Vec<Scored> = Vec::new();
    let mut bear: Vec<Scored> = Vec::new();
    for s in scored {
        if s.row.score < cfg.min_score {
            continue;
        }
        match s.side {
            Some(Side::Bull) => bull.push(s),
            Some(Side::Bear) => bear.push(s),
            None => {}
        }
    }
    let tilt = Tilt { bullish: bull.len(), bearish: bear.len() };
    let rank = |v: &[Scored], extended: bool, n: usize| -> Vec<Row> {
        let mut v: Vec<&Scored> = v.iter().filter(|s| s.extended == extended).collect();
        v.sort_by(|a, b| b.row.score.cmp(&a.row.score).then(b.quote_volume_24h.total_cmp(&a.quote_volume_24h)));
        v.into_iter().take(n).map(|s| s.row.clone()).collect()
    };
    Board {
        status: "ready",
        scored: count,
        tilt,
        liquidations_ready,
        bullish: rank(&bull, false, cfg.top_n),
        bearish: rank(&bear, false, cfg.top_n),
        extended: Extended {
            bullish: rank(&bull, true, EXTENDED_TOP_N),
            bearish: rank(&bear, true, EXTENDED_TOP_N),
        },
        ..Board::warming_up(as_of, universe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../tests/fixtures/klines_15m.json");
    const T0: i64 = 1_790_000_000_000 - 1_790_000_000_000 % HOUR_MS;

    fn bar(i: usize, open: f64, high: f64, low: f64, close: f64) -> Bar {
        Bar {
            t: T0 + i as i64 * BAR_MS,
            c: Candle { open, high, low, close, volume: 100.0, taker_buy_volume: Some(50.0) },
            qv: 100.0 * close,
        }
    }

    /// Bars walking linearly through `pivots`, `per_leg` bars per leg, with a wick that
    /// grows a little each bar so two bars never tie at a pivot (quant-core's own
    /// fixture trick).
    fn series(pivots: &[f64], per_leg: usize) -> Vec<Bar> {
        let mut pts = vec![pivots[0]];
        for w in pivots.windows(2) {
            for s in 1..=per_leg {
                pts.push(w[0] + (w[1] - w[0]) * s as f64 / per_leg as f64);
            }
        }
        pts.windows(2)
            .enumerate()
            .map(|(k, p)| {
                let eps = 0.05 + 0.0001 * k as f64;
                bar(k, p[0], p[0].max(p[1]) + eps, p[0].min(p[1]) - eps, p[1])
            })
            .collect()
    }

    fn now_after(bars: &[Bar]) -> i64 {
        // Mid-way through the last bar: it is the forming one.
        bars.last().unwrap().t + BAR_MS / 2
    }

    fn inputs<'a>(sym: &'a str, bars: &'a [Bar]) -> SymbolInputs<'a> {
        SymbolInputs { symbol: sym, bars, now_ms: now_after(bars), oi_4h_change_pct: None, liq_1h: None }
    }

    /// An uptrend in steps (higher highs and higher lows, 4h per leg so the 1h series
    /// has structure too), cut half-way through its last pullback.
    fn uptrend_pullback() -> Vec<Bar> {
        let mut piv = vec![100.0];
        let mut p = 100.0;
        for _ in 0..30 {
            piv.push(p + 6.0);
            piv.push(p + 3.0);
            p += 3.0;
        }
        let mut bars = series(&piv, 16);
        bars.truncate(bars.len() - 8);
        bars
    }

    #[test]
    fn klines_fixture_parses_and_the_detectors_read_it() {
        let bars = parse_klines(FIXTURE).unwrap();
        assert_eq!(bars.len(), 1000);
        assert!(bars.windows(2).all(|w| w[1].t - w[0].t == BAR_MS), "15m spacing");
        assert!(bars.iter().all(|b| b.qv > 0.0 && b.c.taker_buy_volume.is_some()));
        let c: Vec<Candle> = bars.iter().map(|b| b.c.clone()).collect();
        let lv = analyze_levels(&c[500..]).expect("levels");
        let ms = analyze_structure(&c[500..], &StructureConfig::default()).expect("structure");
        println!(
            "fixture: {} order blocks, {} open FVGs, {} pools, trend {}, atr {:?}",
            lv.order_blocks.len(),
            lv.fvgs.len(),
            lv.pools.len(),
            ms.trend,
            ms.atr_14
        );
        assert!(ms.atr_14.is_some());
    }

    #[test]
    fn parse_klines_drops_bad_rows_and_rejects_non_arrays() {
        let body = r#"[[1,"1","2","0.5","1.5","10",2,"15",3,"4","6","0"],[2,"x","2","0.5","1.5","10",2,"15",3,"4","6","0"],[3,"1"]]"#;
        let bars = parse_klines(body).unwrap();
        assert_eq!(bars.len(), 1);
        assert_eq!((bars[0].t, bars[0].c.close, bars[0].qv, bars[0].c.taker_buy_volume), (1, 1.5, 15.0, Some(4.0)));
        assert!(parse_klines(r#"{"code":-1121}"#).is_err());
    }

    #[test]
    fn hourly_buckets_align_and_drop_only_a_partial_first_hour() {
        // Starts at :30 (two bars of a partial hour), then two full hours, then one bar.
        let mut bars: Vec<Bar> = (0..11).map(|i| bar(i, 10.0 + i as f64, 11.0 + i as f64, 9.0 + i as f64, 10.5 + i as f64)).collect();
        for b in &mut bars {
            b.t += 2 * BAR_MS;
        }
        let h = to_hourly(&bars);
        assert_eq!(h.len(), 3, "partial first hour dropped, forming last hour kept");
        assert_eq!(h[0].t, T0 + HOUR_MS);
        assert_eq!(h[1].t, T0 + 2 * HOUR_MS);
        assert_eq!(h[2].t, T0 + 3 * HOUR_MS);
        // First full hour = bars 2..=5.
        assert_eq!((h[0].c.open, h[0].c.high, h[0].c.low, h[0].c.close), (12.0, 16.0, 11.0, 15.5));
        assert_eq!(h[0].c.volume, 400.0);
        assert_eq!(h[0].c.taker_buy_volume, Some(200.0));
        assert!((h[0].qv - 100.0 * (12.5 + 13.5 + 14.5 + 15.5)).abs() < 1e-9);
        // The forming hour carries its one bar.
        assert_eq!((h[2].c.open, h[2].c.close, h[2].c.volume), (20.0, 20.5, 100.0));

        // A first hour that is complete is kept.
        let full: Vec<Bar> = (0..4).map(|i| bar(i, 1.0, 2.0, 0.5, 1.5)).collect();
        assert_eq!(to_hourly(&full).len(), 1);
        assert!(to_hourly(&[]).is_empty());
    }

    #[test]
    fn polls_merge_by_open_time_and_a_gap_replaces_the_series() {
        let mut s: Vec<Bar> = (0..5).map(|i| bar(i, 1.0, 2.0, 0.5, 1.0)).collect();
        // The forming bar 4 updated, bar 5 opened.
        merge_bars(&mut s, vec![bar(3, 1.0, 2.0, 0.5, 1.0), bar(4, 1.0, 3.0, 0.5, 2.5), bar(5, 2.5, 2.6, 2.4, 2.5)]);
        assert_eq!(s.len(), 6);
        assert_eq!(s[4].c.close, 2.5);
        assert!(s.windows(2).all(|w| w[1].t > w[0].t));
        // A poll that starts after a hole: the series is replaced, not left gapped.
        merge_bars(&mut s, vec![bar(9, 1.0, 1.0, 1.0, 1.0), bar(10, 1.0, 1.0, 1.0, 1.0)]);
        assert_eq!(s.iter().map(|b| b.t).collect::<Vec<_>>(), vec![T0 + 9 * BAR_MS, T0 + 10 * BAR_MS]);
        merge_bars(&mut s, vec![]);
        assert_eq!(s.len(), 2);
        // Capped at KEEP_BARS, oldest dropped.
        let mut long: Vec<Bar> = (0..KEEP_BARS).map(|i| bar(i, 1.0, 1.0, 1.0, 1.0)).collect();
        merge_bars(&mut long, vec![bar(KEEP_BARS, 1.0, 1.0, 1.0, 1.0)]);
        assert_eq!((long.len(), long[0].t), (KEEP_BARS, T0 + BAR_MS));
    }

    #[test]
    fn an_uptrend_retesting_its_order_block_ranks_bullish() {
        let mut bars = uptrend_pullback();
        let s = score_symbol(&inputs("TESTUSDT", &bars)).expect("scored");
        assert_eq!(s.side, Some(Side::Bull), "{:?}", s.row);

        // Put the live price inside the newest bullish order block (its first retest).
        let closed: Vec<Candle> = bars[..bars.len() - 1].iter().map(|b| b.c.clone()).collect();
        let lv = analyze_levels(&closed[closed.len().saturating_sub(WINDOW_15M)..]).unwrap();
        let ob = lv
            .order_blocks
            .iter()
            .filter(|o| o.direction == "bullish")
            .max_by_key(|o| o.break_idx)
            .expect("a bullish order block")
            .clone();
        let last = bars.last_mut().unwrap();
        let mid = (ob.low + ob.high) / 2.0;
        last.c = Candle { open: mid, high: mid, low: mid, close: mid, volume: 1.0, taker_buy_volume: None };
        let s = score_symbol(&SymbolInputs { oi_4h_change_pct: Some(3.0), ..inputs("TESTUSDT", &bars) }).unwrap();
        println!("{:?}", s.row);
        assert_eq!(s.side, Some(Side::Bull));
        assert!(s.row.score >= 60, "{:?}", s.row);
        assert!(s.row.tags.iter().any(|t| t == "At bullish OB"), "{:?}", s.row.tags);
        let z = s.row.zone.as_ref().unwrap();
        assert_eq!((z.kind, z.distance_pct), ("order_block", 0.0));
        assert_eq!(s.row.symbol, "TESTUSDT.P");
    }

    #[test]
    fn a_mirrored_downtrend_ranks_bearish() {
        let up = uptrend_pullback();
        let top = 400.0;
        let bars: Vec<Bar> = up
            .iter()
            .map(|b| Bar {
                t: b.t,
                c: Candle {
                    open: top - b.c.open,
                    high: top - b.c.low,
                    low: top - b.c.high,
                    close: top - b.c.close,
                    volume: b.c.volume,
                    taker_buy_volume: b.c.taker_buy_volume,
                },
                qv: b.qv,
            })
            .collect();
        let s = score_symbol(&inputs("DOWNUSDT", &bars)).unwrap();
        assert_eq!(s.side, Some(Side::Bear), "{:?}", s.row);
        assert!(s.row.score > 0);
        assert!(s.row.tags.iter().any(|t| t.contains("down") || t.contains("bearish") || t.contains("drop")), "{:?}", s.row.tags);
    }

    #[test]
    fn a_flat_choppy_series_scores_zero_and_is_left_off_the_board() {
        // Alternating ±0.3 around 100: no trend, no structure leg of 1 ATR.
        let bars: Vec<Bar> = (0..400)
            .map(|i| {
                let c = if i % 2 == 0 { 100.3 } else { 99.7 };
                bar(i, 100.0, 100.4 + 0.0001 * i as f64, 99.6, c)
            })
            .collect();
        let closed: Vec<Candle> = bars[..bars.len() - 1].iter().map(|b| b.c.clone()).collect();
        assert!(efficiency_ratio(&closed, CHOP_BARS) < CHOP_MAX_ER, "the gate's condition holds");
        let s = score_symbol(&inputs("FLATUSDT", &bars)).unwrap();
        assert_eq!((s.side, s.row.score), (None, 0));
        let board = build_board(vec![s], 1, 1, false, &Cfg { top_n: 20, min_score: 40 });
        assert!(board.bullish.is_empty() && board.bearish.is_empty());
        assert_eq!(board.scored, 1);
    }

    #[test]
    fn an_order_block_retested_long_ago_is_not_live() {
        // Block [9, 10] broken at bar 2, first touched at bar 5 of 20 closed bars.
        let mut c: Vec<Candle> = (0..20).map(|_| Candle { open: 12.0, high: 12.5, low: 11.5, close: 12.0, volume: 1.0, taker_buy_volume: None }).collect();
        c[5].low = 9.8;
        let ob = OrderBlock { direction: "bullish", low: 9.0, high: 10.0, idx: 1, break_idx: 2, mitigated: true };
        assert!(!order_block_live(&c, &ob));
        // The same touch on the second-to-last bar is a fresh retest...
        c[5].low = 11.5;
        c[18].low = 9.8;
        assert!(order_block_live(&c, &ob));
        // ...unless a close since then went through the far edge.
        c[19].close = 8.9;
        assert!(!order_block_live(&c, &ob));
        // Never touched = live.
        c[18].low = 11.5;
        c[19].close = 12.0;
        assert!(order_block_live(&c, &ob));
    }

    #[test]
    fn missing_oi_and_liquidations_score_nothing_and_do_not_panic() {
        let bars = uptrend_pullback();
        let none = score_symbol(&inputs("AUSDT", &bars)).unwrap();
        let with = score_symbol(&SymbolInputs {
            oi_4h_change_pct: Some(f64::NAN),
            liq_1h: Some(LiqWindow::default()),
            ..inputs("AUSDT", &bars)
        })
        .unwrap();
        assert_eq!(none.row.score, with.row.score);
        assert!(!with.row.tags.iter().any(|t| t.starts_with("OI") || t.contains("squeezed")));
    }

    #[test]
    fn liquidations_count_toward_the_side_they_push() {
        let bars = uptrend_pullback();
        let base = score_symbol(&inputs("LUSDT", &bars)).unwrap();
        let qv24: f64 = base.quote_volume_24h;
        let squeeze = LiqWindow { long_usd: 0.0, short_usd: qv24 * 0.001 };
        let s = score_symbol(&SymbolInputs { liq_1h: Some(squeeze), ..inputs("LUSDT", &bars) }).unwrap();
        assert_eq!(s.row.score, base.row.score + LIQ_PTS as u32, "{:?}", s.row);
        // Longs flushed is bearish evidence: the bullish score does not move.
        let flush = LiqWindow { long_usd: qv24 * 0.001, short_usd: 0.0 };
        let s = score_symbol(&SymbolInputs { liq_1h: Some(flush), ..inputs("LUSDT", &bars) }).unwrap();
        assert_eq!(s.row.score, base.row.score);
    }

    #[test]
    fn stale_or_short_series_are_skipped() {
        let bars = uptrend_pullback();
        let mut stale = inputs("SUSDT", &bars);
        stale.now_ms += STALE_MS + BAR_MS;
        assert!(score_symbol(&stale).is_none());
        assert!(score_symbol(&inputs("SUSDT", &bars[..200])).is_none());
        let empty = SymbolInputs { bars: &[], ..inputs("SUSDT", &bars) };
        assert!(score_symbol(&empty).is_none());
    }

    fn scored(sym: &str, side: Option<Side>, score: u32, qv: f64) -> Scored {
        Scored {
            side,
            row: Row { symbol: format!("{sym}.P"), score, price: 1.0, change_24h_pct: 0.0, zone: None, tags: vec![] },
            quote_volume_24h: qv,
            extended: false,
        }
    }

    #[test]
    fn a_runaway_rally_with_no_zone_near_is_extended() {
        // The uptrend, then 20 bars straight up whose ranges overlap (no fair value gap
        // of their own), so the nearest zone is the base the rally left behind.
        let mut bars = uptrend_pullback();
        let mut close = bars.last().unwrap().c.close;
        let start = bars.len();
        for k in 0..20 {
            let open = close;
            close = open + 1.0;
            bars.push(bar(start + k, open, close + 0.6, open - 0.6, close));
        }
        let s = score_symbol(&inputs("RUNUSDT", &bars)).unwrap();
        println!("{:?}", s.row);
        assert_eq!(s.side, Some(Side::Bull));
        assert!(s.extended, "{:?}", s.row);
        assert!(!s.row.tags.iter().any(|t| t.starts_with("At ") || t.starts_with("Near ")), "{:?}", s.row.tags);
        // The same trend sitting in its order block is the board proper.
        let at_zone = score_symbol(&inputs("TESTUSDT", &uptrend_pullback())).unwrap();
        assert!(!at_zone.extended, "{:?}", at_zone.row);
    }

    #[test]
    fn extended_rows_are_listed_apart_and_still_counted_in_the_tilt() {
        let mut far = scored("FAR", Some(Side::Bull), 90, 1.0);
        far.extended = true;
        let mut far_bear = scored("DUMP", Some(Side::Bear), 70, 1.0);
        far_bear.extended = true;
        let mut many: Vec<Scored> = (0..15)
            .map(|i| {
                let mut s = scored(&format!("E{i}"), Some(Side::Bull), 50 + i, 1.0);
                s.extended = true;
                s
            })
            .collect();
        many.extend([far, far_bear, scored("NEAR", Some(Side::Bull), 60, 1.0)]);
        let b = build_board(many, 1, 150, true, &Cfg { top_n: 20, min_score: 40 });
        assert_eq!(b.bullish.iter().map(|r| r.symbol.as_str()).collect::<Vec<_>>(), vec!["NEAR.P"]);
        assert!(b.bearish.is_empty());
        assert_eq!(b.extended.bullish.len(), EXTENDED_TOP_N);
        assert_eq!(b.extended.bullish[0].symbol, "FAR.P");
        assert_eq!(b.extended.bearish.iter().map(|r| r.symbol.as_str()).collect::<Vec<_>>(), vec!["DUMP.P"]);
        assert_eq!(b.tilt, Tilt { bullish: 17, bearish: 1 });
        let v = serde_json::to_value(&b).unwrap();
        assert_eq!(v["extended"]["bullish"][0]["symbol"], "FAR.P");
        let w = serde_json::to_value(Board::warming_up(1, 0)).unwrap();
        assert_eq!(w["extended"], serde_json::json!({ "bullish": [], "bearish": [] }));
    }

    #[test]
    fn the_board_ranks_caps_and_counts_the_tilt() {
        let mut v: Vec<Scored> = (0..25).map(|i| scored(&format!("B{i}"), Some(Side::Bull), 40 + i, 1.0)).collect();
        v.push(scored("TIEHI", Some(Side::Bear), 70, 9.0));
        v.push(scored("TIELO", Some(Side::Bear), 70, 1.0));
        v.push(scored("WEAK", Some(Side::Bear), 39, 1.0));
        v.push(scored("NONE", None, 0, 1.0));
        let b = build_board(v, 1_767_000_000, 150, true, &Cfg { top_n: 20, min_score: 40 });
        assert_eq!(b.bullish.len(), 20);
        assert_eq!(b.bullish[0].symbol, "B24.P");
        assert!(b.bullish.windows(2).all(|w| w[0].score >= w[1].score));
        assert_eq!(b.bearish.iter().map(|r| r.symbol.as_str()).collect::<Vec<_>>(), vec!["TIEHI.P", "TIELO.P"]);
        assert_eq!(b.tilt, Tilt { bullish: 25, bearish: 2 });
        assert_eq!((b.scored, b.universe, b.status), (29, 150, "ready"));
        let names: std::collections::HashSet<_> = b.bullish.iter().chain(&b.bearish).map(|r| &r.symbol).collect();
        assert_eq!(names.len(), b.bullish.len() + b.bearish.len(), "no symbol on both sides");
    }

    #[test]
    fn the_frame_uses_the_v1_field_names() {
        let mut s = scored("SOLUSDT", Some(Side::Bull), 82, 1.0);
        s.row.zone = Some(BoardZone { kind: "order_block", low: 178.2, high: 179.6, distance_pct: -1.6 });
        s.row.tags = vec!["1h uptrend".into()];
        let b = build_board(vec![s], 1_767_000_000, 150, true, &Cfg { top_n: 20, min_score: 40 });
        let v: serde_json::Value = serde_json::to_value(&b).unwrap();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        for k in ["type", "v", "status", "as_of", "timeframe", "trend_timeframe", "universe", "scored", "tilt", "liquidations_ready", "bullish", "bearish"] {
            assert!(keys.contains(&k), "missing {k}");
        }
        assert_eq!(v["type"], "board");
        assert_eq!(v["v"], 1);
        let row = &v["bullish"][0];
        for k in ["symbol", "score", "price", "change_24h_pct", "zone", "tags"] {
            assert!(row.get(k).is_some(), "row missing {k}");
        }
        assert_eq!(row["zone"]["kind"], "order_block");
        assert_eq!(row["zone"]["distance_pct"], -1.6);
        let w = serde_json::to_value(Board::warming_up(5, 0)).unwrap();
        assert_eq!((w["status"].as_str(), w["bullish"].as_array().map(|a| a.len())), (Some("warming_up"), Some(0)));
    }

    #[test]
    fn prints_a_board_built_from_the_fixture() {
        let real = parse_klines(FIXTURE).unwrap();
        let up = uptrend_pullback();
        let now = real.last().unwrap().t + BAR_MS / 2;
        let mut all = vec![score_symbol(&SymbolInputs { symbol: "BTCUSDT", bars: &real, now_ms: now, oi_4h_change_pct: None, liq_1h: None })];
        all.push(score_symbol(&inputs("UPUSDT", &up)));
        let b = build_board(all.into_iter().flatten().collect(), now / 1000, 2, false, &Cfg { top_n: 20, min_score: 0 });
        println!("{}", serde_json::to_string_pretty(&b).unwrap());
        assert_eq!(b.scored, 2);
    }
}
