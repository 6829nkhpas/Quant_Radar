// binance.rs — everything that talks to Binance: the REST client (universe, 15m
// klines, open-interest history) behind a request-weight guard, and the all-market
// liquidation WebSocket.
//
// Public market data needs no API key. The weight guard matters because this service
// shares the crypto-app box's outbound IP with the aggregator, whose own Binance client
// tracks no weight at all; Binance's limit (2400 weight / minute for USDⓈ-M) is per IP,
// so this side holds itself to `RADAR_MAX_WEIGHT_1M` (default 1200, half) and pauses
// every request when the `X-MBX-USED-WEIGHT-1M` header reaches it.
//
// Weights, from Binance's own binance-connector-python source (the docs site renders
// with JavaScript): klines 1 for limit < 100, 5 for 500-1000; exchangeInfo 1;
// ticker/24hr without a symbol 40; openInterestHist 0, but its own limit of 1000
// requests per 5 minutes per IP. Measured 2026-10-29: a limit=1000 kline call
// returned `x-mbx-used-weight-1m: 5`.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use log::{info, warn};
use serde::Deserialize;
use tokio::sync::{Mutex, Semaphore};
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::Message;

use crate::score::{Bar, HOUR_MS, LiqWindow, parse_klines};

/// Copied from crypto-app `aggregator/src/exchange_api.rs` `BREADTH_STABLE_BASES`.
/// ponytail: a name list misses the next stablecoin listing; the upgrade is a
/// stable-pair price band (|24h change| < 0.5% for a week) instead of names.
const STABLE_BASES: [&str; 11] = ["USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "PYUSD", "USD1", "EUR", "EURI", "AEUR"];

/// At most this many requests in flight, and at most `REQUESTS_PER_SEC` started.
/// 150 symbols × weight 5 at startup is then 750 weight spread over ~12 s.
const CONCURRENCY: usize = 5;
const REQUESTS_PER_SEC: u64 = 12;

/// OI: compare against the record nearest 4h before the newest one, within this.
const OI_TOLERANCE_MS: i64 = 15 * 60 * 1000;
/// The newest OI record must be at most this old, or OI is unavailable.
const OI_FRESH_MS: i64 = 15 * 60 * 1000;

/// Liquidations are kept this long; the score reads the last hour.
const LIQ_KEEP_MS: i64 = 4 * HOUR_MS;
/// No liquidation frame for this long is logged as a warning.
const LIQ_SILENCE: Duration = Duration::from_secs(10 * 60);

pub fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

// ── Pure parsers ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ExchangeInfo {
    symbols: Vec<SymbolInfo>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SymbolInfo {
    symbol: String,
    #[serde(default)]
    contract_type: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    base_asset: String,
    #[serde(default)]
    quote_asset: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Ticker {
    symbol: String,
    #[serde(default)]
    quote_volume: String,
}

/// The top `n` USDT perpetuals by 24h quote volume, bare symbols, highest first.
///
/// `contractType == "PERPETUAL"` on purpose: Binance also lists `TRADIFI_PERPETUAL`
/// contracts (gold, stocks: XAUUSDT, TSLAUSDT — 235 of them on 2026-10-29), which are
/// not crypto and trade on a different clock.
pub fn rank_universe(exchange_info: &str, tickers: &str, n: usize) -> Result<Vec<String>, String> {
    let info: ExchangeInfo = serde_json::from_str(exchange_info).map_err(|e| format!("exchangeInfo: {e}"))?;
    let tickers: Vec<Ticker> = serde_json::from_str(tickers).map_err(|e| format!("ticker/24hr: {e}"))?;
    let volume: HashMap<&str, f64> = tickers
        .iter()
        .filter_map(|t| t.quote_volume.parse::<f64>().ok().filter(|v| v.is_finite() && *v > 0.0).map(|v| (t.symbol.as_str(), v)))
        .collect();
    let mut ranked: Vec<(&str, f64)> = info
        .symbols
        .iter()
        .filter(|s| {
            s.contract_type == "PERPETUAL"
                && s.status == "TRADING"
                && s.quote_asset == "USDT"
                && !STABLE_BASES.contains(&s.base_asset.as_str())
        })
        .filter_map(|s| volume.get(s.symbol.as_str()).map(|v| (s.symbol.as_str(), *v)))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(b.0)));
    Ok(ranked.into_iter().take(n).map(|(s, _)| s.to_string()).collect())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct OiRecord {
    sum_open_interest: String,
    timestamp: i64,
}

/// Open interest change over the 4h ending at the newest record, in percent.
///
/// Binance omits 5m periods with no trades, so the record 48 places back is not
/// necessarily 4h back: the comparison record is the one whose timestamp is nearest
/// 4h before the newest, and OI is unavailable (`None`) when none is within ±15 min or
/// the newest record is more than 15 min old.
pub fn oi_change_pct(body: &str, now_ms: i64) -> Option<f64> {
    let records: Vec<OiRecord> = serde_json::from_str(body).ok()?;
    let points: Vec<(i64, f64)> = records
        .iter()
        .filter_map(|r| r.sum_open_interest.parse::<f64>().ok().filter(|v| v.is_finite() && *v > 0.0).map(|v| (r.timestamp, v)))
        .collect();
    let latest = *points.iter().max_by_key(|p| p.0)?;
    if now_ms - latest.0 > OI_FRESH_MS {
        return None;
    }
    let target = latest.0 - 4 * HOUR_MS;
    let past = *points.iter().min_by_key(|p| (p.0 - target).abs())?;
    if (past.0 - target).abs() > OI_TOLERANCE_MS || past.0 == latest.0 {
        return None;
    }
    Some((latest.1 - past.1) / past.1 * 100.0)
}

/// How long every request must wait after this response, if at all.
///
/// 429 = over the limit: wait `Retry-After` (default 60 s). 418 = the IP is banned for
/// ignoring 429s: wait `Retry-After` (default 5 min). Otherwise, once the used weight
/// reaches `max_weight`, wait for the next minute window plus a second of slack.
pub fn guard_wait(used: Option<u32>, status: u16, retry_after_s: Option<u64>, max_weight: u32, now_ms: i64) -> Option<Duration> {
    match status {
        429 => Some(Duration::from_secs(retry_after_s.unwrap_or(60))),
        418 => Some(Duration::from_secs(retry_after_s.unwrap_or(300))),
        _ => match used {
            Some(u) if u >= max_weight => Some(Duration::from_millis((60_000 - now_ms.rem_euclid(60_000) + 1_000) as u64)),
            _ => None,
        },
    }
}

// ── REST client ──────────────────────────────────────────────────────────────

pub struct Binance {
    http: reqwest::Client,
    fapi: String,
    max_weight: u32,
    permits: Semaphore,
    /// The earliest instant the next request may start: the rate throttle and every
    /// guard pause share this one gate.
    gate: Mutex<Instant>,
    pub used_weight: AtomicU32,
    pub peak_weight: AtomicU32,
}

impl Binance {
    pub fn new(fapi: String, max_weight: u32) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .user_agent("quant_radar/0.1")
            .build()
            .expect("reqwest client");
        Binance {
            http,
            fapi: fapi.trim_end_matches('/').to_string(),
            max_weight,
            permits: Semaphore::new(CONCURRENCY),
            gate: Mutex::new(Instant::now()),
            used_weight: AtomicU32::new(0),
            peak_weight: AtomicU32::new(0),
        }
    }

    async fn wait_turn(&self) {
        let at = {
            let mut gate = self.gate.lock().await;
            let at = (*gate).max(Instant::now());
            *gate = at + Duration::from_millis(1000 / REQUESTS_PER_SEC);
            at
        };
        tokio::time::sleep_until(at).await;
    }

    async fn pause(&self, wait: Duration) {
        let until = Instant::now() + wait;
        let mut gate = self.gate.lock().await;
        *gate = (*gate).max(until);
    }

    async fn get(&self, path: &str, query: &[(&str, String)]) -> Result<String, String> {
        let _permit = self.permits.acquire().await.map_err(|e| e.to_string())?;
        for attempt in 0..3 {
            self.wait_turn().await;
            let resp = self
                .http
                .get(format!("{}{path}", self.fapi))
                .query(query)
                .send()
                .await
                .map_err(|e| format!("{path}: {e}"))?;
            let status = resp.status().as_u16();
            let header = |name: &str| resp.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_string);
            let used = header("x-mbx-used-weight-1m").and_then(|v| v.parse::<u32>().ok());
            let retry_after = header("retry-after").and_then(|v| v.parse::<u64>().ok());
            if let Some(u) = used {
                self.used_weight.store(u, Ordering::Relaxed);
                self.peak_weight.fetch_max(u, Ordering::Relaxed);
            }
            if let Some(wait) = guard_wait(used, status, retry_after, self.max_weight, now_ms()) {
                warn!("[binance] {path}: HTTP {status}, used weight {used:?}; pausing all requests {}s", wait.as_secs());
                self.pause(wait).await;
                if status == 429 && attempt < 2 {
                    continue;
                }
            }
            let body = resp.text().await.map_err(|e| format!("{path}: {e}"))?;
            if !(200..300).contains(&status) {
                let snippet: String = body.chars().take(200).collect();
                return Err(format!("{path}: HTTP {status}: {snippet}"));
            }
            return Ok(body);
        }
        Err(format!("{path}: still rate limited after 3 attempts"))
    }

    pub async fn universe(&self, n: usize) -> Result<Vec<String>, String> {
        let info = self.get("/fapi/v1/exchangeInfo", &[]).await?;
        let tickers = self.get("/fapi/v1/ticker/24hr", &[]).await?;
        rank_universe(&info, &tickers, n)
    }

    pub async fn klines(&self, symbol: &str, limit: usize) -> Result<Vec<Bar>, String> {
        let q = [("symbol", symbol.to_string()), ("interval", "15m".to_string()), ("limit", limit.to_string())];
        parse_klines(&self.get("/fapi/v1/klines", &q).await?)
    }

    pub async fn oi_change_4h(&self, symbol: &str) -> Result<Option<f64>, String> {
        let q = [("symbol", symbol.to_string()), ("period", "5m".to_string()), ("limit", "49".to_string())];
        Ok(oi_change_pct(&self.get("/futures/data/openInterestHist", &q).await?, now_ms()))
    }
}

// ── Liquidations ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Liquidation {
    pub symbol: String,
    pub ts: i64,
    /// `true` = a long was force-sold (`S == "SELL"`), `false` = a short force-bought.
    pub is_long: bool,
    pub usd: f64,
}

/// One `!forceOrder@arr` frame, or `None` for anything else (an ack, another event, a
/// COIN-M order). Since Binance's CM-UM integration (2026-06-30) the stream also
/// carries COIN-M liquidations, tagged `st: 2`; those are dropped here.
pub fn parse_force_order(raw: &str) -> Option<Liquidation> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    if v["e"] != "forceOrder" {
        return None;
    }
    let o = &v["o"];
    if o["st"].as_i64() == Some(2) {
        return None;
    }
    let is_long = match o["S"].as_str()? {
        "SELL" => true,
        "BUY" => false,
        _ => return None,
    };
    let num = |k: &str| o[k].as_str().and_then(|s| s.parse::<f64>().ok()).filter(|x| x.is_finite());
    // Average fill × filled quantity; a not-yet-filled order falls back to its limit
    // price × order quantity.
    let filled = num("ap").zip(num("z")).map(|(a, z)| a * z).filter(|u| *u > 0.0);
    let usd = filled.or_else(|| num("p").zip(num("q")).map(|(p, q)| p * q)).filter(|u| *u > 0.0)?;
    Some(Liquidation {
        symbol: o["s"].as_str()?.to_string(),
        ts: o["T"].as_i64().or_else(|| v["E"].as_i64())?,
        is_long,
        usd,
    })
}

/// The last 4 hours of liquidations per symbol.
///
/// ponytail: Binance pushes at most ONE liquidation per symbol per second on this
/// stream (the latest), so every total here is a lower bound — a cascade of many
/// orders in one second counts once. The exact figure needs the per-symbol
/// `<symbol>@forceOrder` streams (same 1 s snapshot) or the account-only REST history;
/// neither is available here, and a lower bound still ranks one side against the other.
pub struct Liquidations {
    by_symbol: HashMap<String, VecDeque<(i64, bool, f64)>>,
    first_frame_ms: Option<i64>,
    /// How long after the first frame the 1h window is trusted (`RADAR_LIQ_WARMUP_MINS`).
    warmup_ms: i64,
    pub seen: u64,
}

impl Liquidations {
    pub fn new(warmup_ms: i64) -> Self {
        Liquidations { by_symbol: HashMap::new(), first_frame_ms: None, warmup_ms, seen: 0 }
    }

    pub fn record(&mut self, l: Liquidation, now: i64) {
        self.first_frame_ms.get_or_insert(now);
        self.seen += 1;
        let q = self.by_symbol.entry(l.symbol).or_default();
        q.push_back((l.ts, l.is_long, l.usd));
        while q.front().is_some_and(|e| e.0 < now - LIQ_KEEP_MS) {
            q.pop_front();
        }
    }

    /// Drop everything older than 4 hours, and symbols left with nothing.
    pub fn prune(&mut self, now: i64) {
        for q in self.by_symbol.values_mut() {
            while q.front().is_some_and(|e| e.0 < now - LIQ_KEEP_MS) {
                q.pop_front();
            }
        }
        self.by_symbol.retain(|_, q| !q.is_empty());
    }

    pub fn window(&self, symbol: &str, now: i64, span_ms: i64) -> LiqWindow {
        let mut w = LiqWindow::default();
        for &(ts, is_long, usd) in self.by_symbol.get(symbol).into_iter().flatten() {
            if ts >= now - span_ms && ts <= now {
                if is_long { w.long_usd += usd } else { w.short_usd += usd }
            }
        }
        w
    }

    /// `warmup_ms` after the first frame (an hour by default): before a full hour the
    /// 1h window under-counts, so a shorter warm-up is for local checks only.
    pub fn ready(&self, now: i64) -> bool {
        self.first_frame_ms.is_some_and(|f| now - f >= self.warmup_ms)
    }
}

/// Hold the all-market liquidation stream open forever, handing every text frame to
/// `on_frame`.
///
/// It lives on the perp `/market` path — measured 2026-10-29 from this machine, frames
/// arrive there with `st: 1` (crypto-app's ingestion note had it unmeasured). Binance
/// pings and closes a connection that does not pong, and closes every connection at
/// 24 h; both end the inner loop and reconnect with 2-30 s backoff, the same shape as
/// crypto-app `ingestion/src/main.rs`.
pub async fn liquidation_feed<F: FnMut(&str) + Send>(url: String, mut on_frame: F) {
    let mut backoff = Duration::from_secs(2);
    loop {
        match tokio_tungstenite::connect_async(url.as_str()).await {
            Ok((ws, _)) => {
                info!("[liq] connected to {url}");
                backoff = Duration::from_secs(2);
                let (mut tx, mut rx) = ws.split();
                loop {
                    match tokio::time::timeout(LIQ_SILENCE, rx.next()).await {
                        Err(_) => warn!(
                            "[liq] no liquidation frame for {} minutes — check that `!forceOrder@arr` still arrives on the /market path",
                            LIQ_SILENCE.as_secs() / 60
                        ),
                        Ok(Some(Ok(Message::Text(t)))) => on_frame(t.as_str()),
                        Ok(Some(Ok(Message::Ping(p)))) => {
                            if let Err(e) = tx.send(Message::Pong(p)).await {
                                warn!("[liq] pong failed: {e}");
                                break;
                            }
                        }
                        Ok(Some(Ok(Message::Close(f)))) => {
                            info!("[liq] server closed the connection: {f:?}");
                            break;
                        }
                        Ok(Some(Ok(_))) => {}
                        Ok(Some(Err(e))) => {
                            warn!("[liq] websocket error: {e}");
                            break;
                        }
                        Ok(None) => break,
                    }
                }
            }
            Err(e) => warn!("[liq] connect to {url} failed: {e}"),
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXCHANGE_INFO: &str = include_str!("../tests/fixtures/exchange_info.json");
    const TICKERS: &str = include_str!("../tests/fixtures/ticker_24hr.json");
    const OI: &str = include_str!("../tests/fixtures/oi_hist.json");
    const FORCE_ORDERS: &str = include_str!("../tests/fixtures/force_order.json");

    #[test]
    fn the_universe_is_usdt_perps_by_volume_without_stablecoins() {
        // The fixture also holds USDCUSDT (stable base), XAUUSDT (TRADIFI_PERPETUAL),
        // OMGUSDT (SETTLING, no ticker), BTCUSDC (USDC quote), BTCUSDT_261225 (quarterly).
        let all = rank_universe(EXCHANGE_INFO, TICKERS, 10).unwrap();
        assert_eq!(all, vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]);
        assert_eq!(rank_universe(EXCHANGE_INFO, TICKERS, 2).unwrap(), vec!["BTCUSDT", "ETHUSDT"]);
        assert!(rank_universe("{}", TICKERS, 2).is_err());
    }

    fn oi_points() -> Vec<(i64, f64)> {
        let r: Vec<OiRecord> = serde_json::from_str(OI).unwrap();
        r.iter().map(|r| (r.timestamp, r.sum_open_interest.parse().unwrap())).collect()
    }

    fn oi_body(points: &[(i64, f64)]) -> String {
        let rows: Vec<String> = points
            .iter()
            .map(|(t, v)| format!(r#"{{"symbol":"BTCUSDT","sumOpenInterest":"{v}","sumOpenInterestValue":"1","timestamp":{t}}}"#))
            .collect();
        format!("[{}]", rows.join(","))
    }

    #[test]
    fn oi_change_compares_against_the_record_nearest_four_hours_back() {
        let p = oi_points();
        assert_eq!(p.len(), 49);
        let (first, last) = (p[0], p[48]);
        assert_eq!(last.0 - first.0, 4 * HOUR_MS, "the fixture spans exactly 4h");
        let now = last.0 + 60_000;
        let got = oi_change_pct(OI, now).unwrap();
        assert!((got - (last.1 - first.1) / first.1 * 100.0).abs() < 1e-9);

        // A gap at the 4h mark: the nearest record 15 min off still counts, 20 min does not.
        let gap15 = oi_body(&p[3..]);
        assert!(oi_change_pct(&gap15, now).is_some());
        let gap20 = oi_body(&p[4..]);
        assert_eq!(oi_change_pct(&gap20, now), None);
        // Stale newest record, and an empty / malformed body.
        assert_eq!(oi_change_pct(OI, last.0 + OI_FRESH_MS + 1), None);
        assert_eq!(oi_change_pct("[]", now), None);
        assert_eq!(oi_change_pct(r#"{"code":-1121}"#, now), None);
    }

    #[test]
    fn the_weight_guard_pauses_on_limits_and_bans() {
        let at = 1_793_289_330_000; // 30 s into a minute
        assert_eq!(guard_wait(Some(100), 200, None, 1200, at), None);
        assert_eq!(guard_wait(None, 200, None, 1200, at), None);
        assert_eq!(guard_wait(Some(1200), 200, None, 1200, at), Some(Duration::from_secs(31)));
        assert_eq!(guard_wait(Some(10), 429, Some(7), 1200, at), Some(Duration::from_secs(7)));
        assert_eq!(guard_wait(Some(10), 429, None, 1200, at), Some(Duration::from_secs(60)));
        assert_eq!(guard_wait(None, 418, None, 1200, at), Some(Duration::from_secs(300)));
        assert_eq!(guard_wait(None, 418, Some(1000), 1200, at), Some(Duration::from_secs(1000)));
        assert_eq!(guard_wait(Some(5), 500, None, 1200, at), None);
    }

    fn frames() -> Vec<String> {
        let v: Vec<serde_json::Value> = serde_json::from_str(FORCE_ORDERS).unwrap();
        v.iter().map(|f| f.to_string()).collect()
    }

    #[test]
    fn force_orders_parse_side_and_usd_and_drop_coin_m() {
        let f = frames();
        let a = parse_force_order(&f[0]).unwrap();
        assert_eq!((a.symbol.as_str(), a.is_long, a.ts), ("HYPEUSDT", false, 1793304740130));
        assert!((a.usd - 93.816 * 30.0).abs() < 1e-6);
        let b = parse_force_order(&f[1]).unwrap();
        assert!(b.is_long);
        assert!((b.usd - 0.0135749 * 184999.0).abs() < 1e-6);
        // Nothing filled yet: limit price × order quantity.
        let c = parse_force_order(&f[2]).unwrap();
        assert!((c.usd - 0.05874 * 2630.0).abs() < 1e-6);
        assert_eq!(parse_force_order(&f[3]), None, "COIN-M (st: 2) is dropped");
        assert_eq!(parse_force_order(&f[4]), None, "not a forceOrder");
        assert_eq!(parse_force_order("not json"), None);
    }

    #[test]
    fn liquidation_windows_sum_by_side_and_evict_after_four_hours() {
        let now = 10 * HOUR_MS;
        let mut l = Liquidations::new(HOUR_MS);
        assert!(!l.ready(now));
        let liq = |ts: i64, is_long: bool, usd: f64| Liquidation { symbol: "XUSDT".into(), ts, is_long, usd };
        l.record(liq(now - 5 * HOUR_MS, true, 999.0), now - 5 * HOUR_MS);
        l.record(liq(now - 2 * HOUR_MS, true, 50.0), now);
        l.record(liq(now - 30 * 60_000, true, 100.0), now);
        l.record(liq(now - 10 * 60_000, false, 300.0), now);
        assert_eq!(l.seen, 4);
        assert_eq!(l.window("XUSDT", now, HOUR_MS), LiqWindow { long_usd: 100.0, short_usd: 300.0 });
        assert_eq!(l.window("XUSDT", now, 4 * HOUR_MS), LiqWindow { long_usd: 150.0, short_usd: 300.0 });
        assert_eq!(l.window("NOPE", now, HOUR_MS), LiqWindow::default());
        // The first frame arrived 5h before `now`, so the window is trustworthy.
        assert!(l.ready(now));
        // 3h45 later only the newest (10 min before `now`) is still inside 4 hours.
        let later = now + 4 * HOUR_MS - 15 * 60_000;
        l.prune(later);
        assert_eq!(l.window("XUSDT", later, 4 * HOUR_MS), LiqWindow { long_usd: 0.0, short_usd: 300.0 });
        l.prune(now + 5 * HOUR_MS);
        assert!(l.by_symbol.is_empty());
    }

    #[test]
    fn the_warm_up_is_measured_from_the_first_frame() {
        let mut l = Liquidations::new(5 * 60_000);
        l.record(Liquidation { symbol: "XUSDT".into(), ts: 1_000, is_long: true, usd: 1.0 }, 1_000);
        assert!(!l.ready(1_000 + 5 * 60_000 - 1));
        assert!(l.ready(1_000 + 5 * 60_000));
    }
}
