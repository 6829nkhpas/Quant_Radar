// main.rs — quant_radar: scan the top USDⓈ-M perps every minute, rank the strongest
// bullish and bearish setups, and push the one board to every connected client.
//
// Shape:
//   scanner task   universe (hourly) → 15m klines (backfill once, then the last few
//                  bars per minute) → OI history (every 5 min) → score → publish
//   liq task       one WebSocket to Binance's all-market liquidation stream
//   axum server    `/health`, and a WebSocket on every other path (the gateway strips
//                  `/ws/radar` with `handle_path`, so the path that arrives varies)
//
// Every client gets the stored snapshot first, then each new board. The work is the
// same whether one user or ten thousand are connected: one scan, one serialization,
// one ~10 KB frame fanned out by a broadcast channel.

mod binance;
mod score;

use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::extract::ws::{Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use futures_util::StreamExt;
use log::{info, warn};
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::RecvError;
use tokio::time::Instant;

use binance::{Binance, Liquidations, now_ms, parse_force_order};
use score::{Bar, Board, Cfg, HOUR_MS, LiqWindow, Scored, SymbolInputs, build_board, merge_bars, score_symbol};

/// A board older than this (once warmed up) makes `/health` answer 503.
const STALE_BOARD_S: i64 = 180;
const OI_EVERY: Duration = Duration::from_secs(5 * 60 - 10);
const UNIVERSE_EVERY: Duration = Duration::from_secs(60 * 60);
/// Clients send nothing we read; anything bigger than this is refused.
const MAX_CLIENT_MESSAGE: usize = 4096;
const PING_EVERY: Duration = Duration::from_secs(30);
/// A client that cannot take one ~7 KB frame in this long is dropped. Without it, a
/// peer that vanished without a FIN (a phone losing signal) blocks its `send` until
/// the kernel's TCP retransmit timeout (~15 min) and holds a `RADAR_MAX_CLIENTS` slot.
const SEND_TIMEOUT: Duration = Duration::from_secs(10);
/// Requests in flight per batch; the Binance client enforces its own cap as well.
const BATCH: usize = 5;

struct Config {
    port: u16,
    universe: usize,
    top_n: usize,
    min_score: u32,
    interval_secs: u64,
    max_weight: u32,
    max_clients: usize,
    liq_warmup_mins: i64,
    fapi: String,
    fstream: String,
    oneshot: bool,
}

fn env_or<T: FromStr>(key: &str, default: T) -> T {
    match std::env::var(key) {
        Ok(v) if !v.trim().is_empty() => v.trim().parse().unwrap_or_else(|_| {
            warn!("{key}={v:?} is not valid; using the default");
            default
        }),
        _ => default,
    }
}

impl Config {
    fn from_env() -> Self {
        Config {
            port: env_or("RADAR_PORT", 8090),
            universe: env_or("RADAR_UNIVERSE", 150usize).clamp(1, 500),
            top_n: env_or("RADAR_TOP_N", 20usize).clamp(1, 100),
            min_score: env_or("RADAR_MIN_SCORE", 40u32).min(100),
            interval_secs: env_or("RADAR_SCORE_INTERVAL_SECS", 60u64).clamp(15, 900),
            max_weight: env_or("RADAR_MAX_WEIGHT_1M", 1200u32).clamp(100, 2400),
            max_clients: env_or("RADAR_MAX_CLIENTS", 2000usize).max(1),
            liq_warmup_mins: env_or("RADAR_LIQ_WARMUP_MINS", 60i64).clamp(0, 240),
            fapi: env_or("BINANCE_FAPI_BASE", "https://fapi.binance.com".to_string()),
            fstream: env_or("BINANCE_FSTREAM_BASE", "wss://fstream.binance.com".to_string()),
            oneshot: env_or("RADAR_ONESHOT", 0u8) == 1,
        }
    }
}

#[derive(Clone)]
struct HealthState {
    status: &'static str,
    as_of: i64,
    universe: usize,
    scored: usize,
}

struct Shared {
    frame: RwLock<Utf8Bytes>,
    tx: broadcast::Sender<Utf8Bytes>,
    health: Mutex<HealthState>,
    clients: AtomicUsize,
    max_clients: usize,
    liqs: Mutex<Liquidations>,
    universe: RwLock<HashSet<String>>,
    binance: Arc<Binance>,
}

impl Shared {
    fn new(binance: Arc<Binance>, max_clients: usize, liq_warmup_mins: i64) -> Arc<Self> {
        let board = Board::warming_up(now_ms() / 1000, 0);
        let (tx, _) = broadcast::channel(16);
        Arc::new(Shared {
            frame: RwLock::new(Utf8Bytes::from(serde_json::to_string(&board).expect("board json"))),
            tx,
            health: Mutex::new(HealthState { status: board.status, as_of: board.as_of, universe: 0, scored: 0 }),
            clients: AtomicUsize::new(0),
            max_clients,
            liqs: Mutex::new(Liquidations::new(liq_warmup_mins * 60_000)),
            universe: RwLock::new(HashSet::new()),
            binance,
        })
    }

    /// Serialize once, store as the snapshot new clients get, and fan it out.
    fn publish(&self, board: &Board) {
        let frame = Utf8Bytes::from(serde_json::to_string(board).expect("board json"));
        *self.frame.write().unwrap() = frame.clone();
        // An error only means nobody is connected right now.
        let _ = self.tx.send(frame);
        *self.health.lock().unwrap() =
            HealthState { status: board.status, as_of: board.as_of, universe: board.universe, scored: board.scored };
    }
}

// ── Scanner ──────────────────────────────────────────────────────────────────

struct Scanner {
    symbols: Vec<String>,
    series: HashMap<String, Vec<Bar>>,
    oi: HashMap<String, Option<f64>>,
}

impl Scanner {
    /// Poll every symbol's 15m klines: a full 1000-bar backfill for a symbol with no
    /// series OR fewer than `MIN_BARS` (a short series is never scored, and topping it
    /// up 3 bars a minute would take ~75 h to qualify — after a missed-poll gap made
    /// `merge_bars` replace it, for instance), otherwise just enough bars to reach back
    /// to the last one held (3 in steady state: the forming bar and two closed ones,
    /// weight 1).
    async fn poll_klines(&mut self, bn: &Binance) {
        let now = now_ms();
        let current = now - now.rem_euclid(score::BAR_MS);
        let jobs: Vec<(String, usize)> = self
            .symbols
            .iter()
            .map(|s| {
                // ponytail: a listing younger than ~3 days has under MIN_BARS of history
                // at all, so it re-requests 1000 bars (weight 5, not 1) every pass until
                // it qualifies; fine for the handful in the top 150.
                let held = self.series.get(s).filter(|v| v.len() > score::MIN_BARS);
                let limit = held.and_then(|v| v.last()).map_or(score::KEEP_BARS, |last| {
                    (((current - last.t) / score::BAR_MS) as usize + 1).clamp(3, score::KEEP_BARS)
                });
                (s.clone(), limit)
            })
            .collect();
        let results: Vec<_> = futures_util::stream::iter(jobs)
            .map(|(s, limit)| async move {
                let r = bn.klines(&s, limit).await;
                (s, r)
            })
            .buffer_unordered(BATCH)
            .collect()
            .await;
        let mut failed = 0;
        let mut first_err = None;
        for (s, r) in results {
            match r {
                Ok(bars) => merge_bars(self.series.entry(s).or_default(), bars),
                Err(e) => {
                    failed += 1;
                    first_err.get_or_insert(e);
                }
            }
        }
        if let Some(e) = first_err {
            warn!("[scan] {failed}/{} kline polls failed; first: {e}", self.symbols.len());
        }
    }

    async fn refresh_oi(&mut self, bn: &Binance) {
        let results: Vec<_> = futures_util::stream::iter(self.symbols.clone())
            .map(|s| async move {
                let r = bn.oi_change_4h(&s).await;
                (s, r)
            })
            .buffer_unordered(BATCH)
            .collect()
            .await;
        let mut failed = 0;
        for (s, r) in results {
            match r {
                Ok(v) => {
                    self.oi.insert(s, v);
                }
                // Keep the previous reading: at most one refresh (5 min) old.
                Err(_) => failed += 1,
            }
        }
        if failed > 0 {
            warn!("[scan] {failed}/{} OI refreshes failed", self.symbols.len());
        }
    }

    fn score(&self, shared: &Shared, cfg: &Config, now: i64) -> Board {
        // Copy the liquidation windows out first: the feed task locks the same mutex on
        // every frame, and scoring is the slow part.
        let (ready, windows) = {
            let mut liqs = shared.liqs.lock().unwrap();
            liqs.prune(now);
            let ready = liqs.ready(now);
            let w: HashMap<&str, _> =
                if ready { self.symbols.iter().map(|s| (s.as_str(), liqs.window(s, now, HOUR_MS))).collect() } else { HashMap::new() };
            (ready, w)
        };
        if ready {
            // What the liquidation factor is reading, in exact dollars, so it can be checked
            // against the raw stream: the board shows at most 3 tags and the 5-point
            // liquidation tag rarely makes the cut.
            let gross = |w: &LiqWindow| w.long_usd + w.short_usd;
            let mut active: Vec<(&str, LiqWindow)> = windows.iter().filter(|(_, w)| gross(w) > 0.0).map(|(s, w)| (*s, *w)).collect();
            active.sort_by(|a, b| gross(&b.1).total_cmp(&gross(&a.1)).then(a.0.cmp(b.0)));
            let (long, short) = active.iter().fold((0.0, 0.0), |(l, s), (_, w)| (l + w.long_usd, s + w.short_usd));
            let top: Vec<String> =
                active.iter().take(5).map(|(s, w)| format!("{s} longs ${:.0} / shorts ${:.0}", w.long_usd, w.short_usd)).collect();
            info!("[liq] last 1h: {} symbols, longs ${long:.0}, shorts ${short:.0}; top: {}", active.len(), top.join(", "));
        }
        let scored: Vec<Scored> = tokio::task::block_in_place(|| {
            self.symbols
                .iter()
                .filter_map(|s| {
                    score_symbol(&SymbolInputs {
                        symbol: s,
                        bars: self.series.get(s)?,
                        now_ms: now,
                        oi_4h_change_pct: self.oi.get(s).copied().flatten(),
                        liq_1h: windows.get(s.as_str()).copied(),
                    })
                })
                .collect()
        });
        build_board(scored, now / 1000, self.symbols.len(), ready, &Cfg { top_n: cfg.top_n, min_score: cfg.min_score })
    }
}

async fn fetch_universe(bn: &Binance, n: usize) -> Option<Vec<String>> {
    match bn.universe(n).await {
        Ok(u) if !u.is_empty() => Some(u),
        Ok(_) => {
            warn!("[scan] the universe came back empty");
            None
        }
        Err(e) => {
            warn!("[scan] universe fetch failed: {e}");
            None
        }
    }
}

fn set_universe(shared: &Shared, symbols: &[String]) {
    *shared.universe.write().unwrap() = symbols.iter().cloned().collect();
}

/// The scan loop. With `cfg.oneshot` it runs one full pass, prints the board and
/// returns; otherwise it never returns.
async fn run_scanner(shared: Arc<Shared>, cfg: Arc<Config>) {
    let bn = shared.binance.clone();
    let symbols = loop {
        if let Some(u) = fetch_universe(&bn, cfg.universe).await {
            break u;
        }
        if cfg.oneshot {
            return;
        }
        tokio::time::sleep(Duration::from_secs(30)).await;
    };
    info!("[scan] universe: {} symbols ({} … {})", symbols.len(), symbols[0], symbols[symbols.len() - 1]);
    set_universe(&shared, &symbols);
    shared.publish(&Board::warming_up(now_ms() / 1000, symbols.len()));

    let mut scan = Scanner { symbols, series: HashMap::new(), oi: HashMap::new() };
    let mut universe_at = Instant::now();
    let mut oi_at: Option<Instant> = None;
    let mut pass: u64 = 0;
    loop {
        let started = Instant::now();
        if universe_at.elapsed() >= UNIVERSE_EVERY {
            universe_at = Instant::now();
            if let Some(u) = fetch_universe(&bn, cfg.universe).await {
                let keep: HashSet<&String> = u.iter().collect();
                scan.series.retain(|k, _| keep.contains(k));
                scan.oi.retain(|k, _| keep.contains(k));
                set_universe(&shared, &u);
                scan.symbols = u;
            }
        }
        scan.poll_klines(&bn).await;
        if oi_at.is_none_or(|t| t.elapsed() >= OI_EVERY) {
            oi_at = Some(Instant::now());
            scan.refresh_oi(&bn).await;
        }
        let board = scan.score(&shared, &cfg, now_ms());
        info!(
            "[scan] pass {pass}: scored {}/{}, OI {}, tilt {} bullish / {} bearish, listed {}+{}, liquidations {}, weight {} (peak {}), {:.1}s",
            board.scored,
            board.universe,
            scan.oi.values().filter(|v| v.is_some()).count(),
            board.tilt.bullish,
            board.tilt.bearish,
            board.bullish.len(),
            board.bearish.len(),
            if board.liquidations_ready { "ready" } else { "warming up" },
            bn.used_weight.load(Ordering::Relaxed),
            bn.peak_weight.load(Ordering::Relaxed),
            started.elapsed().as_secs_f64()
        );
        if cfg.oneshot {
            println!("{}", serde_json::to_string_pretty(&board).expect("board json"));
            return;
        }
        shared.publish(&board);
        pass += 1;

        // Sleep to the next interval boundary + 5 s, so a pass reads bars Binance has
        // just closed rather than ones about to close.
        let every = cfg.interval_secs as i64 * 1000;
        let now = now_ms();
        let next = (now / every + 1) * every + 5_000;
        tokio::time::sleep(Duration::from_millis((next - now) as u64)).await;
    }
}

// ── Server ───────────────────────────────────────────────────────────────────

fn router(shared: Arc<Shared>) -> Router {
    Router::new().route("/health", get(health)).fallback(ws_handler).with_state(shared)
}

async fn health(State(s): State<Arc<Shared>>) -> Response {
    let h = s.health.lock().unwrap().clone();
    let age = now_ms() / 1000 - h.as_of;
    let (seen, ready) = {
        let l = s.liqs.lock().unwrap();
        (l.seen, l.ready(now_ms()))
    };
    let body = serde_json::json!({
        "status": h.status,
        "as_of": h.as_of,
        "board_age_s": age,
        "universe": h.universe,
        "scored": h.scored,
        "clients": s.clients.load(Ordering::Relaxed),
        "used_weight_1m": s.binance.used_weight.load(Ordering::Relaxed),
        "liquidations_seen": seen,
        "liquidations_ready": ready,
    });
    let code = if h.status == "ready" && age > STALE_BOARD_S { StatusCode::SERVICE_UNAVAILABLE } else { StatusCode::OK };
    (code, axum::Json(body)).into_response()
}

async fn ws_handler(ws: WebSocketUpgrade, State(s): State<Arc<Shared>>) -> Response {
    if s.clients.load(Ordering::Relaxed) >= s.max_clients {
        return (StatusCode::SERVICE_UNAVAILABLE, "too many clients").into_response();
    }
    ws.max_message_size(MAX_CLIENT_MESSAGE).max_frame_size(MAX_CLIENT_MESSAGE).on_upgrade(move |socket| client(socket, s))
}

/// Decrements the client count on every way out of `client`.
struct ClientSlot<'a>(&'a AtomicUsize);
impl Drop for ClientSlot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Send one message, failing on an error OR on `SEND_TIMEOUT`.
async fn send(socket: &mut WebSocket, m: Message) -> bool {
    matches!(tokio::time::timeout(SEND_TIMEOUT, socket.send(m)).await, Ok(Ok(())))
}

async fn client(mut socket: WebSocket, s: Arc<Shared>) {
    s.clients.fetch_add(1, Ordering::Relaxed);
    let _slot = ClientSlot(&s.clients);
    // Subscribe BEFORE reading the snapshot: a board published in between then arrives
    // twice (harmless) instead of never.
    let mut rx = s.tx.subscribe();
    let snapshot = s.frame.read().unwrap().clone();
    if !send(&mut socket, Message::Text(snapshot)).await {
        return;
    }
    let mut ping = tokio::time::interval(PING_EVERY);
    ping.tick().await;
    loop {
        tokio::select! {
            r = rx.recv() => {
                let frame = match r {
                    Ok(f) => f,
                    // Too slow to keep up: skip straight to the newest board.
                    Err(RecvError::Lagged(_)) => s.frame.read().unwrap().clone(),
                    Err(RecvError::Closed) => break,
                };
                if !send(&mut socket, Message::Text(frame)).await {
                    break;
                }
            }
            _ = ping.tick() => {
                if !send(&mut socket, Message::Ping(Bytes::new())).await {
                    break;
                }
            }
            m = socket.recv() => match m {
                None | Some(Err(_)) | Some(Ok(Message::Close(_))) => break,
                Some(Ok(_)) => {}
            },
        }
    }
}

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cfg = Arc::new(Config::from_env());
    let bn = Arc::new(Binance::new(cfg.fapi.clone(), cfg.max_weight));
    let shared = Shared::new(bn, cfg.max_clients, cfg.liq_warmup_mins);

    if cfg.oneshot {
        let started = Instant::now();
        run_scanner(shared.clone(), cfg.clone()).await;
        info!(
            "[oneshot] done in {:.1}s, peak used weight {}",
            started.elapsed().as_secs_f64(),
            shared.binance.peak_weight.load(Ordering::Relaxed)
        );
        return;
    }

    let feed_state = shared.clone();
    let url = format!("{}/market/ws/!forceOrder@arr", cfg.fstream.trim_end_matches('/'));
    tokio::spawn(binance::liquidation_feed(url, move |raw| {
        if let Some(l) = parse_force_order(raw)
            && feed_state.universe.read().unwrap().contains(&l.symbol)
        {
            feed_state.liqs.lock().unwrap().record(l, now_ms());
        }
    }));
    tokio::spawn(run_scanner(shared.clone(), cfg.clone()));

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", cfg.port)).await.unwrap_or_else(|e| panic!("bind :{}: {e}", cfg.port));
    info!("[server] listening on :{} (WebSocket on any path, /health)", cfg.port);
    axum::serve(listener, router(shared)).await.expect("server");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_tungstenite::tungstenite::Message as WsMessage;

    async fn serve(shared: Arc<Shared>) -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router(shared)).await.unwrap() });
        addr
    }

    fn shared() -> Arc<Shared> {
        Shared::new(Arc::new(Binance::new("http://127.0.0.1:9".into(), 1200)), 10, 60)
    }

    async fn next_text(ws: &mut (impl futures_util::Stream<Item = Result<WsMessage, tokio_tungstenite::tungstenite::Error>> + Unpin)) -> String {
        loop {
            let m = tokio::time::timeout(Duration::from_secs(5), ws.next()).await.expect("a frame in time").unwrap().unwrap();
            if let WsMessage::Text(t) = m {
                return t.as_str().to_string();
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_new_client_gets_the_snapshot_first_then_broadcasts() {
        let s = shared();
        let mut board = Board::warming_up(now_ms() / 1000, 150);
        board.status = "ready";
        s.publish(&board);
        let addr = serve(s.clone()).await;

        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/any/path")).await.unwrap();
        let first: serde_json::Value = serde_json::from_str(&next_text(&mut ws).await).unwrap();
        assert_eq!((first["status"].as_str(), first["universe"].as_u64()), (Some("ready"), Some(150)));

        // Wait for the server to count the client, then publish a new board.
        for _ in 0..50 {
            if s.clients.load(Ordering::Relaxed) == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        board.universe = 151;
        s.publish(&board);
        let second: serde_json::Value = serde_json::from_str(&next_text(&mut ws).await).unwrap();
        assert_eq!(second["universe"].as_u64(), Some(151));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn health_reports_503_for_a_stale_board() {
        let s = shared();
        let addr = serve(s.clone()).await;
        let url = format!("http://{addr}/health");
        // Warming up is healthy however old.
        assert_eq!(reqwest::get(&url).await.unwrap().status(), 200);

        let mut board = Board::warming_up(now_ms() / 1000, 150);
        board.status = "ready";
        s.publish(&board);
        let r = reqwest::get(&url).await.unwrap();
        assert_eq!(r.status(), 200);
        let body: serde_json::Value = r.json().await.unwrap();
        assert_eq!((body["status"].as_str(), body["universe"].as_u64()), (Some("ready"), Some(150)));

        board.as_of -= STALE_BOARD_S + 5;
        s.publish(&board);
        assert_eq!(reqwest::get(&url).await.unwrap().status(), 503);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_plain_http_request_is_not_upgraded() {
        let addr = serve(shared()).await;
        let r = reqwest::get(format!("http://{addr}/")).await.unwrap();
        assert!(r.status().is_client_error(), "{}", r.status());
    }
}
