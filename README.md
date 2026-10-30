# Quant_Radar

Every minute this service scans the top 150 Binance USDⓈ-M USDT perpetuals. It ranks the strongest **bullish** and **bearish** 15m setups, confirmed by the 1h trend, and pushes the same board to every connected client over one WebSocket. The Strat AI terminal's **Quant Radar** panel displays it. From a row, a user charts the symbol or opens the AI agent to run Find Trade on it.

The service is independent of crypto-app. It reads only Binance public market data, which needs no API key, and has no database, Kafka or Redis. It runs on dev-vm next to the strat-sentiment research API, and browsers reach it at `wss://sentiment-api.stratai.live/ws/radar` (Cloudflare, then the VM's nginx).

## How a symbol is scored

Each symbol gets a 0-100 score for each side. It is listed on whichever side scores higher, if that score is at least `RADAR_MIN_SCORE`. The rules below are for bullish; bearish mirrors them. All weights are in one block at the top of `src/score.rs`. They are first-guess heuristics, to be tuned after watching the live board.

| Factor | Pts | Rule |
|---|---|---|
| 1h trend | 25 | the latest 1h break (BOS or CHoCH) is bullish and at most 72 bars old: 15; EMA stack bullish: +10 |
| 15m structure | 20 | the latest 15m break is bullish: BOS 20, CHoCH 15, full points up to 24 bars, fading to 0 at 48 |
| Zone | 20 | the nearest live bullish order block or open FVG at or below price: inside it 20, within 1 ATR 15, within 2 ATR 8 |
| Momentum | 15 | the 4h move in ATR(14), clamped to 0-3, × 5 |
| OI vs price (4h) | 10 | price at least +0.5% with OI at least +1% (new longs): 10; price up with OI at most −1% (shorts covering): 4 |
| Liquidity | 10 | equal lows swept within 24 bars: 5; shorts liquidated minus longs in the last hour at least 0.05% of 24h volume: 5 |

**Choppy symbols.** A symbol scores 0 on both sides only when all three hold: its 32-bar (8h) efficiency ratio is below 0.25, it has had no 15m break in 48 bars, and it has had no 1h break in 72 bars. A coin chopping on the 15m inside a live 1h trend still gets a side. (Before 2026-10-30 the 1h was not checked, and BTC read "No clear setup" at 0 while its 1h trend was clearly down.)

**Majors.** `RADAR_MAJORS` (default `BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,XAUUSDT`) is always scanned, even outside the top 150, and listed in the board's `majors` in that order whatever it scores. Each entry carries `side` (`bullish`, `bearish`, or `none` when choppy or balanced, with no tags or zone) and `extended`. A major still appears on the Bullish or Bearish list when it ranks there.

Gold is XAUUSDT, which Binance lists as a `TRADIFI_PERPETUAL`. TradFi contracts (stocks, commodities) are kept out of the ranked universe; a major is the only way one gets in. XAUUSDT was checked on 2026-10-30 to trade 24/7 (continuous 4h klines through a weekend), so it scores like any perp. A major Binance doesn't list as a tradable USDT perp is logged and skipped.

**Extended symbols.** A listed symbol is extended when its 4h move in the side's favour is at least 3 ATR and it has no live order block or FVG within 2 ATR of price. These are strong moves with nothing to enter at right now. They go in a separate `extended` list (up to 10 per side), so the main board holds setups a trader can act on. Their score is unchanged, and the tilt still counts them.

**Live order blocks.** quant-core marks a block `mitigated` at its first touch. The radar treats a block as live if price has not come back to it yet, or if the first retest was within the last 3 closed bars and no close has gone through the far edge.

**Liquidations.** These come from `!forceOrder@arr`. They count only after an hour of collection. Binance sends at most one liquidation per symbol per second, so the totals are a lower bound.

**Which contracts are scanned.** The universe is `PERPETUAL` contracts with status `TRADING` and quote asset USDT, excluding stablecoin bases. Binance's `TRADIFI_PERPETUAL` contracts (gold, stocks) are left out.

**Detectors.** The order block, FVG, liquidity pool and structure detectors are crypto-app's `quant-core`.

## The board frame (v1)

```json
{ "type": "board", "v": 1, "status": "ready", "as_of": 1767000000,
  "timeframe": "15m", "trend_timeframe": "1h", "universe": 150, "scored": 147,
  "tilt": { "bullish": 37, "bearish": 22 }, "liquidations_ready": true,
  "bullish": [ { "symbol": "SOLUSDT.P", "score": 82, "price": 182.41, "change_24h_pct": 4.1,
                 "zone": { "kind": "order_block", "low": 178.2, "high": 179.6, "distance_pct": -1.6 },
                 "tags": ["1h uptrend", "15m BOS up", "OI +6% 4h"] } ],
  "bearish": [],
  "extended": { "bullish": [ { "symbol": "ALPINEUSDT.P", "score": 55, "...": "same row shape" } ],
                "bearish": [] },
  "majors": [ { "symbol": "BTCUSDT.P", "score": 0, "side": "none", "extended": false, "tags": [], "zone": null, "...": "same row shape" },
              { "symbol": "XAUUSDT.P", "score": 59, "side": "bearish", "extended": false, "...": "same row shape" } ] }
```

`extended` and `majors` were added to v1 without a version bump: a client that doesn't know a field ignores it.

- `status` is `warming_up`, with empty lists, until the first full pass finishes (about 25 s after start).
- `zone` is `null` when there is no live zone. `zone.kind` is `order_block` or `fvg`. `distance_pct` is negative below price and positive above it.
- `tilt` counts every symbol at or above the minimum score, not only the listed ones.
- A client gets the latest frame immediately on connect, then one frame per pass.
- The service answers the WebSocket on every path and ignores anything a client sends.

## Run

```bash
cargo test                       # 31 tests, fixtures recorded from Binance
cargo run --release              # ws://127.0.0.1:8090/ and http://127.0.0.1:8090/health
RADAR_ONESHOT=1 cargo run --release > board.json   # one pass, print, exit
```

Configuration comes from the environment; every variable has a default. See `.env.example`.

`GET /health` returns a JSON body: `{status, as_of, board_age_s, universe, scored, clients, used_weight_1m, liquidations_seen, liquidations_ready}`. It answers 503 once the board is more than 180 s old after warm-up.

## Resources (measured 2026-10-29/30)

| | |
|---|---|
| Image | ~45 MB uncompressed (distroless/cc + the binary) |
| Memory | ~29 MiB idle; 1,000 clients add ~13 MiB (51 MiB peak). The per-socket read buffer is capped at 4 KiB; tungstenite's 128 KiB default cost ~122 MiB for the same 1,000. Limit 256 MiB. |
| CPU | ~13 s of mostly-waiting work per minute, under 1% of a core; a broadcast to 1,000 clients is a 1-2 s burst (0.5 CPU limit) |
| Binance | startup backfill ~790 weight once; steady state 150 weight/min (limit 2400/IP, self-capped at 1200); OI 150 calls per 5 min (separate 1000 per 5 min quota) |
| Bandwidth | one ~11.4 KB frame per client per minute (2.2 KB gzipped; the WebSocket is not compressed), ~16 GB/day at 1,000 clients connected all day |

## Deploy

On every push to `main`, CI (`.github/workflows/deploy.yml`) runs `cargo test`, then builds and pushes `ghcr.io/6829nkhpas/quant_radar:{sha,latest}`. While the repo variable `DEPLOY_HOST` is set, it then SSHes into dev-vm as `radar-deploy`, whose key can run only `/opt/quant-radar/update.sh` ([deploy/update.sh](deploy/update.sh)): pull `latest`, `docker compose up -d`, drop superseded radar images.

**Where it runs.** dev-vm: GCP `cs-project-sntynm1y`, `asia-south1-a`, e2-medium (2 vCPU, 4 GB), Ubuntu 24.04. The strat-sentiment research API shares the VM. The VM's IP is ephemeral, so a stop/start changes it: Cloudflare's DNS record and the `DEPLOY_HOST` variable would both need the new one.

**Path of a client.** Browser → Cloudflare (`sentiment-api.stratai.live`, proxied) → nginx on the VM's :443 (the firewall admits 443 only from Cloudflare; the origin certificate is self-signed for that name) → `location = /ws/radar` → `127.0.0.1:8090`. Each client is two nginx connections, so nginx.conf has `worker_connections 4096` and `worker_rlimit_nofile 8192` (the stock 768 allowed ~750 radar clients).

### One-time VM setup (done 2026-10-30)

1. **Service folder.** `/opt/quant-radar` (root-owned) holds this repo's `docker-compose.yml` and `update.sh` (root:root 755), plus a `.env` only if a default needs changing. The repo is public, so the image pulls without a registry login. Start: `cd /opt/quant-radar && sudo docker compose up -d`.
2. **nginx.** In `/etc/nginx/sites-available/sentiment-api`, inside the `sentiment-api.stratai.live` server:
   ```nginx
   location = /ws/radar {
       proxy_pass http://127.0.0.1:8090;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
       proxy_set_header Host $host;
       proxy_read_timeout 120s;
       proxy_send_timeout 120s;
   }
   ```
   Then `sudo nginx -t && sudo systemctl reload nginx`.
3. **Deploy user.** `radar-deploy`: no password, not in the `docker` group. `/etc/sudoers.d/radar-deploy` allows exactly `radar-deploy ALL=(root) NOPASSWD: /opt/quant-radar/update.sh`, and its `authorized_keys` line is `restrict,command="sudo -n /opt/quant-radar/update.sh" <CI public key>`.
4. **GitHub settings.** Secret `DEPLOY_SSH_KEY` (that key's private half), variables `DEPLOY_HOST`, `DEPLOY_USER=radar-deploy` and `DEPLOY_FINGERPRINT` (`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`, the `SHA256:…` field).

Locally, `docker compose up -d` (with `TAG=local` for an image you built) serves `ws://127.0.0.1:8090`.

## quant-core is a copy

`quant-core/` is a verbatim copy of `quant-core/Cargo.toml` and `quant-core/src/` from crypto-app (`github.com/thestratai/crypto-app`) at commit **`37453bd50dfd728838033dd99c340ef722229eeb`**, which had a clean working tree. It is copied rather than used as a git dependency, so this repo builds with no access to that private repo.

Fixes made in crypto-app reach the radar only when you re-sync:

```bash
rm -rf quant-core/src && cp -r ../crypto-app/quant-core/src quant-core/src
cp ../crypto-app/quant-core/Cargo.toml quant-core/Cargo.toml
git -C ../crypto-app rev-parse HEAD     # put this SHA in this section
cargo test
```

Never edit `quant-core/` here. If you do, the next re-sync will silently overwrite the change.
