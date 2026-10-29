// quant-core/src/price_action.rs — Entry triggers at key levels.
//
// What a trader waits for once the levels are marked: how price behaves AT them, bar
// by bar, on closes:
//   1. breakout_close: a close beyond a level after a close on the other side;
//   2. retest_hold: after a breakout close, a bar that came back to the level and still
//      closed on the breakout side;
//   3. failed_breakout: after a breakout close, a close back through the level within
//      `FAILED_BREAKOUT_BARS` bars — reported in the direction of the failure;
//   4. sweep_reclaim: a wick beyond the level's far edge with the close back inside (the
//      stops behind the level taken, the level kept);
//   5. rejection_wick: a long wick INTO the level (not beyond it) from the side price
//      sits on, closing back away from it;
//   6. trendline_break: the first close through a trendline that was still respected
//      when the levels were drawn.
//
// The levels must be drawn on the bars BEFORE the window events are reported in
// (`bars[..from]`): a level has to exist before the bar that reacts to it, or a swing
// formed at bar n-4 would "explain" a close at n-9.
// ponytail: so a level formed inside the event window is not traded against; the
// upgrade path is a per-level "known from" index.
//
// Pure: no I/O, no clock, deterministic. Every index is into the input slice.

use crate::levels::{Trendline, TRENDLINE_ATR};
use crate::patterns::{prior_volume_sma, Candle, VOLUME_SMA_BARS};

/// "Back at the level": within a tenth of an ATR of its edge — a retest that stops a few
/// ticks short is still the retest a trader drew.
pub const TOUCH_ATR: f64 = 0.1;
/// A close back through the level within this many bars of the breakout is a failed
/// breakout; later than that the market has built above it and it is a new break.
pub const FAILED_BREAKOUT_BARS: usize = 3;
/// A rejection wick is at least half the bar's range: the rejection dominates the bar.
pub const REJECTION_WICK_RATIO: f64 = 0.5;
/// Volume at least this many times the prior 20-bar mean confirms the event bar — the
/// participation a break needs to be believed.
pub const VOLUME_CONFIRM_RATIO: f64 = 1.5;
/// Every event `level_events` reports.
pub const EVENT_KINDS: [&str; 6] =
    ["breakout_close", "retest_hold", "sweep_reclaim", "failed_breakout", "rejection_wick", "trendline_break"];
/// Strength a trendline break carries in the one-per-bar dedupe: a 3-touch line is about
/// a tested swing zone, below fresh order blocks and the dealing-range edges.
const TRENDLINE_STRENGTH: f64 = 0.6;

/// A level to watch: a zone `low..high`, or a line (`low == high`).
#[derive(Debug, Clone, PartialEq)]
pub struct PaLevel {
    pub low: f64,
    pub high: f64,
    /// What produced it (`swing_zone`, `ob_bullish`, `range_high`, ...).
    pub kind: String,
    /// 0-1: which of two levels an event on the same bar is reported against.
    pub strength: f64,
}

/// One trigger at one level.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelEvent {
    /// One of `EVENT_KINDS`.
    pub kind: &'static str,
    /// `bullish` or `bearish`: the side the event favours.
    pub direction: &'static str,
    /// The edge the rule tested: a zone's high for a bullish break / retest / failure
    /// or a bullish rejection, its low for the mirrors and for a bullish sweep (the
    /// high for a bearish sweep); a trendline's value at the event bar.
    pub level: f64,
    /// `PaLevel.kind`, or `trendline_support` / `trendline_resistance`.
    pub level_type: String,
    /// The bar that completed the event.
    pub idx: usize,
    /// The event bar's volume against `VOLUME_CONFIRM_RATIO` × the mean of the 20 bars
    /// before it; `None` when fewer than 20 bars precede it (no baseline is "cannot
    /// say", never a fabricated false).
    pub volume_confirmed: Option<bool>,
    pub detail: String,
}

fn event(
    bars: &[Candle],
    kind: &'static str,
    direction: &'static str,
    level: f64,
    level_type: &str,
    idx: usize,
    detail: String,
) -> LevelEvent {
    LevelEvent {
        kind,
        direction,
        level,
        level_type: level_type.to_string(),
        idx,
        volume_confirmed: prior_volume_sma(bars, idx, VOLUME_SMA_BARS)
            .map(|s| bars[idx].volume >= VOLUME_CONFIRM_RATIO * s),
        detail,
    }
}

/// The events at `levels` and `trendlines` on bars `from..`, newest first.
///
/// `from` is the first bar an event may be reported at: the caller drew `levels` and
/// `trendlines` on `bars[..from]`. It is clamped to at least 1 so every event bar has a
/// previous close. `atr` (ignored unless positive and finite) sizes the touch tolerance
/// and the trendline tolerance; without it only exact touches count. Levels with a
/// non-finite edge or `low > high` are skipped. One event per bar, kind and direction:
/// the strongest level's.
pub fn level_events(
    bars: &[Candle],
    from: usize,
    levels: &[PaLevel],
    trendlines: &[Trendline],
    atr: Option<f64>,
) -> Vec<LevelEvent> {
    let n = bars.len();
    let from = from.max(1);
    if from >= n {
        return Vec::new();
    }
    let atr = atr.filter(|a| a.is_finite() && *a > 0.0);
    let tol = atr.map_or(0.0, |a| a * TOUCH_ATR);
    let gap = |d: f64| atr.map_or(String::new(), |a| format!(" {:.2} ATR", d.abs() / a));
    let close = |k: usize| bars[k].close;
    let mut found: Vec<(f64, LevelEvent)> = Vec::new();

    for lv in levels {
        if !(lv.low.is_finite() && lv.high.is_finite() && lv.low <= lv.high) {
            continue;
        }
        let (lo, hi, ty, s) = (lv.low, lv.high, lv.kind.as_str(), lv.strength);
        for k in from..n {
            let (b, c, pc) = (&bars[k], close(k), close(k - 1));

            if c > hi && pc <= hi {
                let detail = format!("closed{} above it; the bar before closed at or below it", gap(c - hi));
                found.push((s, event(bars, "breakout_close", "bullish", hi, ty, k, detail)));
                // What followed it: the first close back through the level ends the
                // breakout (a failure when within FAILED_BREAKOUT_BARS); before that, the
                // first bar back at the level that still closed above it is the retest.
                // A retest followed by a failure reports both — the newer one leads.
                let mut retested = false;
                for j in k + 1..n {
                    if close(j) <= hi {
                        if j - k <= FAILED_BREAKOUT_BARS {
                            let detail = format!("closed above it {} bar(s) earlier, then closed back below it", j - k);
                            found.push((s, event(bars, "failed_breakout", "bearish", hi, ty, j, detail)));
                        }
                        break;
                    }
                    if !retested && bars[j].low <= hi + tol {
                        retested = true;
                        let detail =
                            format!("broke above it {} bar(s) earlier, came back to it and closed above it", j - k);
                        found.push((s, event(bars, "retest_hold", "bullish", hi, ty, j, detail)));
                    }
                }
            }
            if c < lo && pc >= lo {
                let detail = format!("closed{} below it; the bar before closed at or above it", gap(lo - c));
                found.push((s, event(bars, "breakout_close", "bearish", lo, ty, k, detail)));
                let mut retested = false;
                for j in k + 1..n {
                    if close(j) >= lo {
                        if j - k <= FAILED_BREAKOUT_BARS {
                            let detail = format!("closed below it {} bar(s) earlier, then closed back above it", j - k);
                            found.push((s, event(bars, "failed_breakout", "bullish", lo, ty, j, detail)));
                        }
                        break;
                    }
                    if !retested && bars[j].high >= lo - tol {
                        retested = true;
                        let detail =
                            format!("broke below it {} bar(s) earlier, came back to it and closed below it", j - k);
                        found.push((s, event(bars, "retest_hold", "bearish", lo, ty, j, detail)));
                    }
                }
            }

            // A sweep's wick goes BEYOND the far edge (`b.low < lo`); a rejection's wick
            // only reaches INTO the level (`b.low >= lo`), so the two never share a bar
            // and a direction.
            if b.low < lo && c >= lo && pc >= lo {
                let detail = format!("wick{} below it, closed back above it", gap(lo - b.low));
                found.push((s, event(bars, "sweep_reclaim", "bullish", lo, ty, k, detail)));
            }
            if b.high > hi && c <= hi && pc <= hi {
                let detail = format!("wick{} above it, closed back below it", gap(b.high - hi));
                found.push((s, event(bars, "sweep_reclaim", "bearish", hi, ty, k, detail)));
            }
            let range = b.range();
            if pc > hi
                && c > hi
                && b.low >= lo
                && b.low <= hi + tol
                && range > 0.0
                && b.lower_shadow() >= REJECTION_WICK_RATIO * range
            {
                let detail = format!("{:.0}% lower wick into it, closed above it", b.lower_shadow() / range * 100.0);
                found.push((s, event(bars, "rejection_wick", "bullish", hi, ty, k, detail)));
            }
            if pc < lo
                && c < lo
                && b.high <= hi
                && b.high >= lo - tol
                && range > 0.0
                && b.upper_shadow() >= REJECTION_WICK_RATIO * range
            {
                let detail = format!("{:.0}% upper wick into it, closed below it", b.upper_shadow() / range * 100.0);
                found.push((s, event(bars, "rejection_wick", "bearish", lo, ty, k, detail)));
            }
        }
    }

    // The same line and the same "through" as `levels::trendlines`, so a break here is
    // the close that would turn the line `broken` there.
    let tl_tol = atr.map_or(0.0, |a| a * TRENDLINE_ATR);
    for t in trendlines.iter().filter(|t| t.status == "respected") {
        let support = t.kind == "support";
        let at = |x: usize| t.first_touch_price + t.slope_per_bar * (x as f64 - t.first_touch_idx as f64);
        let through = |x: usize| if support { close(x) < at(x) - tl_tol } else { close(x) > at(x) + tl_tol };
        if let Some(k) = (from..n).find(|&k| through(k) && !through(k - 1)) {
            let (direction, side, level_type) =
                if support { ("bearish", "below", "trendline_support") } else { ("bullish", "above", "trendline_resistance") };
            let detail = format!("closed {} its {}-touch {} trendline", side, t.touches, t.kind);
            found.push((TRENDLINE_STRENGTH, event(bars, "trendline_break", direction, at(k), level_type, k, detail)));
        }
    }

    found.sort_by(|a, b| b.1.idx.cmp(&a.1.idx).then(b.0.total_cmp(&a.0)));
    let mut out: Vec<LevelEvent> = Vec::new();
    for (_, e) in found {
        if !out.iter().any(|o| o.idx == e.idx && o.kind == e.kind && o.direction == e.direction) {
            out.push(e);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levels::analyze_levels;
    use proptest::prelude::*;

    fn bar(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Candle {
        Candle { open, high, low, close, volume, taker_buy_volume: None }
    }

    fn zone(low: f64, high: f64, kind: &str, strength: f64) -> PaLevel {
        PaLevel { low, high, kind: kind.to_string(), strength }
    }

    /// The level every example uses: 100-101, so with ATR 2 the touch tolerance is 0.2.
    fn level() -> Vec<PaLevel> {
        vec![zone(100.0, 101.0, "swing_zone", 0.6)]
    }

    fn summary(events: &[LevelEvent]) -> Vec<(&'static str, &'static str, f64, usize)> {
        events.iter().map(|e| (e.kind, e.direction, e.level, e.idx)).collect()
    }

    // Every fixture below was checked so that NO other rule fires on it: a low under 100
    // adds a bullish sweep, a high over 101 with the previous close at or below 101 a
    // bearish sweep, and a lower wick of half the range at the level a rejection.
    // Re-check any fixture you change.

    /// Bar 0 closes at 100 (inside), bar 1 at 102 (above 101, low 100.1 not under 100).
    fn breakout() -> Vec<Candle> {
        vec![bar(99.0, 100.5, 98.5, 100.0, 10.0), bar(100.2, 102.5, 100.1, 102.0, 10.0)]
    }

    #[test]
    fn a_close_above_the_level_is_a_bullish_breakout() {
        let ev = level_events(&breakout(), 1, &level(), &[], Some(2.0));
        assert_eq!(summary(&ev), vec![("breakout_close", "bullish", 101.0, 1)]);
        // 102 - 101 = 1 = 0.5 ATR.
        assert!(ev[0].detail.starts_with("closed 0.50 ATR above it"), "{}", ev[0].detail);
        assert_eq!(ev[0].level_type, "swing_zone");
        assert_eq!(ev[0].volume_confirmed, None, "one bar before it, not twenty");
    }

    #[test]
    fn coming_back_to_the_level_and_closing_above_it_holds_the_retest() {
        // Bar 2: low 101.1 is within 101 + 0.2, the close 102.3 stays above 101, and its
        // lower wick (101.5 - 101.1 = 0.4 of a 1.5 range) is too short for a rejection.
        let mut bars = breakout();
        bars.push(bar(101.5, 102.6, 101.1, 102.3, 10.0));
        let ev = level_events(&bars, 1, &level(), &[], Some(2.0));
        assert_eq!(summary(&ev), vec![("retest_hold", "bullish", 101.0, 2), ("breakout_close", "bullish", 101.0, 1)]);
        assert_eq!(ev[0].detail, "broke above it 1 bar(s) earlier, came back to it and closed above it");
    }

    #[test]
    fn closing_back_below_it_fails_the_breakout() {
        // Bar 2 closes at 100.5, back under 101. Its high 102.2 is over 101, but the
        // previous close (102) was above 101, so it is not a bearish sweep.
        let mut bars = breakout();
        bars.push(bar(102.0, 102.2, 100.2, 100.5, 10.0));
        let ev = level_events(&bars, 1, &level(), &[], Some(2.0));
        assert_eq!(summary(&ev), vec![("failed_breakout", "bearish", 101.0, 2), ("breakout_close", "bullish", 101.0, 1)]);
        assert_eq!(ev[0].detail, "closed above it 1 bar(s) earlier, then closed back below it");

        // A held retest that then closes back below within three bars reports both.
        let mut bars = breakout();
        bars.push(bar(101.5, 102.6, 101.1, 102.3, 10.0));
        bars.push(bar(102.3, 102.4, 100.3, 100.5, 10.0));
        let ev = level_events(&bars, 1, &level(), &[], Some(2.0));
        assert_eq!(
            summary(&ev),
            vec![
                ("failed_breakout", "bearish", 101.0, 3),
                ("retest_hold", "bullish", 101.0, 2),
                ("breakout_close", "bullish", 101.0, 1),
            ]
        );
    }

    #[test]
    fn a_wick_under_the_level_closing_back_above_it_is_a_sweep() {
        // Bar 1: low 99.5 under 100, close 100.6 back above, previous close 101 above.
        // Its high is exactly 101 (no bearish sweep) and the previous close is not above
        // 101 (no rejection).
        let bars = vec![bar(101.0, 101.5, 100.5, 101.0, 10.0), bar(101.0, 101.0, 99.5, 100.6, 10.0)];
        let ev = level_events(&bars, 1, &level(), &[], Some(2.0));
        assert_eq!(summary(&ev), vec![("sweep_reclaim", "bullish", 100.0, 1)]);
        assert_eq!(ev[0].detail, "wick 0.25 ATR below it, closed back above it");
    }

    #[test]
    fn a_long_wick_into_the_level_from_below_is_a_bearish_rejection() {
        // Level 105-106. Bar 1 wicks to 105.2 (inside, not beyond 106), closes 103.2;
        // upper wick 2.0 of a 2.3 range.
        let lv = vec![zone(105.0, 106.0, "ob_bearish", 0.65)];
        let bars = vec![bar(103.0, 103.5, 102.5, 103.0, 10.0), bar(103.0, 105.2, 102.9, 103.2, 10.0)];
        let ev = level_events(&bars, 1, &lv, &[], Some(2.0));
        assert_eq!(summary(&ev), vec![("rejection_wick", "bearish", 105.0, 1)]);
        assert_eq!(ev[0].detail, "87% upper wick into it, closed below it");
    }

    #[test]
    fn the_first_close_through_a_respected_trendline_breaks_it() {
        let line = Trendline {
            kind: "support",
            slope_per_bar: 0.0,
            touches: 3,
            first_touch_idx: 0,
            first_touch_price: 100.0,
            last_touch_idx: 2,
            price_at_last: 100.0,
            status: "respected",
        };
        // ATR 2: tolerance 0.4, so "through" is a close under 99.6. Bar 3 closes at 99.
        let mut bars: Vec<Candle> = (0..3).map(|_| bar(101.0, 101.2, 100.8, 101.0, 10.0)).collect();
        bars.push(bar(101.0, 101.1, 98.8, 99.0, 10.0));
        let ev = level_events(&bars, 1, &[], &[line.clone()], Some(2.0));
        assert_eq!(summary(&ev), vec![("trendline_break", "bearish", 100.0, 3)]);
        assert_eq!(ev[0].level_type, "trendline_support");
        assert_eq!(ev[0].detail, "closed below its 3-touch support trendline");
        // A line already broken when the levels were drawn breaks nothing new.
        let broken = Trendline { status: "broken", ..line };
        assert!(level_events(&bars, 1, &[], &[broken], Some(2.0)).is_empty());
    }

    #[test]
    fn volume_confirms_against_the_twenty_bars_before() {
        // 21 flat-ish bars of volume 10, then the breakout bar. `from` 21: from 1 every
        // flat bar (low 99.5 under 100, close 100) would also read as a bullish sweep.
        let run = |volume: f64| {
            let mut bars: Vec<Candle> = (0..21).map(|_| bar(100.0, 100.5, 99.5, 100.0, 10.0)).collect();
            bars.push(bar(100.2, 102.5, 100.1, 102.0, volume));
            level_events(&bars, 21, &level(), &[], Some(2.0))
        };
        // Mean of bars 1..=20 is 10: 16 >= 15, 14 is not.
        let ev = run(16.0);
        assert_eq!(summary(&ev), vec![("breakout_close", "bullish", 101.0, 21)]);
        assert_eq!(ev[0].volume_confirmed, Some(true));
        assert_eq!(run(14.0)[0].volume_confirmed, Some(false));
    }

    #[test]
    fn one_event_per_bar_and_kind_from_the_strongest_level() {
        let lv = vec![zone(100.0, 101.0, "swing_zone", 0.6), zone(100.0, 101.0, "ob_bullish", 0.8)];
        let ev = level_events(&breakout(), 1, &lv, &[], Some(2.0));
        assert_eq!(summary(&ev), vec![("breakout_close", "bullish", 101.0, 1)]);
        assert_eq!(ev[0].level_type, "ob_bullish");
    }

    #[test]
    fn nothing_is_reported_before_from() {
        // The breakout is bar 1; bar 2 neither breaks (previous close above) nor comes
        // back to the level (low 101.5 > 101.2), and follow-ups only run from a reported
        // breakout.
        let mut bars = breakout();
        bars.push(bar(102.0, 102.6, 101.5, 102.4, 10.0));
        assert!(level_events(&bars, 2, &level(), &[], Some(2.0)).is_empty());
    }

    #[test]
    fn degenerate_input_reports_nothing() {
        assert!(level_events(&[], 0, &level(), &[], Some(2.0)).is_empty());
        assert!(level_events(&breakout(), 2, &level(), &[], Some(2.0)).is_empty());
        let bad = vec![zone(f64::NAN, 101.0, "swing_zone", 0.6), zone(101.0, 100.0, "swing_zone", 0.6)];
        assert!(level_events(&breakout(), 1, &bad, &[], Some(2.0)).is_empty());
        // Without an ATR the breakout still reads; only the distance is left out.
        let ev = level_events(&breakout(), 0, &level(), &[], None);
        assert_eq!(ev[0].detail, "closed above it; the bar before closed at or below it");
    }

    /// A copy of `levels::tests::walk` (private there): a bounded random walk, some bars
    /// flat, wicks of random size, never below 0.5, volume 10.
    fn walk() -> impl Strategy<Value = Vec<Candle>> {
        prop::collection::vec((-3.0f64..3.0, 0.0f64..2.0, 0.0f64..2.0, 0u8..5), 0..260).prop_map(|steps| {
            let mut price = 1000.0f64;
            steps
                .into_iter()
                .map(|(delta, up, down, flat)| {
                    let open = price;
                    let close = if flat == 0 { open } else { (open + delta).max(1.0) };
                    price = close;
                    bar(open, open.max(close) + up, (open.min(close) - down).max(0.5), close, 10.0)
                })
                .collect()
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn events_follow_their_rules(
            candles in walk(),
            split in 0usize..300,
            extra in prop::collection::vec((900.0f64..1100.0, 0.0f64..5.0), 3)
        ) {
            let n = candles.len();
            let from = split.min(n);
            let kl = analyze_levels(&candles[..from]);
            let mut levels: Vec<PaLevel> = kl
                .as_ref()
                .map(|k| k.zones.iter().map(|z| zone(z.low, z.high, "swing_zone", z.strength)).collect())
                .unwrap_or_default();
            levels.extend(extra.iter().map(|&(low, w)| zone(low, low + w, "random", 0.5)));
            let lines: Vec<Trendline> = kl.as_ref().map(|k| k.trendlines.clone()).unwrap_or_default();
            let atr = kl.as_ref().and_then(|k| k.atr);

            let ev = level_events(&candles, from, &levels, &lines, atr);
            prop_assert_eq!(&ev, &level_events(&candles, from, &levels, &lines, atr));
            for e in &ev {
                prop_assert!(from.max(1) <= e.idx && e.idx < n, "{:?}", e);
                prop_assert!(EVENT_KINDS.contains(&e.kind));
                prop_assert!(e.direction == "bullish" || e.direction == "bearish");
                prop_assert!(e.level.is_finite());
                prop_assert_eq!(e.volume_confirmed.is_some(), e.idx >= VOLUME_SMA_BARS);
                let (c, pc, b) = (candles[e.idx].close, candles[e.idx - 1].close, &candles[e.idx]);
                let bull = e.direction == "bullish";
                match e.kind {
                    "breakout_close" if bull => prop_assert!(c > e.level && pc <= e.level),
                    "breakout_close" => prop_assert!(c < e.level && pc >= e.level),
                    "sweep_reclaim" if bull => prop_assert!(b.low < e.level && c >= e.level && pc >= e.level),
                    "sweep_reclaim" => prop_assert!(b.high > e.level && c <= e.level && pc <= e.level),
                    "retest_hold" if bull => prop_assert!(c > e.level),
                    "retest_hold" => prop_assert!(c < e.level),
                    "failed_breakout" if bull => prop_assert!(c >= e.level),
                    "failed_breakout" => prop_assert!(c <= e.level),
                    _ => {}
                }
            }
            for w in ev.windows(2) {
                prop_assert!(w[0].idx >= w[1].idx, "not newest first: {:?}", w);
            }
            for (i, a) in ev.iter().enumerate() {
                prop_assert!(
                    ev[i + 1..].iter().all(|b| (b.idx, b.kind, b.direction) != (a.idx, a.kind, a.direction)),
                    "duplicate {:?}", a
                );
            }
        }
    }
}
