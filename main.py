"""Forecast GTA VI trophy earn rates on Exophase at each case's target.

Approach: an earn rate is a ratio, earners / players tracked, and the two
counts behave very differently over a launch week. The players-tracked
count only ever jumps up, in steps, whenever Exophase refreshes the page,
and every such jump dilutes most rates on the spot. Earners per trophy, by
contrast, are a cumulative count: they never fall, and they grow as the
tracked cohort plays through the game. So instead of extrapolating the
rate itself, this program fits a separate growth curve to each count and
forecasts the ratio.

Both curves are straight lines on log-log axes: ln(count) against
ln(1 + hours since 2026-11-19T00:00Z), fitted by recency-weighted least
squares over the distinct page states in the case history (consecutive
reads of an unchanged page collapse into one state). Slopes are clamped
to sane ranges so a noisy early segment cannot extrapolate to nonsense.

The spread around each median is a logit-normal sigma built from the
baseline random-walk term 0.5*sqrt(horizon days), the model's own
in-sample logit residuals, a term that grows with how far the target
extrapolates past the last observed log-time, and the observed refresh
volatility: the typical absolute logit move a trophy's rate makes when
the page refreshes and the tracked-player count jumps. A history with
no refresh at all earns a narrow (but never zero) floor, since a flat
page has shown no dilution; a history whose refreshes move rates a lot
keeps a wide spread. Quantiles come from the standard normal z values
at the case's levels.

Each median also blends the ratio forecast with the trophy's last
observed rate, in logit space. When the history shows no refresh, the
last rate is nearly the whole story and carries most of the weight;
once refreshes are on record the ratio forecast dominates and the last
rate is only an anchor against extrapolation overshoot.

Standard library only, no network, deterministic: the same case always
produces the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Release-day epoch the growth curves are measured from.
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
# Standard normal quantiles for the levels this contest scores.
Z_BY_LEVEL = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Recency weighting time constant for the curve fits, in hours.
WEIGHT_TAU_H = 72.0
# Damping for the earners trend past the last read: earners are bounded by
# the players count, so as a trophy's rate matures its earners growth must
# converge to the players growth. Only this share of the earners slope in
# excess of the players slope is carried into the extrapolation.
EARNERS_TREND_DAMPING = 0.8
# Slope clamps for the log-log fits (players, earners).
PLAYERS_SLOPE_RANGE = (0.0, 2.0)
EARNERS_SLOPE_RANGE = (0.0, 3.0)
# How much the in-sample logit residual widens the forecast spread.
RESID_SPREAD_FACTOR = 1.5
# Extra spread per unit of log-time the target lies past the last read:
# trend error compounds the further the curves are extrapolated.
EXTRAP_SPREAD_PER_LOGTIME = 0.35
# How much the observed per-refresh logit volatility widens the spread.
# Half weight: the residual term already prices the refreshes the
# smooth curves failed to fit in sample, so this term only adds the
# forward-looking share of dilution uncertainty.
VOL_SPREAD_FACTOR = 0.5
# Spread floor for a history with no refresh at all: a flat page has
# shown no dilution, so the band may be narrow, but never zero. The
# floor also grows with the horizon (see sigma_floor), so a long
# unrefreshed history still prices unobserved dilution risk.
FLAT_SIGMA_FLOOR = 0.4
# Spread floor once at least one refresh is on record.
REFRESHED_SIGMA_FLOOR = 0.5
# Hard cap on any trophy's logit spread.
SIGMA_CAP = 3.0
# Weight on the last observed rate in the median blend (logit space).
# A flat history: the last rate carries most of the weight. Each
# refresh on record hands weight to the ratio forecast, down to the
# minimum anchor weight.
PERSIST_WEIGHT_FLAT = 0.85
PERSIST_WEIGHT_MIN = 0.3


def clamp_rate(pct):
    """Keep a percent inside the range that has a finite logit."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(z):
    if z >= 0:
        return 100.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp from a case file."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since_epoch(dt):
    return (dt - EPOCH).total_seconds() / 3600.0


def log_time(hours):
    """The curve feature: ln(1 + hours), guarded for reads before the epoch."""
    return math.log1p(max(hours, -0.9))


def weighted_line(xs, ys, ws, slope_lo, slope_hi):
    """Weighted least-squares line y = a + b*x with the slope clamped.

    Returns (a, b), or None when the points cannot support a line.
    """
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    slope = min(slope_hi, max(slope_lo, sxy / sxx))
    intercept = my - slope * mx
    return (intercept, slope)


def distinct_states(history):
    """Collapse consecutive reads of an unchanged page into one state each."""
    states = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != prev_key:
            states.append(read)
            prev_key = key
    return states


def refresh_stats(states):
    """Count refreshes and measure their typical logit move.

    A refresh is a distinct page state whose players_tracked differs
    from the previous state's. The volatility is the median absolute
    logit change across every trophy observed on both sides of every
    refresh boundary: how far a rate typically jumps when the tracked
    cohort is diluted. Returns (n_refresh, volatility), volatility 0.0
    when no refresh (or no comparable trophy) is on record.
    """
    moves = []
    n_refresh = 0
    for prev, cur in zip(states, states[1:]):
        p_prev = prev.get("players_tracked")
        p_cur = cur.get("players_tracked")
        if not p_prev or not p_cur or float(p_prev) == float(p_cur):
            continue
        n_refresh += 1
        rates_prev = prev.get("rates") or []
        rates_cur = cur.get("rates") or []
        for a, b in zip(rates_prev, rates_cur):
            if a is not None and b is not None:
                moves.append(abs(logit(float(b)) - logit(float(a))))
    if not moves:
        return n_refresh, 0.0
    moves.sort()
    mid = len(moves) // 2
    if len(moves) % 2:
        return n_refresh, moves[mid]
    return n_refresh, 0.5 * (moves[mid - 1] + moves[mid])


def sigma_floor(days, n_refresh):
    """Minimum logit spread for a case, by horizon and refresh record.

    No refresh on record: the flat floor, but never below the baseline
    random walk for the horizon, so a long unrefreshed history still
    prices the dilution it has not yet shown. With refreshes on
    record, a modest fixed floor; the quadrature terms do the rest.
    """
    if n_refresh <= 0:
        return max(FLAT_SIGMA_FLOOR, 0.5 * math.sqrt(days))
    return REFRESHED_SIGMA_FLOOR


def persist_weight(n_refresh):
    """Weight on the last observed rate in the median blend."""
    if n_refresh <= 0:
        return PERSIST_WEIGHT_FLAT
    return max(PERSIST_WEIGHT_MIN, PERSIST_WEIGHT_FLAT / (1.0 + n_refresh))


def read_time_hours(read):
    stamp = read.get("at") or read.get("slot")
    return hours_since_epoch(parse_time(stamp))


def recency_weights(times_h):
    if not times_h:
        return []
    last = max(times_h)
    return [math.exp(-(last - t) / WEIGHT_TAU_H) for t in times_h]


def horizon_days(case):
    minutes = case.get("horizon_minutes")
    if minutes:
        return float(minutes) / 1440.0
    try:
        span = parse_time(case["target"]) - parse_time(case["at"])
        return max(span.total_seconds() / 86400.0, 0.0)
    except Exception:
        return 1.0


def quantiles_from(median_pct, sigma, levels):
    center = logit(median_pct)
    out = []
    running = 0.0
    for i, level in enumerate(levels):
        z = Z_BY_LEVEL.get(level)
        if z is None:
            z = Z_BY_LEVEL[DEFAULT_LEVELS[min(i, len(DEFAULT_LEVELS) - 1)]]
        value = round(inv_logit(center + z * sigma), 4)
        value = min(100.0, max(0.0, value))
        running = max(running, value)
        out.append(running)
    return out


def last_known_rates(case):
    """Each trophy's most recent non-null rate in the history (or None)."""
    count = len(case["trophies"])
    found = [None] * count
    for read in case.get("history") or []:
        rates = read.get("rates") or []
        for i in range(min(count, len(rates))):
            if rates[i] is not None:
                found[i] = float(rates[i])
    return found


def persistence_rows(case, levels):
    """Fallback forecast: last rate held flat, baseline random-walk spread."""
    days = horizon_days(case)
    sigma = max(0.5 * math.sqrt(days), sigma_floor(days, 0))
    last = last_known_rates(case)
    known = sorted(r for r in last if r is not None)
    fill = known[len(known) // 2] if known else 1.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        median = last[i] if last[i] is not None else fill
        rows.append({"id": trophy["id"], "q": quantiles_from(median, sigma, levels)})
    return rows


def forecast_case(case):
    levels = case.get("levels") or DEFAULT_LEVELS
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        return {"trophies": persistence_rows(case, levels)}

    states = distinct_states(history)
    times_h = [read_time_hours(s) for s in states]
    xs = [log_time(t) for t in times_h]
    weights = recency_weights(times_h)

    target_h = hours_since_epoch(parse_time(case["target"]))
    x_target = log_time(target_h)

    # --- players-tracked curve: ln(players) against log-time ---
    p_xs, p_ys, p_ws = [], [], []
    last_players = None
    for state, x, w in zip(states, xs, weights):
        players = state.get("players_tracked")
        if players:
            players = float(players)
            last_players = players
            p_xs.append(x)
            p_ys.append(math.log(players))
            p_ws.append(w)
    players_fit = weighted_line(p_xs, p_ys, p_ws, *PLAYERS_SLOPE_RANGE)
    players_slope = players_fit[1] if players_fit is not None else 0.0
    if players_fit is not None:
        players_forecast = math.exp(players_fit[0] + players_fit[1] * x_target)
    elif last_players:
        players_forecast = last_players
    else:
        players_forecast = 1.0
    players_forecast = max(players_forecast, 1.0)

    def players_at(x):
        if players_fit is not None:
            return max(math.exp(players_fit[0] + players_fit[1] * x), 1.0)
        return players_forecast

    days = horizon_days(case)
    base_sigma = 0.5 * math.sqrt(days)
    n_refresh, refresh_vol = refresh_stats(states)
    case_floor = sigma_floor(days, n_refresh)
    w_persist = persist_weight(n_refresh)
    last_rates = last_known_rates(case)
    known_rates = sorted(r for r in last_rates if r is not None)
    fill_rate = known_rates[len(known_rates) // 2] if known_rates else 1.0

    rows = []
    for i, trophy in enumerate(trophies):
        # --- earners curve for this trophy: ln(earners + 0.5) on log-time ---
        e_xs, e_ys, e_ws, e_actual = [], [], [], []
        for state, x, w in zip(states, xs, weights):
            rates = state.get("rates") or []
            players = state.get("players_tracked")
            if i < len(rates) and rates[i] is not None and players:
                earners = float(rates[i]) * float(players) / 100.0
                e_xs.append(x)
                e_ys.append(math.log(earners + 0.5))
                e_ws.append(w)
                e_actual.append(float(rates[i]))

        earners_fit = None
        if len(e_xs) >= 3:
            earners_fit = weighted_line(e_xs, e_ys, e_ws, *EARNERS_SLOPE_RANGE)

        if earners_fit is None:
            # Too little usable history: hold the last rate, spread wide.
            median = last_rates[i] if last_rates[i] is not None else fill_rate
            sigma = max(base_sigma, case_floor)
            rows.append({"id": trophy["id"], "q": quantiles_from(median, sigma, levels)})
            continue

        a_e, b_e = earners_fit
        # Extrapolate earners with the excess-over-players slope damped:
        # rates saturate, so earners growth converges to players growth.
        x_last = max(xs)
        if x_target > x_last:
            extrap_slope = players_slope + EARNERS_TREND_DAMPING * (b_e - players_slope)
            log_earners = a_e + b_e * x_last + extrap_slope * (x_target - x_last)
        else:
            log_earners = a_e + b_e * x_target
        earners_forecast = math.exp(log_earners) - 0.5
        earners_forecast = min(max(earners_forecast, 0.0), players_forecast)
        median = clamp_rate(100.0 * earners_forecast / players_forecast)

        # Anchor the median to the last observed rate, in logit space.
        # On a flat history the last rate carries most of the weight;
        # with refreshes on record the ratio forecast dominates and
        # the anchor only reins in extrapolation overshoot.
        if last_rates[i] is not None:
            blended = (1.0 - w_persist) * logit(median) + w_persist * logit(last_rates[i])
            median = clamp_rate(inv_logit(blended))

        # In-sample logit residuals of the ratio model, for the spread.
        sq = 0.0
        n_resid = 0
        for x, actual in zip(e_xs, e_actual):
            pred_earners = math.exp(a_e + b_e * x) - 0.5
            pred_players = players_at(x)
            pred_earners = min(max(pred_earners, 0.0), pred_players)
            pred_rate = clamp_rate(100.0 * pred_earners / pred_players)
            diff = logit(actual) - logit(pred_rate)
            sq += diff * diff
            n_resid += 1
        resid_std = math.sqrt(sq / n_resid) if n_resid >= 2 else 0.0

        extrap = EXTRAP_SPREAD_PER_LOGTIME * max(0.0, x_target - max(xs))
        vol = VOL_SPREAD_FACTOR * refresh_vol
        sigma = math.sqrt(
            base_sigma ** 2
            + (RESID_SPREAD_FACTOR * resid_std) ** 2
            + extrap ** 2
            + vol ** 2
        )
        sigma = max(sigma, case_floor)
        sigma = min(sigma, SIGMA_CAP)

        rows.append({"id": trophy["id"], "q": quantiles_from(median, sigma, levels)})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            # Never leave a case invalid: fall back to persistence rows.
            levels = case.get("levels") or DEFAULT_LEVELS
            forecast = {"trophies": persistence_rows(case, levels)}
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
