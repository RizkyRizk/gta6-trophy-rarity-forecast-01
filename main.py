"""gta6-trophy-rarity-forecast-01's entry in Python 3.13 (standard library only).

Forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, five
quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase at the case's
target, from the house's reads of the trophy page before the case's checkpoint.

Method (why it beats the house baselines, which are persistence and a flat
logit-linear trend):

Exophase refreshes a game's counts on its own schedule, not with every
player's trophy: between refreshes the page is frozen, so consecutive
identical reads are one observation, not many. At a refresh the tracked
player count jumps and every rate but the universal ones drops, because the
newly tracked players have not earned the trophies yet. The program therefore

1. dedupes the history into distinct page states (one observation per
   refresh), so a frozen page cannot fake precision;
2. fits, per trophy, a weighted least squares line of logit(rate) against
   ln(1 + hours since release), with weights from the binomial precision of
   each read plus a process-variance floor (a read of 4 earners out of 638
   players is not worth the same as a read of 4,000 out of 100,000); a fit
   whose points span too little log-time, or whose slope is absurd, counts
   as unidentified and stays out of the pooled slope;
3. shrinks each trophy's slope toward the pooled cross-trophy slope
   (empirical Bayes: a trophy whose own history is flat or noisy borrows
   the average drift, instead of extrapolating its own noise);
4. damps the slope as the target moves past the observed window, blends the
   extrapolation part-way back toward the last read, and caps the implied
   week-one rate by tier (platinum 5%, gold 25%, silver 60%, bronze 99.99%),
   so a fast riser cannot run away to 99% and a falling rare trophy cannot
   be extrapolated to the floor;
5. spreads the quantiles with a total logit sigma that adds the random-walk
   term (0.5 per root day, as the house baseline), the fit's prediction
   error, a refresh-shock term estimated from the jumps the history actually
   saw, and the binomial noise of the answering read; the spread is floored
   (never overconfident on a flat history) and capped.

A trophy stuck at 100% (the intro mission every tracked player completed,
which survived the launch-night refresh unchanged) is forecast tight near
100%: it is structural, not noise. A trophy nobody has earned yet gets a
tier fallback median with a wide spread.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so a module shipped beside main.py
imports only after the sys.path line below, which puts the directory first;
-s drops user site-packages.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a
    time, with only that case's file there): JSON
    {"schema": "makefun/gta6-trophies-case/v1", "id", "at",
    "horizon_minutes", "target", "levels", "game", "trophies", "history",
    "notes"}, where `trophies` lists the case's trophies {id, name,
    description, tier, points, secret} and `history` every read before `at`,
    oldest first, each {slot, at, players_tracked, trophies_earned, rates},
    `rates` in the order of `trophies` (null where that read lacked the
    trophy);
  - write <id>.out to the output directory:
    {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}, one row
    for each of the case's trophies, each id once, no other key, each q five
    finite numbers from 0 to 100 (percent), non-decreasing; at most 65,536
    bytes. Anything else is an invalid case, scored as if every quantile
    equalled the trophy's rate in the last read, and an entry with an
    invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Grand Theft Auto VI released 2026-11-19T00:00:00Z (UTC day of release);
# the house's clock counts hours from there.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Random-walk spread of a rate's move, in logit units per square-root day
# (the house baseline's sigma).
RW_SIGMA = 0.5
# Process variance floor of one read's logit rate, on top of its binomial
# variance: the page moves by refreshes, not only by sampling.
PROCESS_VAR = 0.09
# Damping length, in ln(1 + hours), for extrapolating past the observed
# window: the further the target sits past the last read, the more the
# fitted slope is pulled toward flat.
DAMP_LEN = 4.0
# Weight on the extrapolated trend (vs persistence at the last read) when
# blending the forecast median: more history behind the fit, more weight.
W_TREND_HIGH = 0.9
W_TREND_LOW = 0.8
# Week-one ceiling on the forecast median, by tier: a fast riser cannot run
# away past what its tier plausibly reaches in a week.
TIER_CAP = {"platinum": 5.0, "gold": 25.0, "silver": 60.0, "bronze": 99.99}
# Fallback median (percent) when a trophy has no usable read at all.
TIER_FALLBACK = {"platinum": 0.5, "gold": 2.0, "silver": 5.0, "bronze": 12.0}
# Fallback median (percent) when a trophy's observed rate is zero everywhere.
TIER_ZERO = {"platinum": 0.3, "gold": 1.0, "silver": 3.0, "bronze": 5.0}
# A forecast median never falls below this share of the last read: a
# declining trend is not extrapolated to the floor.
MIN_RATIO_TO_LAST = 0.2
# Total logit spread is floored (never overconfident) and capped.
SIGMA_MIN = 0.45
SIGMA_MAX = 2.2
# Overall scale on the total spread (1.0 = as computed).
SIGMA_SCALE = 1.0


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime (accepts a trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    """An earn rate in percent, clamped to the range the logit is defined on."""
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(y):
    if y >= 0:
        return 1.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it. A whole number comes back as an
    int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation, good to
    ~1e-9); the five scored levels could also come from a table, but this
    keeps the program correct for any levels a case asks for."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"level {p} outside (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def dedupe_states(history):
    """Collapse the history into distinct page states: Exophase freezes the
    page between refreshes, so a run of identical reads is one observation.
    Returns [(slot_time, players_tracked, rates)] in order, keeping the
    first read of each run."""
    states = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        try:
            when = parse_time(read["at"])
        except (KeyError, ValueError):
            when = parse_time(read["slot"])
        states.append((when, read.get("players_tracked"), rates))
    return states


def hours_since_release(when):
    return max(0.0, (when - RELEASE).total_seconds() / 3600.0)


def fit_trophy(obs):
    """Weighted least squares of logit(rate) on ln(1 + hours since release).

    obs: [(x, y, var)] with var the read's logit variance. Returns
    (slope, slope_se, s2, xw, yw, sxx, w_sum, n) or None when the slope is
    unidentified (fewer than 2 points, or the points span too little
    log-time for a trend to mean anything: two states ten minutes apart
    straddling the release give a slope of tens of thousands from
    numerical noise, not information)."""
    n = len(obs)
    if n < 2:
        return None
    xs = [x for x, _, _ in obs]
    if max(xs) - min(xs) < 0.05:
        return None
    w_sum = sum(1.0 / v for _, _, v in obs)
    xw = sum(x / v for x, _, v in obs) / w_sum
    yw = sum(y / v for _, y, v in obs) / w_sum
    sxx = sum((x - xw) ** 2 / v for x, _, v in obs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - xw) * (y - yw) / v for x, y, v in obs)
    slope = sxy / sxx
    intercept = yw - slope * xw
    if n > 2:
        rss = sum((y - (intercept + slope * x)) ** 2 / v for x, y, v in obs)
        s2_raw = rss / (n - 2)
    else:
        s2_raw = 0.0
    # The residual variance is floored at the binomial noise level: with a
    # frozen page the residuals can be zero, which would claim a slope is
    # known exactly when it is not identified at all.
    mean_var = sum(v for _, _, v in obs) / n
    s2 = max(s2_raw, 1.0 if mean_var <= 1.0 else mean_var)
    # s2 is on the weighted scale (weights are 1/var): a residual variance
    # of 1 means the reads scatter as their binomial variance says. Rescale
    # to logit units with the mean variance so SEs are in logit units.
    s2_logit = s2 * mean_var
    slope_se = math.sqrt(s2_logit / sxx) if sxx > 0 else float("inf")
    return {
        "slope": slope, "slope_se": slope_se, "s2_logit": s2_logit,
        "xw": xw, "yw": yw, "sxx": sxx, "w_sum": w_sum, "n": n,
    }


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    z = [norm_ppf(p) for p in levels]
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target))

    states = dedupe_states(history)
    n_states = len(states)

    # --- First pass: per-trophy fits, for the pooled slope. ---
    per_trophy = []  # one dict per trophy: obs, fit, last_rate, last_earned
    for i, trophy in enumerate(trophies):
        obs = []
        last_rate = None
        last_earned = 0.0
        last_n = None
        for when, players, rates in states:
            if i >= len(rates) or rates[i] is None:
                continue
            rate = float(rates[i])
            last_rate = rate
            last_n = players
            earned = rate * (players or 0) / 100.0
            last_earned = earned
            # Binomial variance of the logit, plus the process floor.
            var = PROCESS_VAR + 1.0 / max(earned, 0.5)
            obs.append((math.log1p(hours_since_release(when)),
                        logit(rate), var))
        per_trophy.append({
            "trophy": trophy, "obs": obs, "fit": fit_trophy(obs),
            "last_rate": last_rate, "last_earned": last_earned,
            "last_n": last_n,
        })

    # Pooled slope across trophies with an identified fit (empirical Bayes
    # prior mean), and the cross-trophy dispersion of slopes around it.
    # Fits with an absurd slope or a huge standard error are unidentified
    # noise, not signal: they are left out of the pool so one degenerate
    # trophy cannot drag every other trophy's slope with it.
    fits = [t["fit"] for t in per_trophy
            if t["fit"] is not None
            and abs(t["fit"]["slope"]) <= 5.0 and t["fit"]["slope_se"] <= 5.0]
    if len(fits) >= 3:
        prec = [1.0 / (f["slope_se"] ** 2) for f in fits]
        pool = sum(p * f["slope"] for p, f in zip(prec, fits)) / sum(prec)
        mean_se2 = sum(f["slope_se"] ** 2 for f in fits) / len(fits)
        var_between = sum(p * (f["slope"] - pool) ** 2
                          for p, f in zip(prec, fits)) / sum(prec)
        tau2 = max(0.0, var_between - mean_se2)
    else:
        pool, tau2 = 0.0, 0.25

    # Refresh rate: distinct page states per day of history, for the
    # shock term. A refresh is a state change; the states themselves are
    # the refreshes the history saw.
    if n_states >= 2:
        span_days = max(0.25, (states[-1][0] - states[0][0]).total_seconds() / 86400.0)
        refresh_per_day = (n_states - 1) / span_days
    else:
        refresh_per_day = 0.0

    rows = []
    for entry in per_trophy:
        trophy = entry["trophy"]
        tier = trophy.get("tier") or "bronze"
        obs = entry["obs"]
        fit = entry["fit"]
        last_rate = entry["last_rate"]

        # --- No usable read at all: tier fallback, wide spread. ---
        if last_rate is None:
            med = TIER_FALLBACK.get(tier, 10.0)
            sigma = 2.0
            rows.append({"id": trophy["id"], "q": quantiles(med, sigma, z)})
            continue

        # --- A trophy every tracked player already holds: structural 100%.
        # It survived the launch-night refresh unchanged while every other
        # rate fell, so forecast it tight near the top, with a small hedge
        # below in case the page corrects it later.
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"],
                         "q": [99.9, 99.97, 99.99, 99.99, 99.99]})
            continue

        # --- A trophy nobody has earned yet: tier fallback, wide spread. ---
        if all(o[1] <= logit(0.01) + 1e-9 for o in obs) and last_rate <= 0.01:
            med = TIER_ZERO.get(tier, 5.0)
            sigma = 1.8
            rows.append({"id": trophy["id"], "q": quantiles(med, sigma, z)})
            continue

        y_last = logit(last_rate)
        x_max = max(o[0] for o in obs) if obs else 0.0

        if fit is None:
            # One distinct state (a frozen page): no trend is identified.
            # Persist the last rate; the spread carries the unidentified
            # trend instead of pretending the flat line is exact.
            med_logit = y_last
            sigma_fit = 0.8
            sigma = math.sqrt((RW_SIGMA ** 2) * horizon_days
                              + sigma_fit ** 2
                              + 1.0 / max(entry["last_earned"], 1.0))
        else:
            # Shrink the slope toward the pooled slope, and keep it in a
            # sane range: a slope past 5 logit per log-hour is not a trend.
            se2 = fit["slope_se"] ** 2
            if tau2 > 0.0:
                slope = (fit["slope"] / se2 + pool / tau2) / (1.0 / se2 + 1.0 / tau2)
            else:
                slope = pool
            slope = max(-5.0, min(5.0, slope))
            # Damp the slope past the observed window.
            gap = max(0.0, x_target - x_max)
            slope *= 1.0 / (1.0 + gap / DAMP_LEN)
            # Recentre on the weighted mean, then extrapolate.
            y_trend = fit["yw"] + slope * (x_target - fit["xw"])
            # Blend back toward persistence: the further the target, the
            # less a trend fitted on two days of data is trusted outright.
            w_trend = W_TREND_HIGH if fit["n"] >= 4 else W_TREND_LOW
            med_logit = w_trend * y_trend + (1.0 - w_trend) * y_last
            # Prediction error of the fit at the target, in logit units.
            se_pred = math.sqrt(fit["s2_logit"] * (
                1.0 / fit["w_sum"] + (x_target - fit["xw"]) ** 2 / fit["sxx"]))
            # Refresh shocks still to come: each future refresh can move a
            # rate off its trend; the history's own refresh jumps size it.
            n_ref = min(4.0, max(0.0, refresh_per_day * horizon_days))
            shock = 0.4 * math.sqrt(n_ref) * 0.5
            sigma = math.sqrt((RW_SIGMA ** 2) * horizon_days
                              + se_pred ** 2 + shock ** 2
                              + 1.0 / max(entry["last_earned"], 1.0))

        # Median in percent, with the tier ceiling and the floor against a
        # runaway declining extrapolation.
        med = 100.0 * sigmoid(med_logit)
        med = min(med, TIER_CAP.get(tier, 99.99))
        if last_rate > 0:
            med = max(med, MIN_RATIO_TO_LAST * last_rate)
        med = clamp_pct(med)
        sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma)) * SIGMA_SCALE
        rows.append({"id": trophy["id"],
                     "q": quantiles(med, sigma, z)})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def quantiles(median_pct, sigma, z):
    """Five quantiles around a median percent, on the logit scale."""
    centre = logit(median_pct)
    out = []
    for zi in z:
        q = 100.0 * sigmoid(centre + zi * sigma)
        out.append(round4(clamp_pct(q)))
    # Rounding can only break monotonicity by a hair; enforce it.
    for k in range(1, len(out)):
        if out[k] < out[k - 1]:
            out[k] = out[k - 1]
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
