"""Token cost panel for the Open Chat Reviewer web application.

This file is NOT part of the upstream project. It is dropped into the package
so the FastAPI application can serve a live cost report at /token-report.

Because git does not track it, `git pull` leaves it alone; only the two small
edits to api.py and web/src/App.tsx can ever conflict, and the patch script
that made them re-applies cleanly.

Usage figures are extracted out of the raw payloads once into the private
token_report schema, then topped up incrementally on each request. Every
upstream table is read-only here.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import os
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg

DEFAULT_PRICING = {
    "as_at": "2026-09-04",
    "source": "Anthropic published API pricing",
    "display_currency": "AUD",
    "rate_per_usd": 1.395,
    "prices_usd_per_mtok": {
        "claude-opus-5":   {"inp": 5.0,  "out": 25.0, "cw5m": 6.25,  "cw1h": 10.0, "read": 0.50},
        "claude-opus-4-8": {"inp": 5.0,  "out": 25.0, "cw5m": 6.25,  "cw1h": 10.0, "read": 0.50},
        "claude-sonnet-5": {"inp": 2.0,  "out": 10.0, "cw5m": 2.50,  "cw1h": 4.00, "read": 0.20},
        "claude-fable-5":  {"inp": 10.0, "out": 50.0, "cw5m": 12.50, "cw1h": 20.0, "read": 1.00},
    },
}

# Operators edit this file, not the code. Missing or malformed falls back to the
# defaults above and the page says so rather than failing.
PRICING_PATH = Path(__file__).resolve().parents[2] / ".chatreview" / "token-pricing.json"
STALE_AFTER_DAYS = 90


def load_pricing() -> dict:
    """Read the pricing file, falling back to the built in defaults."""
    config = dict(DEFAULT_PRICING)
    config["loaded_from"] = "built in defaults"
    try:
        if PRICING_PATH.is_file():
            raw = json.loads(PRICING_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                merged = dict(DEFAULT_PRICING)
                merged.update({k: v for k, v in raw.items() if v is not None})
                merged["loaded_from"] = str(PRICING_PATH)
                config = merged
    except (OSError, ValueError) as exc:
        config["load_error"] = f"{type(exc).__name__}: {exc}"
    return config


def pricing_age_days(config: dict) -> int | None:
    try:
        stamped = dt.date.fromisoformat(str(config.get("as_at", "")))
    except ValueError:
        return None
    return (dt.datetime.now(ZoneInfo(resolve_timezone())).date() - stamped).days


def resolve_timezone() -> str:
    """Match the project's own resolution order."""
    name = (os.environ.get("CHATREVIEW_TIMEZONE") or os.environ.get("TZ") or "UTC").strip()
    try:
        ZoneInfo(name)
    except Exception:  # noqa: BLE001 - an unknown zone must not break the report
        return "UTC"
    return name


PRICES = {m: v for m, v in DEFAULT_PRICING["prices_usd_per_mtok"].items()}
AUD_PER_USD = float(DEFAULT_PRICING["rate_per_usd"])
TZ = "UTC"

CAT_LIGHT = ["#0B5299", "#C43F0B", "#6B4E9E", "#15803D"]

CAT_DARK  = ["#3B87C9", "#CE5B2E", "#8C74C0", "#2DA55E"]

SERIES = ["Cache read", "Cache write", "Output", "Input"]

def split_cache_write(r: dict) -> tuple[int, int]:
    """Return (5m, 1h) cache-write tokens, allocating any unlabelled remainder to 5m."""
    five, hour, total = int(r["cw5m"] or 0), int(r["cw1h"] or 0), int(r["cw_total"] or 0)
    if five + hour == 0 and total:
        return total, 0
    gap = total - (five + hour)
    return five + max(gap, 0), hour

def usd_cost(r: dict) -> dict[str, float]:
    p = PRICES.get(r["model"])
    if not p:
        return {"Input": 0.0, "Output": 0.0, "Cache write": 0.0, "Cache read": 0.0}
    five, hour = split_cache_write(r)
    return {
        "Input":       int(r["inp"] or 0)  / 1e6 * p["inp"],
        "Output":      int(r["out"] or 0)  / 1e6 * p["out"],
        "Cache write": five / 1e6 * p["cw5m"] + hour / 1e6 * p["cw1h"],
        "Cache read":  int(r["read"] or 0) / 1e6 * p["read"],
    }

def aud(x: float) -> float:
    return x * AUD_PER_USD

def money(x: float) -> str:
    if x >= 1000:
        return f"${x:,.0f}"
    if x >= 10:
        return f"${x:,.2f}"
    return f"${x:,.2f}"

def compact(n: float) -> str:
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= div:
            return f"{n / div:,.2f}{suf}"
    return f"{n:,.0f}"

def esc(s) -> str:
    return html.escape(str(s), quote=True)

def bars_vertical(pairs, width=760, height=190, pad_l=8, label_every=7):
    """pairs: [(date, value)] -> single-hue vertical bars, 4px rounded tops."""
    if not pairs:
        return "<p class='muted'>No data.</p>"
    top = max(v for _, v in pairs) or 1
    n = len(pairs)
    gap = 2
    bw = max(2.0, (width - pad_l * 2 - gap * (n - 1)) / n)
    plot_h = height - 30
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" preserveAspectRatio="none">']
    for i in range(1, 4):
        y = plot_h - plot_h * i / 4
        out.append(f'<line x1="{pad_l}" x2="{width - pad_l}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
    for i, (d, v) in enumerate(pairs):
        x = pad_l + i * (bw + gap)
        h = max(2.0, plot_h * (v / top))
        y = plot_h - h
        out.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3" class="bar">'
            f'<title>{esc(d)} — {money(v)} AUD</title></rect>'
        )
        last_tick = ((n - 1) // label_every) * label_every
        show = (i % label_every == 0 and i != last_tick) or i == n - 1
        if n - 1 - last_tick >= label_every // 2:
            show = show or i == last_tick
        if show:
            out.append(
                f'<text x="{x + bw / 2:.1f}" y="{plot_h + 15:.0f}" class="tick" '
                f'text-anchor="middle">{esc(str(d)[5:])}</text>'
            )
    out.append(f'<line x1="{pad_l}" x2="{width - pad_l}" y1="{plot_h}" y2="{plot_h}" class="axis"/>')
    out.append("</svg>")
    return "".join(out)

def bars_horizontal(pairs, colors=None, width=760, row_h=30, label_w=190):
    """pairs: [(label, value)] -> horizontal bars with direct value labels."""
    if not pairs:
        return "<p class='muted'>No data.</p>"
    top = max(v for _, v in pairs) or 1
    height = row_h * len(pairs)
    track = width - label_w - 108
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    for i, (label, v) in enumerate(pairs):
        y = i * row_h
        bw = max(2.0, track * (v / top))
        fill = (f' data-ci="{i}" style="fill:{colors[i % len(colors)]}"') if colors else ""
        cls = "bar"
        out.append(
            f'<text x="0" y="{y + row_h / 2 + 4:.0f}" class="rowlabel">{esc(label[:34])}</text>'
            f'<rect x="{label_w}" y="{y + 5:.0f}" width="{bw:.1f}" height="{row_h - 12:.0f}" '
            f'rx="3" class="{cls}"{fill}><title>{esc(label)} — {money(v)} AUD</title></rect>'
            f'<text x="{label_w + bw + 9:.1f}" y="{y + row_h / 2 + 4:.0f}" class="rowvalue">{money(v)}</text>'
        )
    out.append("</svg>")
    return "".join(out)

def table(headers, rows, aligns=None) -> str:
    aligns = aligns or ["left"] + ["right"] * (len(headers) - 1)
    h = "".join(f'<th style="text-align:{a}">{esc(x)}</th>' for x, a in zip(headers, aligns, strict=False))
    body = "".join(
        "<tr>"
        + "".join(f'<td style="text-align:{a}">{esc(c)}</td>' for c, a in zip(r, aligns, strict=False))
        + "</tr>"
        for r in rows
    )
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'

def build(rows: list[dict], pricing: dict | None = None) -> str:
    if not rows:
        raise ValueError("no usage rows")

    unknown = sorted({r["model"] for r in rows if r["model"] not in PRICES})
    by_day, by_model, by_project, by_session, by_kind = (
        defaultdict(float), defaultdict(float), defaultdict(float),
        defaultdict(lambda: [0.0, 0, None, ""]), defaultdict(float),
    )
    tok = defaultdict(int)
    msgs = 0
    day_tokens = defaultdict(int)

    for r in rows:
        c = usd_cost(r)
        total = aud(sum(c.values()))
        d = r["day"]
        by_day[d] += total
        by_model[r["model"]] += total
        by_project[r["project"]] += total
        s = by_session[r["session_ext"]]
        s[0] += total
        s[1] += int(r["messages"])
        s[2] = d if s[2] is None else min(s[2], d)
        s[3] = r["project"]
        for k, v in c.items():
            by_kind[k] += aud(v)
        five, hour = split_cache_write(r)
        tok["Input"] += int(r["inp"] or 0)
        tok["Output"] += int(r["out"] or 0)
        tok["Cache write"] += five + hour
        tok["Cache read"] += int(r["read"] or 0)
        day_tokens[d] += int(r["inp"] or 0) + int(r["out"] or 0) + five + hour + int(r["read"] or 0)
        msgs += int(r["messages"])

    days = sorted(by_day)
    grand = sum(by_day.values())
    total_tokens = sum(tok.values())
    first, last = days[0], days[-1]
    span = (last - first).days + 1
    active = len(days)

    def window(n: int) -> float:
        cut = last - dt.timedelta(days=n - 1)
        return sum(v for d, v in by_day.items() if d >= cut)

    # -------- charts
    daily_series = [(d, by_day[d]) for d in days][-60:]
    chart_daily = bars_vertical(daily_series)
    kind_pairs = [(k, by_kind[k]) for k in ["Cache read", "Cache write", "Output", "Input"]]
    chart_kind = bars_horizontal(kind_pairs, colors=CAT_LIGHT)
    chart_model = bars_horizontal(sorted(by_model.items(), key=lambda kv: -kv[1]))
    top_projects = sorted(by_project.items(), key=lambda kv: -kv[1])[:10]
    chart_project = bars_horizontal(top_projects)

    # -------- tables
    weeks, months = defaultdict(float), defaultdict(float)
    for d, v in by_day.items():
        weeks[d - dt.timedelta(days=d.weekday())] += v
        months[d.strftime("%Y-%m")] += v

    t_month = table(
        ["Month", "Cost AUD", "Share"],
        [[m, money(v), f"{v / grand * 100:.1f}%"] for m, v in sorted(months.items())],
    )
    t_week = table(
        ["Week beginning", "Cost AUD", "Daily average"],
        [[str(w), money(v), money(v / 7)] for w, v in sorted(weeks.items())],
    )
    t_day = table(
        ["Day", "Cost AUD", "Tokens"],
        [[str(d), money(by_day[d]), compact(day_tokens[d])] for d in reversed(days[-21:])],
    )
    t_sess = table(
        ["Session", "Project", "First seen", "Messages", "Cost AUD"],
        [[k[:26], v[3][:24], str(v[2]), f"{v[1]:,}", money(v[0])]
         for k, v in sorted(by_session.items(), key=lambda kv: -kv[1][0])[:15]],
        aligns=["left", "left", "left", "right", "right"],
    )
    t_tok = table(
        ["Token type", "Tokens", "Share of tokens", "Cost AUD", "Share of cost"],
        [[k, compact(tok[k]), f"{tok[k] / total_tokens * 100:.1f}%",
          money(by_kind[k]), f"{by_kind[k] / grand * 100:.1f}%"]
         for k in ["Cache read", "Cache write", "Output", "Input"]],
    )

    legend = "".join(
        f'<span class="lg"><i style="background:{CAT_LIGHT[i]}" data-dark="{CAT_DARK[i]}"></i>{esc(s)}</span>'
        for i, s in enumerate(SERIES)
    )
    pricing = pricing or {}
    notices = []
    if unknown:
        notices.append(
            '<div class="callout warn"><b>Unpriced models.</b> '
            + esc(", ".join(unknown))
            + " carry no price entry, so their tokens are counted but cost nothing. "
            "Add them to the pricing file to include them.</div>"
        )
    age = pricing_age_days(pricing) if pricing else None
    stamped = esc(str(pricing.get("as_at", "unknown")))
    if age is not None and age > STALE_AFTER_DAYS:
        notices.append(
            f'<div class="callout warn"><b>Prices are {age} days old.</b> They were last '
            f"confirmed on {stamped}, and neither the rates nor the exchange rate update "
            "themselves. Check them against current published pricing before quoting these "
            f"figures. Edit <code>{esc(str(PRICING_PATH))}</code>.</div>"
        )
    if pricing.get("load_error"):
        notices.append(
            '<div class="callout warn"><b>The pricing file could not be read</b>, so built in '
            f"defaults were used. {esc(str(pricing['load_error']))}</div>"
        )
    warn = "".join(notices)

    return PAGE.format(
        generated=dt.datetime.now(ZoneInfo(TZ)).strftime("%d %B %Y, %H:%M"),
        rate=f"{AUD_PER_USD:.3f}",
        grand=money(grand), tokens=compact(total_tokens),
        msgs=f"{msgs:,}", perday=money(grand / active),
        first=first, last=last, span=span, active=active,
        w1=money(window(1)), w7=money(window(7)), w30=money(window(30)), w90=money(window(90)),
        chart_daily=chart_daily, chart_kind=chart_kind,
        chart_model=chart_model, chart_project=chart_project,
        legend=legend, warn=warn,
        t_month=t_month, t_week=t_week, t_day=t_day, t_sess=t_sess, t_tok=t_tok,
        cat_light=",".join(CAT_LIGHT), cat_dark=",".join(CAT_DARK),
        as_at=esc(str(pricing.get('as_at', 'an unknown date'))),
        loaded_from=esc(str(pricing.get('loaded_from', 'built in defaults'))),
        tzname=esc(TZ),
    )

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Token Spend</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{{
  --ground:#F6F7F9;--surface:#FFFFFF;--surface-2:#EDF0F4;--inset:#F1F3F7;
  --ink:#171A1F;--ink-2:#565E6B;--ink-3:#868E9B;--rule:#DCE1E8;
  --accent:#0B5299;--warn:#8A5200;--warn-soft:#FBF2E2;--warn-line:#E8D3AB;
  --c1:#0B5299;--c2:#C43F0B;--c3:#6B4E9E;--c4:#15803D;
  --shadow:0 1px 2px rgba(20,26,38,.06),0 6px 20px -12px rgba(20,26,38,.28);
  color-scheme:light;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --ground:#14161B;--surface:#1B1E25;--surface-2:#232830;--inset:#191C22;
  --ink:#E6E9EE;--ink-2:#A2AAB6;--ink-3:#767F8C;--rule:#2E333C;
  --accent:#6BAEEC;--warn:#E3AC63;--warn-soft:#2A2113;--warn-line:#4C3B1E;
  --c1:#3B87C9;--c2:#CE5B2E;--c3:#8C74C0;--c4:#2DA55E;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -14px rgba(0,0,0,.7);
  color-scheme:dark;
}}}}
:root[data-theme="dark"]{{
  --ground:#14161B;--surface:#1B1E25;--surface-2:#232830;--inset:#191C22;
  --ink:#E6E9EE;--ink-2:#A2AAB6;--ink-3:#767F8C;--rule:#2E333C;
  --accent:#6BAEEC;--warn:#E3AC63;--warn-soft:#2A2113;--warn-line:#4C3B1E;
  --c1:#3B87C9;--c2:#CE5B2E;--c3:#8C74C0;--c4:#2DA55E;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -14px rgba(0,0,0,.7);
  color-scheme:dark;
}}
*{{box-sizing:border-box}}
body{{background:var(--ground);color:var(--ink);margin:0;
  font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;font-size:16px;line-height:1.6}}
.wrap{{max-width:880px;margin:0 auto;padding:44px 20px 90px}}
.eyebrow{{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;font-weight:700;font-size:12px;
  letter-spacing:.14em;text-transform:uppercase;color:var(--ink-3);margin:0 0 10px}}
h1{{font-size:clamp(30px,5vw,42px);line-height:1.12;letter-spacing:-.022em;font-weight:600;margin:0 0 12px}}
.standfirst{{font-size:17px;color:var(--ink-2);margin:0;max-width:62ch}}
h2{{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;font-size:12px;font-weight:700;
  letter-spacing:.14em;text-transform:uppercase;color:var(--ink-3);margin:46px 0 16px;
  padding-bottom:10px;border-bottom:1px solid var(--rule)}}
.card{{background:var(--surface);border:1px solid var(--rule);border-radius:12px;
  padding:20px;box-shadow:var(--shadow);margin-bottom:18px}}
.card h3{{margin:0 0 4px;font-size:17px;font-weight:600;letter-spacing:-.012em}}
.card p.sub{{margin:0 0 14px;font-size:14px;color:var(--ink-2)}}
.hero{{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:14px;margin:26px 0 0}}
.tile{{background:var(--surface);border:1px solid var(--rule);border-radius:12px;
  padding:16px 18px;box-shadow:var(--shadow)}}
.tile .k{{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;font-weight:700;font-size:10.5px;
  letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3);margin:0 0 6px}}
.tile .v{{font-size:27px;font-weight:600;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1.1}}
.tile .n{{font-size:13px;color:var(--ink-2);margin-top:4px}}
.tile.big .v{{color:var(--accent)}}
.chart{{width:100%;height:auto;overflow:visible;display:block}}
.bar{{fill:var(--accent)}}
.grid{{stroke:var(--rule);stroke-width:1}}
.axis{{stroke:var(--ink-3);stroke-width:1}}
.tick{{fill:var(--ink-3);font-size:10px;font-family:"IBM Plex Mono",monospace}}
.rowlabel{{fill:var(--ink);font-size:13px;font-family:"IBM Plex Sans",sans-serif}}
.rowvalue{{fill:var(--ink-2);font-size:12.5px;font-family:"IBM Plex Mono",monospace;font-weight:500}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 0;font-size:13px;color:var(--ink-2)}}
.lg{{display:inline-flex;align-items:center;gap:7px}}
.lg i{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.tw{{overflow-x:auto;margin-top:6px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th{{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;font-weight:700;font-size:10.5px;
  letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);padding:0 12px 8px 0;
  border-bottom:1px solid var(--rule);white-space:nowrap}}
td{{padding:9px 12px 9px 0;border-bottom:1px solid var(--rule);color:var(--ink-2);
  font-variant-numeric:tabular-nums;white-space:nowrap}}
td:first-child{{color:var(--ink);font-weight:500}}
tbody tr:last-child td{{border-bottom:0}}
.callout{{border-radius:10px;padding:14px 16px;font-size:14.5px;margin:16px 0 0;
  background:var(--warn-soft);border:1px solid var(--warn-line);color:var(--ink)}}
.callout b{{font-weight:600}}
.muted{{color:var(--ink-3);font-size:14px}}
footer{{margin-top:52px;padding-top:20px;border-top:1px solid var(--rule);
  font-size:13.5px;color:var(--ink-3)}}
footer p{{margin:6px 0}}
@media (prefers-reduced-motion:reduce){{*{{transition:none!important}}}}
</style></head><body data-palette="{cat_light}" data-palette-dark="{cat_dark}">
<div class="wrap">
<p class="eyebrow">Archive analytics &middot; generated {generated}</p>
<h1>What the tokens cost</h1>
<p class="standfirst">Every assistant message in the archive, priced at Anthropic's published API rates and
converted at {rate} AUD to the dollar. Covering {first} to {last}, which is {span} days
with activity on {active}.</p>

<div class="hero">
  <div class="tile big"><p class="k">Total, all time</p>
    <div class="v">{grand}</div><p class="n">AUD equivalent</p></div>
  <div class="tile"><p class="k">Last 7 days</p>
    <div class="v">{w7}</div><p class="n">{w1} in the last day</p></div>
  <div class="tile"><p class="k">Last 30 days</p>
    <div class="v">{w30}</div><p class="n">{w90} over 90</p></div>
  <div class="tile"><p class="k">Per active day</p>
    <div class="v">{perday}</div><p class="n">{msgs} messages total</p></div>
</div>

{warn}

<h2>Where the money goes</h2>
<div class="card">
  <h3>Cost by token type</h3>
  <p class="sub">Cache reads are billed at a tenth of base input, but at {tokens} tokens overall the volume
  more than makes up for the discount. This is the chart that explains the bill.</p>
  {chart_kind}
  <div class="legend">{legend}</div>
</div>
<div class="card">
  <h3>The same split, with numbers</h3>
  {t_tok}
</div>

<h2>Over time</h2>
<div class="card">
  <h3>Daily cost, most recent 60 days</h3>
  <p class="sub">Hover any bar for the exact figure.</p>
  {chart_daily}
</div>
<div class="card"><h3>By month</h3>{t_month}</div>
<div class="card"><h3>By week</h3>{t_week}</div>
<div class="card"><h3>Last 21 days</h3>{t_day}</div>

<h2>By model and project</h2>
<div class="card"><h3>Cost by model</h3>{chart_model}</div>
<div class="card">
  <h3>Cost by project, top 10</h3>
  <p class="sub">Project attribution comes from the session's working directory, so anything run outside a
  recognised project folder lands in "(unattributed)".</p>
  {chart_project}
</div>
<div class="card"><h3>Most expensive sessions</h3>{t_sess}</div>

<footer>
  <p>Prices are per-million-token API rates as published by Anthropic, with cache writes split by
  their 5-minute and 1-hour TTLs and cache reads priced separately. Last confirmed {as_at},
  loaded from {loaded_from}. Timezone {tzname}, from CHATREVIEW_TIMEZONE.</p>
  <p>If your Claude usage sits under a subscription rather than API billing, treat these figures as
  replacement cost, what the same work would have cost at list price, not money you paid.</p>
  <p>Regenerate any time with <code>uv run python ~/token-report.py</code> after a sync.</p>
</footer>
</div>
<script>
(function(){{
  function paint(){{
    var dark = matchMedia('(prefers-color-scheme: dark)').matches;
    var root = document.documentElement.getAttribute('data-theme');
    if(root === 'dark') dark = true;
    if(root === 'light') dark = false;
    var pal = (dark ? document.body.dataset.paletteDark : document.body.dataset.palette).split(',');
    document.querySelectorAll('.legend .lg i').forEach(function(el, i){{
      el.style.background = pal[i % pal.length];
    }});
    document.querySelectorAll('svg rect[style]').forEach(function(el){{
      var i = el.getAttribute('data-ci');
      if(i !== null) el.style.fill = pal[+i % pal.length];
    }});
  }}
  try{{ paint(); matchMedia('(prefers-color-scheme: dark)').addEventListener('change', paint); }}catch(e){{}}
}})();
</script>
</body></html>
"""


DDL = """
CREATE SCHEMA IF NOT EXISTS token_report;
CREATE TABLE IF NOT EXISTS token_report.usage (
    event_id  bigint PRIMARY KEY,
    model     text   NOT NULL,
    inp       bigint NOT NULL DEFAULT 0,
    out       bigint NOT NULL DEFAULT 0,
    cw5m      bigint NOT NULL DEFAULT 0,
    cw1h      bigint NOT NULL DEFAULT 0,
    cw_total  bigint NOT NULL DEFAULT 0,
    read      bigint NOT NULL DEFAULT 0
);
"""

BACKFILL = """
INSERT INTO token_report.usage (event_id, model, inp, out, cw5m, cw1h, cw_total, read)
SELECT e.id,
       coalesce(j #>> '{message,model}', '(unknown)'),
       coalesce((j #>> '{message,usage,input_tokens}')::bigint, 0),
       coalesce((j #>> '{message,usage,output_tokens}')::bigint, 0),
       coalesce((j #>> '{message,usage,cache_creation,ephemeral_5m_input_tokens}')::bigint, 0),
       coalesce((j #>> '{message,usage,cache_creation,ephemeral_1h_input_tokens}')::bigint, 0),
       coalesce((j #>> '{message,usage,cache_creation_input_tokens}')::bigint, 0),
       coalesce((j #>> '{message,usage,cache_read_input_tokens}')::bigint, 0)
FROM (
    SELECT e.id, convert_from(rp.payload, 'UTF8')::jsonb AS j
    FROM events e
    JOIN raw_records  rr ON rr.id = e.raw_record_id
    JOIN raw_payloads rp ON rp.payload_hash = rr.payload_hash
    WHERE e.id > %s
      AND e.role = 'assistant'
      AND pg_input_is_valid(convert_from(rp.payload, 'UTF8'), 'jsonb')
) AS e
WHERE j #> '{message,usage}' IS NOT NULL
ON CONFLICT (event_id) DO NOTHING
"""

AGGREGATE = """
SELECT (ev.timestamp AT TIME ZONE %s)::date               AS day,
       u.model                                            AS model,
       coalesce(nullif(s.project, ''), '(unattributed)')   AS project,
       coalesce(s.external_id, '(none)')                   AS session_ext,
       count(*)        AS messages,
       sum(u.inp)      AS inp,
       sum(u.out)      AS out,
       sum(u.cw5m)     AS cw5m,
       sum(u.cw1h)     AS cw1h,
       sum(u.cw_total) AS cw_total,
       sum(u.read)     AS read
FROM token_report.usage u
JOIN events ev ON ev.id = u.event_id
LEFT JOIN sessions s ON s.id = ev.session_id
WHERE ev.canonical_event_id IS NULL
  AND ev.timestamp IS NOT NULL
GROUP BY 1, 2, 3, 4
ORDER BY 1
"""


def render_html(database_url: str) -> str:
    """Top up the usage table, then render the report. Safe to call per request."""
    global PRICES, AUD_PER_USD, TZ
    pricing = load_pricing()
    PRICES = dict(pricing.get("prices_usd_per_mtok") or DEFAULT_PRICING["prices_usd_per_mtok"])
    AUD_PER_USD = float(pricing.get("rate_per_usd") or DEFAULT_PRICING["rate_per_usd"])
    TZ = resolve_timezone()

    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        cur.execute("SELECT coalesce(max(event_id), 0) FROM token_report.usage")
        watermark = cur.fetchone()[0]
        cur.execute(BACKFILL, (watermark,))
        conn.commit()
        cur.execute(AGGREGATE, (TZ,))
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
    if not rows:
        return (
            "<!doctype html><meta charset=utf-8><title>Token Spend</title>"
            "<body style='font:16px/1.6 system-ui;padding:44px;max-width:60ch'>"
            "<h1>No usage data yet</h1><p>The archive holds no assistant messages "
            "carrying token counts. Run a sync, then reload.</p></body>"
        )
    return build(rows, pricing)
