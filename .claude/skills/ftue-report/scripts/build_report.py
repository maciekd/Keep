#!/usr/bin/env python3
"""Assemble the FTUE report: the written analysis plus a chat transcript and appendix rendered
from the events.

usage: python3 -I build_report.py --events <dir>/events.json --analysis analysis.html \
           --tester "Alice Carter" --out ~/Desktop/Alice_Carter_FTUE_report.html [--assistant Michelle] [--user Alice]

--analysis is an HTML fragment you wrote (summary, timeline, findings; see SKILL.md for the
sections). --user is the name the tester gave the assistant (defaults to the tester's first
name); --assistant is detected from the first reply when omitted. Six-digit codes typed as a
whole message are masked, since they are 2FA codes.
"""
import argparse
import html
import json
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Stockholm")
TAG = re.compile(r"<[^>]+>")
HERE = os.path.dirname(os.path.abspath(__file__))


def esc(s):
    return html.escape(str(s), quote=False)


def is_ftue_prompt(e):
    return e["ev"] == "user_turn" and e["p"].get("query", "").startswith("The server has just started")


def rich(s, secrets):
    """Escape the reply, then re-enable the few tags the assistant writes (b, i, code, a)."""
    for sec in secrets:
        s = s.replace(sec, "••••••")
    out = html.escape(s, quote=False)
    out = re.sub(r"&lt;(/?)(b|i|code|pre)&gt;", r"<\1\2>", out)
    out = re.sub(r'&lt;a href="([^"]*)"&gt;', r'<a href="\1" target="_blank" rel="noopener">', out)
    out = out.replace("&lt;/a&gt;", "</a>").replace("&amp;amp;", "&amp;")
    return balanced(out).replace("\n", "<br>")


def balanced(out):
    """A reply can carry a tag it never closes (a command cut out of the text takes the closer
    with it), and an open <code> would style the rest of the page. None of these tags nests in
    itself, so an opener still open when the same tag opens again, when an outer tag closes or
    when the reply ends is dropped, and so is a closer that closes nothing."""
    parts, open_tags, last = [], [], 0      # open_tags: (tag, the index of its opener in parts)
    for m in re.finditer(r"<(/?)(b|i|code|pre|a)\b[^>]*>", out):
        parts.append(out[last:m.start()])
        last = m.end()
        closing, tag = m.group(1), m.group(2)
        names = [name for name, _ in open_tags]
        if tag in names:
            while open_tags[-1][0] != tag:
                parts[open_tags.pop()[1]] = ""
            if closing:
                open_tags.pop()
                parts.append(m.group(0))
                continue
            parts[open_tags.pop()[1]] = ""
        if not closing:
            open_tags.append((tag, len(parts)))
            parts.append(m.group(0))
    parts.append(out[last:])
    for _, opener in open_tags:
        parts[opener] = ""
    return "".join(parts)


def gap_text(delta):
    m = int(delta.total_seconds() // 60)
    if m < 60:
        return f"{m} min later"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h} h {m:02d} min later"
    dd, h = divmod(h, 24)
    return f"{dd} day{'s' if dd > 1 else ''} {h} h later"


def load(path):
    d = json.load(open(path))
    events = []
    for uuid, ev, ts, props in d["rows"]:
        p = json.loads(props) if isinstance(props, str) else props
        events.append({"ev": ev, "t": datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(TZ), "p": p})
    return events


def transcript(events, user, assistant):
    hm = lambda t: t.strftime("%H:%M")
    secrets = [e["p"]["query"].strip() for e in events if e["ev"] == "user_turn" and re.fullmatch(r"\d{6}", e["p"].get("query", "").strip())]
    parts, tab_buf = [], []
    last_t = last_day = None

    def flush_tabs():
        if not tab_buf:
            return
        first, last = tab_buf[0]["t"], tab_buf[-1]["t"]
        # dwell is the time spent on the tab *left*; the first entry's dwell is the idle time before browsing began
        longest = max(tab_buf[1:] or tab_buf, key=lambda e: e["p"].get("dwell", 0))
        items = "".join(f'<li><span class="tabname">{esc(e["p"]["tab"])}</span><span class="dwell">left after {e["p"]["dwell"]:.0f}s on {esc(e["p"]["from"])}</span></li>' for e in tab_buf)
        parts.append(f'<div class="evt tabs"><div class="evt-head"><span class="time">{hm(first)}–{hm(last)}</span> <b>{esc(user)} browses the console tabs</b> — {len(tab_buf)} switches in {int((last - first).total_seconds())} s; longest stay {longest["p"]["dwell"]:.0f} s on <b>{esc(longest["p"]["from"])}</b>.</div><ul class="tablist">{items}</ul></div>')
        tab_buf.clear()

    i = 0
    while i < len(events):
        e = events[i]; t, ev, p = e["t"], e["ev"], e["p"]
        if ev == "status" or (ev == "heartbeat" and not p.get("startup")) or ev == "memory_size":
            i += 1; continue
        if ev != "tab_change":
            flush_tabs()
        if last_day != t.date():
            parts.append(f'<div class="daybreak">{t.strftime("%A %d %B %Y")}</div>'); last_day = t.date()
        elif last_t and t - last_t > timedelta(minutes=20):
            parts.append(f'<div class="gap">· · · {gap_text(t - last_t)} · · ·</div>')
        if ev == "heartbeat":
            nxt = events[i + 1] if i + 1 < len(events) else None
            prompt_html = ""
            if nxt and is_ftue_prompt(nxt) and nxt["t"] - t < timedelta(minutes=2):
                prompt_html = f'<details class="prompt"><summary>FTUE prompt injected as the first user turn ({hm(nxt["t"])}) — click to read</summary><pre>{esc(nxt["p"]["query"])}</pre></details>'
                i += 1
            kind = "Server restarted" if last_t else "Server started"
            parts.append(f'<div class="evt restart"><div class="evt-head"><span class="time">{hm(t)}</span> <b>{kind}</b> · Keep {esc(p.get("version", ""))} · {esc(p.get("os", ""))} {esc(p.get("os_version", ""))} {esc(p.get("machine", ""))} · integrations on: {", ".join(p.get("integrations") or ["none"])}</div>{prompt_html}</div>')
        elif ev == "user_turn":
            if is_ftue_prompt(e):
                parts.append(f'<div class="evt restart"><div class="evt-head"><span class="time">{hm(t)}</span> <b>FTUE prompt injected</b> (no start heartbeat beside it)</div><details class="prompt"><summary>Read the prompt</summary><pre>{esc(p["query"])}</pre></details></div>')
            else:
                q = p.get("query", ""); note = ""
                if q.strip() in secrets:
                    q = "••••••"; note = f'<div class="note">{esc(user)} typed a six-digit code (an Apple 2FA code) into the chat; masked here.</div>'
                parts.append(f'<div class="msg user"><div class="who">{esc(user)} <span class="time">{hm(t)}</span> <span class="src">{esc(p.get("source", ""))}</span></div><div class="bubble">{esc(q)}</div>{note}</div>')
        elif ev == "assistant_turn":
            cmds = p.get("commands") or []
            cmd_html = f'<div class="cmds">ran: {"".join(f"<span class=chip>{esc(c)}</span>" for c in cmds)}</div>' if cmds else ""
            pro = p.get("proactive")
            label = f"{esc(assistant)} (assistant)" + (" · proactive nudge" if pro else "")
            parts.append(f'<div class="msg assistant{" proactive" if pro else ""}"><div class="who">{label} <span class="time">{hm(t)}</span> <span class="src">{esc(p.get("source", ""))}</span></div>{cmd_html}<div class="bubble">{rich(p.get("reply", ""), secrets)}</div></div>')
        elif ev in ("integration_enabled", "integration_disabled"):
            parts.append(f'<div class="evt integ"><div class="evt-head"><span class="time">{hm(t)}</span> <b>Integration {ev.split("_")[1]}: {esc(p.get("integration", ""))}</b> · state after restart: <code>{esc(p.get("state"))}</code></div></div>')
        elif ev == "tab_change":
            tab_buf.append(e)
        last_t = t; i += 1
    flush_tabs()
    return "\n".join(parts)


def appendix(events):
    counts = {}
    for e in events:
        counts[e["ev"]] = counts.get(e["ev"], 0) + 1
    count_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    hb_rows = "".join(f'<tr><td>{e["t"]:%d %b %H:%M}</td><td>{"start" if e["p"].get("startup") else "daily"}</td><td>{esc(e["p"].get("version", ""))}</td><td>{", ".join(e["p"].get("integrations") or ["—"])}</td></tr>' for e in events if e["ev"] == "heartbeat")
    st_rows = []
    for e in events:
        if e["ev"] == "status":
            p, secs = e["p"], e["p"].get("prompt_sections", {})
            st_rows.append(f'<tr><td>{e["t"]:%d %b %H:%M}</td><td>{", ".join(p.get("integrations", []))}</td><td>{esc(json.dumps(p.get("feeds")))}</td><td>{p.get("prompt_kb", "")}</td><td>{secs.get("history", "")}</td><td>{secs.get("memory", "")}</td></tr>')
    mem_rows = "".join(f'<tr><td>{e["t"]:%d %b %H:%M}</td><td>{e["p"].get("words")}</td><td>{e["p"].get("watch_items")}</td><td>{e["p"].get("procedures")}</td></tr>' for e in events if e["ev"] == "memory_size")
    mem_table = f'<div class="card"><h3 style="margin-top:0">Memory curations</h3><table><tr><th>Time</th><th>Words</th><th>Watch items</th><th>Procedures</th></tr>{mem_rows}</table></div>' if mem_rows else '<div class="card"><h3 style="margin-top:0">Memory curations</h3><p style="margin:0;color:var(--muted)">No memory_size event: the long-term memory was never curated.</p></div>'
    return count_rows, hb_rows, "".join(st_rows), mem_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True); ap.add_argument("--analysis", required=True)
    ap.add_argument("--tester", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--user"); ap.add_argument("--assistant")
    a = ap.parse_args()
    events = load(a.events)
    distinct_id = os.path.basename(os.path.dirname(os.path.abspath(a.events)))
    user = a.user or a.tester.split()[0]
    assistant = a.assistant
    if not assistant:
        first = next((e for e in events if e["ev"] == "assistant_turn"), None)
        m = first and re.search(r"I'm <b>([^<]+)</b>", first["p"].get("reply", ""))
        assistant = m.group(1) if m else "Assistant"
    count_rows, hb_rows, st_rows, mem_table = appendix(events)
    tpl = open(os.path.join(HERE, "..", "assets", "template.html")).read()
    out = (tpl.replace("{{TESTER}}", esc(a.tester)).replace("{{USER}}", esc(user)).replace("{{ASSISTANT}}", esc(assistant))
           .replace("{{DISTINCT_ID}}", esc(distinct_id))
           .replace("{{FIRST}}", f"{events[0]['t']:%a %d %b %Y, %H:%M}").replace("{{LAST_EV}}", f"{events[-1]['t']:%a %d %b, %H:%M}")
           .replace("{{GEN}}", f"{datetime.now(TZ):%d %b %Y %H:%M}").replace("{{N_EVENTS}}", str(len(events)))
           .replace("{{ANALYSIS}}", open(a.analysis).read())
           .replace("{{TRANSCRIPT}}", transcript(events, user, assistant))
           .replace("{{COUNT_TABLE}}", count_rows).replace("{{HB_TABLE}}", hb_rows).replace("{{STATUS_TABLE}}", st_rows).replace("{{MEMORY_TABLE}}", mem_table))
    left = re.findall(r"{{[A-Z_]+}}", out)
    if left:
        raise SystemExit(f"unfilled placeholders: {sorted(set(left))}")
    open(os.path.expanduser(a.out), "w").write(out)
    print(f"written {a.out} ({len(out) // 1024} KB)")


if __name__ == "__main__":
    main()
