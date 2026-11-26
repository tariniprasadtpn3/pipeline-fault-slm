"""Demo web UI on the standard library only (WSGI, no extra dependencies).

    python -m src.demo.app            # then open http://127.0.0.1:8000

Pick a synthetic fault to see its log, the diagnosed action and what the guarded
agent step does to a toy table, or paste a log to get a diagnosis only. The
default predictor is the rule baseline; pass any object with `predict(log)`
(for example an `ActionModel`) to `make_app` to demo a model instead.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from src.faults import FAULT_TYPES

from .core import RuleModel, diagnose_log, run_sample, sample_for, toy_table

MAX_BODY = 64 * 1024

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pipeline-fault-slm demo</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 52rem; margin: 2rem auto; padding: 0 1rem; }}
pre, textarea {{ background: #f4f4f4; color: #111; padding: .75rem; overflow-x: auto; }}
textarea {{ width: 100%; box-sizing: border-box; }}
table {{ border-collapse: collapse; margin-bottom: 1rem; }}
th, td {{ border: 1px solid #bbb; padding: .25rem .6rem; text-align: left; }}
</style></head><body>
<h1>pipeline-fault-slm demo</h1>
<p>All data is synthetic. Predictor: <b>{predictor}</b>.</p>
<form method="post" action="/">
<p><label>Fault <select name="fault">{options}</select></label>
<label>Seed <input name="seed" type="number" value="{seed}"></label>
<label><input type="checkbox" name="apply" value="1"{checked}> apply (default is dry-run)</label></p>
<p><label>Or paste a failure log (diagnosis only, nothing is executed)<br>
<textarea name="log" rows="5">{log}</textarea></label></p>
<p><button type="submit">Diagnose</button></p>
</form>
{result}
</body></html>
"""


def _json_block(value: Any) -> str:
    return f"<pre>{escape(json.dumps(value, indent=2, sort_keys=True))}</pre>"


def _rows_table(title: str, rows) -> str:
    head = f"<h3>{escape(title)} ({len(rows)} rows)</h3>"
    if not rows:
        return head
    cols: List[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    body = "".join(
        "<tr>" + "".join(f"<td>{escape('null' if r.get(c) is None else str(r.get(c)))}</td>" for c in cols) + "</tr>"
        for r in rows
    )
    header = "".join(f"<th>{escape(c)}</th>" for c in cols)
    return f"{head}<table><tr>{header}</tr>{body}</table>"


def _sample_result(fault: str, seed: int, apply: bool, model: Any) -> str:
    sample = sample_for(fault, seed)
    before = toy_table(sample)
    run = run_sample(sample, model, apply=apply)
    entry = run.audit[0]
    parts = [f"<h2>Failure log</h2><pre>{escape(sample.log)}</pre>"]
    parts.append(f"<h2>Agent step: {escape(entry['status'])}</h2>{_json_block(entry)}")
    predicted = entry.get("action")
    verdict = "matches" if predicted == sample.action else "does not match"
    parts.append(f"<p>Prediction {verdict} the ground-truth action.</p>{_json_block(sample.action)}")
    parts.append(_rows_table("Before", before.rows))
    if entry["status"] == "applied":
        parts.append(_rows_table("After", run.table.rows))
        if run.table.quarantine:
            parts.append(_rows_table("Quarantine", run.table.quarantine))
    else:
        parts.append("<p>Table unchanged.</p>")
    return "".join(parts)


def _log_result(log: str, model: Any) -> str:
    pred = diagnose_log(log, model)
    if not pred.ok:
        return f"<h2>No valid action</h2>{_json_block(pred.errors)}"
    return f"<h2>Diagnosed action (not executed)</h2>{_json_block(pred.action)}"


def render(form: Dict[str, str], model: Any, predictor: str) -> str:
    """The full page for a submitted form (empty dict: the blank form).

    Raises ValueError on an unknown fault type or a non-integer seed.
    """
    fault = form.get("fault", FAULT_TYPES[0])
    seed_text = form.get("seed", "42").strip()
    log = form.get("log", "").replace("\r\n", "\n").strip()
    apply = form.get("apply") == "1"
    result = ""
    if log:
        result = _log_result(log, model)
    elif form:
        try:
            seed = int(seed_text)
        except ValueError:
            raise ValueError("seed must be an integer") from None
        result = _sample_result(fault, seed, apply, model)
    options = "".join(
        f'<option value="{f}"{" selected" if f == fault else ""}>{f}</option>' for f in FAULT_TYPES
    )
    return _PAGE.format(
        predictor=escape(predictor), options=options, seed=escape(seed_text),
        checked=" checked" if apply else "", log=escape(log), result=result,
    )


def make_app(model: Optional[Any] = None, predictor: str = "rule baseline"):
    """A WSGI app serving the demo page at `/`."""
    model = RuleModel() if model is None else model

    def app(environ, start_response):
        def reply(status: str, body: str, kind: str = "text/plain"):
            data = body.encode("utf-8")
            start_response(status, [("Content-Type", f"{kind}; charset=utf-8"),
                                    ("Content-Length", str(len(data)))])
            return [data]

        if environ.get("PATH_INFO", "/") != "/":
            return reply("404 Not Found", "not found")
        method = environ.get("REQUEST_METHOD", "GET")
        if method not in ("GET", "POST"):
            return reply("405 Method Not Allowed", "use GET or POST")
        form: Dict[str, str] = {}
        if method == "POST":
            try:
                size = int(environ.get("CONTENT_LENGTH") or 0)
            except ValueError:
                size = -1
            if size < 0:
                return reply("400 Bad Request", "bad Content-Length")
            if size > MAX_BODY:
                return reply("413 Content Too Large", "request too large")
            text = environ["wsgi.input"].read(size).decode("utf-8", "replace")
            form = {k: v[0] for k, v in parse_qs(text).items()}
        try:
            page = render(form, model, predictor)
        except ValueError as e:
            return reply("400 Bad Request", str(e))
        return reply("200 OK", page, "text/html")

    return app


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run the pipeline-fault demo UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    with make_server(args.host, args.port, make_app()) as server:
        print(f"demo on http://{args.host}:{args.port} (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
