"""
Combine all benchmark-specific overall.html files into:
  - overall.html  (standalone HTML page)
  - overall.ipynb (Jupyter notebook with pre-rendered outputs for GitHub preview)

Usage (from the 08a-judge-evaluation repo root):

    python compile_overall_html.py
"""

import json
import re
from pathlib import Path

BENCHMARKS = [
    ("JudgeBench-gpt",    "benchmarks/JudgeBench-gpt/overall.html"),
    ("RewardBench2-test", "benchmarks/RewardBench2-test/overall.html"),
    ("RM-Bench-train",    "benchmarks/RM-Bench-train/overall.html"),
]

CSS = """
    body { font-family: system-ui, sans-serif; font-size: 13px; padding: 1em 2em; color: #000; }
    h2 { margin-top: 2em; padding-bottom: 0.3em; border-bottom: 2px solid #2d3748; }
    h3 { margin-bottom: 0.4em; }
    table { border-collapse: collapse; white-space: nowrap; margin-bottom: 2em; }
    th { background: #2d3748; color: #edf2f7; padding: 6px 10px; text-align: left; }
    th.num { text-align: right; }
    td { padding: 5px 10px; border: 1px solid #e2e8f0; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    td.best { font-weight: 700; }
    sup { font-size: 9px; color: #000; margin-left: 2px; }
"""


def extract_body(html: str) -> str:
    m = re.search(r"<body>(.*?)</body>", html, re.DOTALL)
    return m.group(1).strip() if m else html


def build_notebook(sections: list[tuple[str, str]]) -> dict:
    """
    Build a notebook dict with pre-populated HTML outputs so GitHub renders
    the tables inline without requiring execution.

    sections: list of (benchmark_name, full_html_string)
    """
    cells = []

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "# Judge evaluation — all benchmarks",
    })

    for benchmark_name, full_html in sections:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": f"## {benchmark_name}",
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": (
                "from IPython.display import HTML\n"
                f"HTML(open('benchmarks/{benchmark_name}/overall.html').read())"
            ),
            "outputs": [{
                "output_type": "execute_result",
                "execution_count": None,
                "data": {
                    "text/html": full_html,
                    "text/plain": "<IPython.core.display.HTML object>",
                },
                "metadata": {},
            }],
        })

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


if __name__ == "__main__":
    loaded: list[tuple[str, str]] = []
    for benchmark_name, html_path in BENCHMARKS:
        path = Path(html_path)
        if not path.exists():
            print(f"  Skipping {html_path}: not found")
            continue
        loaded.append((benchmark_name, path.read_text(encoding="utf-8")))
        print(f"  Loaded {html_path}")

    if not loaded:
        print("No benchmark HTML files found.")
        raise SystemExit(1)

    # ── overall.html ──────────────────────────────────────────────────────────
    body_sections = [f"<h2>{name}</h2>\n{extract_body(html)}" for name, html in loaded]
    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Judge evaluation — all benchmarks</title>
<style>{CSS}</style>
</head>
<body>
{"<hr>\n".join(body_sections)}
</body>
</html>
"""
    Path("results/overall.html").write_text(html_out, encoding="utf-8")
    print("Wrote results/overall.html")

    # ── overall.ipynb ─────────────────────────────────────────────────────────
    notebook = build_notebook(loaded)
    Path("results/overall.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print("Wrote results/overall.ipynb")
