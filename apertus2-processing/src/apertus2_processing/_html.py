"""Standalone HTML report for inspected documents: tokens coloured by loss weight.

`STYLE` and `SCRIPT` are inserted verbatim into `PAGE`, a `string.Template`, so braces
in CSS and JavaScript need no escaping. The document data travels as JSON in a data
script element with every `<` escaped, which keeps any text safe inside the page.
"""

import html
import json
from string import Template

STYLE = """
:root { --bg:#fff; --fg:#1b1b1b; --muted:#6b6b6b; --line:#ddd; --panel:#f6f6f6;
        --ctl:#e8eefc; --ctlfg:#2a4a9a; --loss:#ffd27a; --bad:#f5c2c2;
        --bound:#7a7a7a; --alt:rgba(0,0,0,.06); color-scheme:light; }
:root[data-theme="dark"], :root:not([data-theme="light"]).system-dark {
  --bg:#161616; --fg:#e6e6e6; --muted:#9a9a9a; --line:#333; --panel:#1f1f1f;
  --ctl:#23304f; --ctlfg:#a9c0ff; --loss:#8a6a1a; --bad:#7a2d2d;
  --bound:#9a9a9a; --alt:rgba(255,255,255,.08); color-scheme:dark; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:14px/1.5 system-ui, sans-serif; }
header { padding:14px 20px 10px; border-bottom:1px solid var(--line); position:relative; }
header h1 { font-size:20px; margin:0 0 2px; }
header .sub { color:var(--muted); margin:0 0 6px; }
header .src { margin:0 0 10px; }
header .src code { background:var(--panel); padding:1px 6px; border-radius:4px; }
.controls { display:flex; flex-wrap:wrap; gap:16px; align-items:center; }
#theme { position:absolute; top:12px; right:20px; padding:4px 10px;
         border:1px solid var(--line); border-radius:6px; background:var(--panel);
         color:var(--fg); cursor:pointer; }
label { user-select:none; }
main { padding:0 20px 40px; }
nav { display:flex; flex-wrap:wrap; gap:6px; padding:12px 0; }
nav a { padding:2px 8px; border:1px solid var(--line); border-radius:6px;
        text-decoration:none; color:var(--fg); }
section { border-top:1px solid var(--line); padding:16px 0; }
h2 { font-size:15px; margin:0 0 8px; }
.meta { color:var(--muted); margin-bottom:8px; }
.doc { font:13px/1.7 ui-monospace, Menlo, Consolas, monospace; white-space:pre-wrap;
       word-break:break-word; background:var(--panel); padding:12px; border-radius:8px; }
.t { border-radius:2px; }
.t.loss { background:var(--loss); }
.t.ctl { background:var(--ctl); color:var(--ctlfg); font-weight:600; padding:0 2px; }
.t.bad { background:var(--bad); }
body.bounds .t { box-shadow:inset 0 0 0 1px var(--bound); border-radius:0; }
body.bounds .t:nth-child(even):not(.loss):not(.ctl):not(.bad) { background:var(--alt); }
body.nolegend .t.loss { background:transparent; }
body.nolegend.bounds .t.loss:nth-child(even) { background:var(--alt); }
.nl { color:var(--muted); }
details { margin-top:10px; }
summary { cursor:pointer; color:var(--muted); }
pre { white-space:pre-wrap; word-break:break-word; background:var(--panel);
      padding:12px; border-radius:8px; font-size:12px; }
.legend span { display:inline-block; padding:0 6px; border-radius:3px; margin-right:6px; }
.note { flex-basis:100%; color:var(--muted); font-size:13px; margin:0; }
"""

SCRIPT = r"""
const data = JSON.parse(document.getElementById('data').textContent);
const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function tokenSpan(d, i, controls) {
  const piece = d.pieces[i], w = d.weights[i];
  const cls = ['t', controls.has(d.ids[i]) ? 'ctl' : '', piece.includes('�') ? 'bad' : '',
               w > 0 ? 'loss' : ''].join(' ');
  const text = esc(piece).replace(/\n/g, '<span class="nl">⏎</span>\n');
  return `<span class="${cls}" title="#${i} id=${d.ids[i]} w=${w}">${text}</span>`;
}

function sourcePanel(source) {
  if (!source) return '';
  let pretty;
  try { pretty = JSON.stringify(JSON.parse(source.json), null, 1); } catch { pretty = source.json; }
  return `<details><summary>source ${esc(source.path)} row ${source.row}</summary>` +
         `<pre>${esc(pretty)}</pre></details>`;
}

function renderDocument(d) {
  const controls = new Set(d.controls);
  const spans = d.ids.map((_, i) => tokenSpan(d, i, controls)).join('');
  const section = document.createElement('section');
  section.id = 'd' + d.index;
  section.innerHTML = `<h2>document ${d.index} (of ${d.total})</h2>` +
    `<div class="meta">${esc(d.meta)}</div><div class="doc">${spans}</div>` +
    sourcePanel(d.source);
  return section;
}

function setupTheme() {
  const root = document.documentElement, button = document.getElementById('theme');
  const systemDark = matchMedia('(prefers-color-scheme: dark)');
  const applySystem = () => root.classList.toggle('system-dark', systemDark.matches);
  const isDark = () => root.dataset.theme ? root.dataset.theme === 'dark' : systemDark.matches;
  const label = () => { button.textContent = isDark() ? '☀ light' : '☽ dark'; };
  applySystem();
  systemDark.addEventListener('change', () => { applySystem(); label(); });
  button.onclick = () => { root.dataset.theme = isDark() ? 'light' : 'dark'; label(); };
  label();
}

const main = document.querySelector('main'), nav = document.querySelector('nav');
for (const d of data.documents) {
  const a = document.createElement('a');
  a.href = '#d' + d.index;
  a.textContent = d.index;
  nav.appendChild(a);
  main.appendChild(renderDocument(d));
}
document.getElementById('bounds').onchange =
  e => document.body.classList.toggle('bounds', e.target.checked);
document.getElementById('legend').onchange =
  e => document.body.classList.toggle('nolegend', !e.target.checked);
setupTheme();
"""

PAGE = Template("""<!doctype html>
<html><head><meta charset="utf-8"><title>apertus-data inspect</title>
<style>$style</style></head>
<body>
<header>
<button id="theme" type="button"></button>
<h1>apertus-data inspect</h1>
<p class="sub">Inspecting tokenized dataset samples: decoded text token by token, with loss
weights and source rows.</p>
<p class="src">Source path: <code>$run</code></p>
<div class="controls">
<label><input type="checkbox" id="bounds"> token boundaries</label>
<label><input type="checkbox" id="legend" checked> colour loss</label>
<span class="legend"><span class="t loss">loss weight &gt; 0</span>
<span class="t ctl">control</span><span class="t bad">partial character</span></span>
<p class="note">Without a written loss mask the colouring shows the Megatron trainer
defaults: loss on output items and waits, none on system and input tokens. With
<code>--weights</code> it shows the written <code>loss_weights</code>.</p>
</div>
</header>
<main><nav></nav></main>
<script id="data" type="application/json">$data</script>
<script>$script</script>
</body></html>
""")


def report(records, *, run):
    """Render records (see `_inspect.documents`) into one self-contained HTML page."""
    payload = {"run": run, "documents": list(records)}
    # `<` never appears in the data block, so no text can close or comment it out.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return PAGE.substitute(style=STYLE, script=SCRIPT, run=html.escape(run), data=data)
