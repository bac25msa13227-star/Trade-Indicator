"""Generate HTML report for risk=4.5% WF run."""
import pandas as pd
import re

# ── 1. Parse fold results from log ──────────────────────────────────────────
log = open('outputs/wf_risk0045_verify.txt').read()
fold_rows = []
for l in log.splitlines():
    # Two-pass: first get fold number + dates + bal, then extract fields
    m = re.search(
        r'Fold\s+(\d+).*?Test:\s+(\S+)\s+\S+\s+(\S+)'
        r'.*?AUC=([\d.]+).*?Prec=([\d.]+)'
        r'.*?Bal:\s+\$200\S\$([\d,]+(?:\.\d+)?)\s+\(([-+][\d.]+)%\)',
        l
    )
    if m:
        fold_rows.append({
            'fold': int(m.group(1)),
            'partial': 'PARTIAL' in l,
            'test_start': m.group(2),
            'test_end': m.group(3),
            'auc': float(m.group(4)),
            'prec': float(m.group(5)),
            'end_bal': float(m.group(6).replace(',', '')),
            'pct': float(m.group(7)),
        })

print(f"Parsed {len(fold_rows)} folds")

# ── 2. Load sim trades ───────────────────────────────────────────────────────
df = pd.read_csv('outputs/walkforward_trades_acc1_v14pp_profit_sim_trades.csv')
df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M')
df['result'] = df.apply(lambda r: 'WIN' if r['is_win'] else ('LOSS' if r['is_loss'] else 'DRAW'), axis=1)
df['realized_rr'] = df['realized_rr'].round(3)
df['net_rr'] = df['net_rr'].round(3)
df['pnl'] = df['pnl'].round(2)
df['balance_after'] = df['balance_after'].round(2)
df['drawdown'] = (df['drawdown'] * 100).round(2)
df['probability'] = (df['probability'] * 100).round(1)
df['entry_price'] = df['entry_price'].round(3)

print(f"Loaded {len(df)} trades")

# fold summary from df
fold_trade_counts = df.groupby('fold').agg(
    trades=('pnl', 'count'),
    wins=('is_win', 'sum'),
    pnl_sum=('pnl', 'sum')
).reset_index()
fold_trade_counts['wr'] = (fold_trade_counts['wins'] / fold_trade_counts['trades'] * 100).round(1)
fold_trade_counts['pnl_sum'] = fold_trade_counts['pnl_sum'].round(2)
ftc = fold_trade_counts.set_index('fold').to_dict('index')

# ── 3. Generate HTML ─────────────────────────────────────────────────────────
fold_html_parts = []
for f in fold_rows:
    fi = ftc.get(f['fold'], {})
    trades = fi.get('trades', 0)
    wr = fi.get('wr', 0)
    pnl_sum = fi.get('pnl_sum', 0)
    cls = 'loss' if f['pct'] < 0 else ('big' if f['pct'] > 1000 else '')
    partial_tag = '<span class="partial">PARTIAL</span>' if f['partial'] else ''
    pct_cls = 'neg' if f['pct'] < 0 else 'pos'
    pnl_cls = 'neg' if pnl_sum < 0 else 'pos'
    fold_html_parts.append(f"""
    <tr class="{cls}">
      <td>{f['fold']}{partial_tag}</td>
      <td>{f['test_start']}</td>
      <td>{f['test_end']}</td>
      <td>{f['auc']:.4f}</td>
      <td>{f['prec']:.4f}</td>
      <td>${f['end_bal']:,.0f}</td>
      <td class="{pct_cls}">{f['pct']:+.1f}%</td>
      <td>{trades}</td>
      <td>{wr:.1f}%</td>
      <td class="{pnl_cls}">${pnl_sum:,.2f}</td>
    </tr>""")

fold_html = '\n'.join(fold_html_parts)

trades_html_parts = []
reg_map = {0: 'Normal', 1: 'High-Vol', 2: 'Low-Vol'}
for _, r in df.iterrows():
    cls = 'win-row' if r['result'] == 'WIN' else ('loss-row' if r['result'] == 'LOSS' else 'draw-row')
    reg = reg_map.get(int(r.get('volatility_regime', 0)), str(r.get('volatility_regime', '')))
    dd_cls = ' dd-warn' if r['drawdown'] < -10 else ''
    pnl_cls = 'neg' if r['pnl'] < 0 else 'pos'
    side_cls = 'buy' if r['side'] == 'buy' else 'sell'
    throttle_reason = str(r.get('risk_throttle_reason', ''))
    throttle_mult = float(r.get('risk_throttle_multiplier', 1))
    trades_html_parts.append(f"""
    <tr class="{cls}">
      <td>{r['time']}</td>
      <td>{r['fold']}</td>
      <td class="{side_cls}">{r['side'].upper()}</td>
      <td>{r['entry_price']:.3f}</td>
      <td>{r['probability']:.1f}%</td>
      <td>{r['realized_rr']:.3f}</td>
      <td>{r['net_rr']:.3f}</td>
      <td class="{pnl_cls}">${r['pnl']:,.2f}</td>
      <td>${r['balance_after']:,.2f}</td>
      <td class="throttle{dd_cls}">{r['drawdown']:.2f}%</td>
      <td class="res-{r['result'].lower()}">{r['result']}</td>
      <td>{reg}</td>
      <td class="throttle" title="{throttle_reason}">{throttle_mult:.2f}x</td>
    </tr>""")

trades_html = '\n'.join(trades_html_parts)

total_pnl = df['pnl'].sum()
total_trades = len(df)
wins = int(df['is_win'].sum())
wr_total = wins / total_trades * 100
fold_opts = ''.join(f'<option value="{i}">{i}</option>' for i in range(1, 30))

html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WF Risk=4.5% Report — XAUUSD AI</title>
<style>
  :root {{
    --bg: #0d1117; --bg2: #161b22; --bg3: #1c2128;
    --border: #30363d; --text: #e6edf3; --muted: #8b949e;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --blue: #58a6ff; --purple: #bc8cff; --orange: #ffa657;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Consolas', monospace; font-size: 12px; }}
  h1 {{ color: var(--blue); padding: 20px; font-size: 18px; border-bottom: 1px solid var(--border); }}
  h2 {{ color: var(--orange); padding: 16px 20px 8px; font-size: 14px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; padding: 16px 20px; }}
  .card {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }}
  .card .label {{ color: var(--muted); font-size: 10px; text-transform: uppercase; margin-bottom: 4px; }}
  .card .value {{ font-size: 20px; font-weight: 700; }}
  .card .value.green {{ color: var(--green); }}
  .card .value.red {{ color: var(--red); }}
  .card .value.yellow {{ color: var(--yellow); }}
  .card .value.blue {{ color: var(--blue); }}
  .section {{ padding: 0 20px 20px; }}
  .table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  thead th {{ background: var(--bg3); color: var(--muted); text-align: left; padding: 8px 10px;
               border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 1; white-space: nowrap; cursor: pointer; user-select: none; }}
  thead th::after {{ content: ' ↕'; opacity: 0.3; }}
  thead th:hover {{ color: var(--text); }}
  tbody tr:hover {{ background: var(--bg3); }}
  tbody td {{ padding: 5px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  .loss {{ background: rgba(248,81,73,0.06); }}
  .big {{ background: rgba(63,185,80,0.04); }}
  .pos {{ color: var(--green); }}
  .neg {{ color: var(--red); }}
  .buy {{ color: var(--green); font-weight: 700; }}
  .sell {{ color: var(--red); font-weight: 700; }}
  .win-row td:first-child {{ border-left: 2px solid var(--green); }}
  .loss-row td:first-child {{ border-left: 2px solid var(--red); }}
  .draw-row td:first-child {{ border-left: 2px solid var(--muted); }}
  .res-win {{ color: var(--green); font-weight: 700; }}
  .res-loss {{ color: var(--red); font-weight: 700; }}
  .res-draw {{ color: var(--muted); }}
  .dd-warn {{ color: var(--yellow) !important; }}
  .partial {{ background: var(--yellow); color: black; font-size: 9px; padding: 1px 4px; border-radius: 3px; margin-left: 4px; vertical-align: middle; }}
  .throttle {{ color: var(--muted); }}
  .filter-bar {{ padding: 8px 20px; display: flex; gap: 12px; flex-wrap: wrap; align-items: center; background: var(--bg2); border-bottom: 1px solid var(--border); }}
  .filter-bar label {{ color: var(--muted); font-size: 11px; display: flex; align-items: center; gap: 5px; }}
  .filter-bar select, .filter-bar input {{ background: var(--bg); border: 1px solid var(--border); color: var(--text);
    padding: 4px 8px; border-radius: 4px; font-size: 11px; font-family: inherit; }}
  #tradeCount {{ color: var(--muted); font-size: 11px; padding: 6px 20px; border-bottom: 1px solid var(--border); }}
  tbody tr.hidden {{ display: none; }}
  .chip {{ background: var(--bg3); padding: 2px 6px; border-radius: 10px; font-size: 10px; color: var(--muted); }}
</style>
</head>
<body>

<h1>Walk-Forward Report — XAUUSD AI &nbsp;<span class="chip">risk=4.5%</span> <span class="chip">combo133</span> <span class="chip">no-compound</span> <span class="chip">2024-01 → 2026-04</span></h1>

<div class="summary-grid">
  <div class="card"><div class="label">Total PnL (29 folds)</div><div class="value green">${total_pnl:,.0f}</div></div>
  <div class="card"><div class="label">Total Trades</div><div class="value blue">{total_trades:,}</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value yellow">{wr_total:.1f}%</div></div>
  <div class="card"><div class="label">Wins / Losses</div><div class="value" style="font-size:14px"><span style="color:var(--green)">{wins}</span> / <span style="color:var(--red)">{total_trades - wins}</span></div></div>
  <div class="card"><div class="label">Losing Folds</div><div class="value red">3 / 29</div></div>
  <div class="card"><div class="label">Worst Fold</div><div class="value red">-17.6% ($165)</div></div>
  <div class="card"><div class="label">Best Fold</div><div class="value green">+27,467% ($55k)</div></div>
  <div class="card"><div class="label">Avg Max DD</div><div class="value yellow">-14.84%</div></div>
  <div class="card"><div class="label">Profit Factor</div><div class="value blue">2.419</div></div>
</div>

<h2>Fold-by-Fold Results <span class="chip">click header to sort</span></h2>
<div class="section">
<div class="table-wrap">
<table id="foldTable">
  <thead>
    <tr>
      <th>Fold</th><th>Test Start</th><th>Test End</th><th>AUC</th><th>Precision</th>
      <th>End Balance</th><th>Return%</th><th>Trades</th><th>WR%</th><th>Net PnL</th>
    </tr>
  </thead>
  <tbody>
{fold_html}
  </tbody>
</table>
</div>
</div>

<h2>Trade Log — {total_trades:,} trades <span class="chip">click headers to sort</span></h2>
<div class="filter-bar">
  <label>Fold: <select id="fFold"><option value="">All</option>{fold_opts}</select></label>
  <label>Result: <select id="fResult">
    <option value="">All</option><option value="WIN">WIN</option>
    <option value="LOSS">LOSS</option><option value="DRAW">DRAW</option>
  </select></label>
  <label>Side: <select id="fSide">
    <option value="">All</option><option value="BUY">BUY</option><option value="SELL">SELL</option>
  </select></label>
  <label>Min Net RR &ge; <input id="fMinRR" type="number" step="0.5" placeholder="-2" style="width:65px"></label>
  <label>Max Net RR &le; <input id="fMaxRR" type="number" step="0.5" placeholder="20" style="width:65px"></label>
</div>
<div id="tradeCount"></div>
<div class="section">
<div class="table-wrap" style="max-height:620px; overflow-y:auto;">
<table id="tradeTable">
  <thead>
    <tr>
      <th>Time</th><th>Fold</th><th>Side</th><th>Entry</th><th>Prob%</th>
      <th>RealRR</th><th>NetRR</th><th>PnL ($)</th><th>Balance</th><th>DD%</th>
      <th>Result</th><th>Regime</th><th>Throttle</th>
    </tr>
  </thead>
  <tbody id="tradeTbody">
{trades_html}
  </tbody>
</table>
</div>
</div>

<script>
function applyFilters() {{
  const fold   = document.getElementById('fFold').value;
  const result = document.getElementById('fResult').value;
  const side   = document.getElementById('fSide').value;
  const minRR  = document.getElementById('fMinRR').value !== '' ? parseFloat(document.getElementById('fMinRR').value) : -Infinity;
  const maxRR  = document.getElementById('fMaxRR').value !== '' ? parseFloat(document.getElementById('fMaxRR').value) : Infinity;
  const rows = document.querySelectorAll('#tradeTbody tr');
  let visible = 0;
  rows.forEach(r => {{
    const cells = r.querySelectorAll('td');
    const show = (!fold   || cells[1].textContent.trim() === fold) &&
                 (!result || cells[10].textContent.trim() === result) &&
                 (!side   || cells[2].textContent.trim() === side) &&
                 (parseFloat(cells[6].textContent) >= minRR) &&
                 (parseFloat(cells[6].textContent) <= maxRR);
    r.classList.toggle('hidden', !show);
    if (show) visible++;
  }});
  document.getElementById('tradeCount').textContent =
    'Showing ' + visible.toLocaleString() + ' of {total_trades:,} trades';
}}
['fFold','fResult','fSide','fMinRR','fMaxRR'].forEach(id =>
  document.getElementById(id).addEventListener('input', applyFilters));
applyFilters();

// Sortable tables
function makeSortable(tableId) {{
  const table = document.getElementById(tableId);
  if (!table) return;
  table.querySelectorAll('thead th').forEach((th, colIdx) => {{
    th.addEventListener('click', () => {{
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr:not(.hidden)'));
      const asc = th.dataset.asc !== 'true';
      table.querySelectorAll('thead th').forEach(t => delete t.dataset.asc);
      th.dataset.asc = asc;
      rows.sort((a, b) => {{
        const av = (a.querySelectorAll('td')[colIdx]?.textContent || '').replace(/[$,+%x]/g,'').trim();
        const bv = (b.querySelectorAll('td')[colIdx]?.textContent || '').replace(/[$,+%x]/g,'').trim();
        const an = parseFloat(av), bn = parseFloat(bv);
        if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      }});
      rows.forEach(r => tbody.appendChild(r));
    }});
  }});
}}
makeSortable('foldTable');
makeSortable('tradeTable');
</script>
</body>
</html>"""

with open('outputs/wf_risk045_report.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Done: outputs/wf_risk045_report.html")
print(f"Folds: {len(fold_rows)}, Trades: {total_trades:,}, PnL: ${total_pnl:,.2f}, WR: {wr_total:.1f}%")
