"""Generate HTML trade report for 2025-2026 trades."""
import pandas as pd

df = pd.read_csv("outputs/walkforward_trades_acc1_v14pp_profit_sim_trades.csv", parse_dates=["time"])
df = df[df["time"].dt.year >= 2025].copy()
df = df.sort_values(["fold", "time"])

df["result"] = df.apply(lambda r: "WIN" if r["is_win"] else ("LOSS" if r["is_loss"] else "DRAW"), axis=1)
df["realized_rr"] = df["realized_rr"].round(2)
df["net_rr"] = df["net_rr"].round(2)
df["pnl"] = df["pnl"].round(2)
df["balance_after"] = df["balance_after"].round(2)
df["fold"] = df["fold"].astype(int)
df["prob"] = (df["probability"] * 100).round(1)

total = len(df)
wins = int(df["is_win"].sum())
losses = int(df["is_loss"].sum())
wr = df["is_win"].mean() * 100
net_pnl = df["pnl"].sum()
gross_win = df.loc[df["pnl"] > 0, "pnl"].sum()
gross_loss = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
pf = gross_win / gross_loss if gross_loss > 0 else 0

rows_html = ""
for _, r in df.iterrows():
    res_class = "win" if r["result"] == "WIN" else ("loss" if r["result"] == "LOSS" else "draw")
    pnl_class = "pos" if r["pnl"] > 0 else "neg"
    rows_html += (
        f'<tr class="{res_class}">'
        f'<td>{r["time"].strftime("%Y-%m-%d")}</td>'
        f'<td>{r["time"].strftime("%H:%M")}</td>'
        f'<td class="side-{r["side"].lower()}">{r["side"].upper()}</td>'
        f'<td><span class="badge {res_class}">{r["result"]}</span></td>'
        f'<td>{r["realized_rr"]:.2f}</td>'
        f'<td>{r["net_rr"]:.2f}</td>'
        f'<td class="{pnl_class}">{r["pnl"]:+.2f}</td>'
        f'<td>{r["balance_after"]:.2f}</td>'
        f'<td>{r["prob"]:.1f}%</td>'
        f'<td>F{r["fold"]}</td>'
        f'</tr>\n'
    )

wr_color = "green" if wr >= 50 else "blue"
pnl_color = "green" if net_pnl > 0 else "red"

html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Trade Results 2025-2026 | ptp=2.5 tp=5.5</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 16px; }}
  h1 {{ color: #58a6ff; margin-bottom: 4px; font-size: 20px; }}
  .subtitle {{ color: #8b949e; font-size: 12px; margin-bottom: 16px; }}
  .stats {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 8px 14px; min-width: 110px; }}
  .stat-label {{ font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: .5px; }}
  .stat-value {{ font-size: 19px; font-weight: 700; margin-top: 2px; }}
  .green {{ color: #3fb950; }}
  .red {{ color: #f85149; }}
  .blue {{ color: #58a6ff; }}
  .controls {{ margin-bottom: 10px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
  input[type=text] {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; color: #c9d1d9; padding: 5px 10px; font-size: 12px; width: 180px; }}
  select {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; color: #c9d1d9; padding: 5px 10px; font-size: 12px; }}
  #rowcount {{ font-size: 11px; color: #8b949e; margin-left: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ background: #161b22; border-bottom: 2px solid #30363d; padding: 7px 9px; text-align: left; position: sticky; top: 0; cursor: pointer; user-select: none; white-space: nowrap; z-index: 1; }}
  th:hover {{ color: #58a6ff; }}
  td {{ padding: 4px 9px; border-bottom: 1px solid #1c2128; white-space: nowrap; }}
  tr.win {{ background: rgba(63,185,80,0.04); }}
  tr.loss {{ background: rgba(248,81,73,0.04); }}
  tr:hover td {{ background: rgba(88,166,255,0.08) !important; }}
  .badge {{ border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: 700; }}
  .badge.win {{ background: rgba(63,185,80,0.2); color: #3fb950; }}
  .badge.loss {{ background: rgba(248,81,73,0.2); color: #f85149; }}
  .badge.draw {{ background: rgba(139,148,158,0.2); color: #8b949e; }}
  .side-buy {{ color: #58a6ff; font-weight: 600; }}
  .side-sell {{ color: #f0883e; font-weight: 600; }}
  .pos {{ color: #3fb950; font-weight: 600; }}
  .neg {{ color: #f85149; font-weight: 600; }}
  .hidden {{ display: none; }}
</style>
</head>
<body>
<h1>Trade Results 2025–2026</h1>
<div class="subtitle">Config: acc1_v14pp_profit &nbsp;|&nbsp; partial_tp_rr=2.5, take_profit_rr=5.5 &nbsp;|&nbsp; --combo133 --risk-pct 0.05 --no-compound &nbsp;|&nbsp; 29 folds verified</div>
<div class="stats">
  <div class="stat"><div class="stat-label">Tổng trades</div><div class="stat-value blue">{total:,}</div></div>
  <div class="stat"><div class="stat-label">Wins</div><div class="stat-value green">{wins:,}</div></div>
  <div class="stat"><div class="stat-label">Losses</div><div class="stat-value red">{losses:,}</div></div>
  <div class="stat"><div class="stat-label">Win Rate</div><div class="stat-value {wr_color}">{wr:.1f}%</div></div>
  <div class="stat"><div class="stat-label">Net P&amp;L</div><div class="stat-value {pnl_color}">{net_pnl:+,.0f}$</div></div>
  <div class="stat"><div class="stat-label">Profit Factor</div><div class="stat-value green">{pf:.3f}</div></div>
</div>
<div class="controls">
  <input type="text" id="searchBox" oninput="filterTable()" placeholder="Tìm theo ngày, WIN, LOSS...">
  <select id="resultFilter" onchange="filterTable()">
    <option value="">Tất cả kết quả</option>
    <option value="WIN">WIN only</option>
    <option value="LOSS">LOSS only</option>
  </select>
  <select id="sideFilter" onchange="filterTable()">
    <option value="">Tất cả side</option>
    <option value="BUY">BUY only</option>
    <option value="SELL">SELL only</option>
  </select>
  <select id="yearFilter" onchange="filterTable()">
    <option value="">2025 + 2026</option>
    <option value="2025">2025 only</option>
    <option value="2026">2026 only</option>
  </select>
  <span id="rowcount">Hiển thị: {total:,} / {total:,} trades</span>
</div>
<table id="tradeTable">
<thead>
<tr>
  <th onclick="sortTable(0)">Date &#8645;</th>
  <th onclick="sortTable(1)">Time &#8645;</th>
  <th onclick="sortTable(2)">Side &#8645;</th>
  <th onclick="sortTable(3)">Result &#8645;</th>
  <th onclick="sortTable(4)">RealizedRR &#8645;</th>
  <th onclick="sortTable(5)">NetRR &#8645;</th>
  <th onclick="sortTable(6)">PnL ($) &#8645;</th>
  <th onclick="sortTable(7)">Balance ($) &#8645;</th>
  <th onclick="sortTable(8)">Prob &#8645;</th>
  <th onclick="sortTable(9)">Fold &#8645;</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
<script>
function filterTable() {{
  const search = document.getElementById("searchBox").value.toLowerCase();
  const result = document.getElementById("resultFilter").value.toUpperCase();
  const side = document.getElementById("sideFilter").value.toUpperCase();
  const year = document.getElementById("yearFilter").value;
  const rows = document.querySelectorAll("#tradeTable tbody tr");
  let visible = 0;
  rows.forEach(row => {{
    const cells = row.querySelectorAll("td");
    const rowText = Array.from(cells).map(c => c.innerText).join(" ").toLowerCase();
    const rowResult = cells[3] ? cells[3].innerText.trim().toUpperCase() : "";
    const rowSide = cells[2] ? cells[2].innerText.trim().toUpperCase() : "";
    const rowDate = cells[0] ? cells[0].innerText.trim() : "";
    const ok = (!search || rowText.includes(search))
             && (!result || rowResult === result)
             && (!side || rowSide === side)
             && (!year || rowDate.startsWith(year));
    row.classList.toggle("hidden", !ok);
    if (ok) visible++;
  }});
  document.getElementById("rowcount").innerText =
    "Hien thi: " + visible.toLocaleString() + " / {total:,} trades";
}}
let sortDir = {{}};
function sortTable(col) {{
  const tbody = document.querySelector("#tradeTable tbody");
  const rows = Array.from(tbody.querySelectorAll("tr"));
  const dir = sortDir[col] = !sortDir[col];
  rows.sort((a, b) => {{
    const av = a.querySelectorAll("td")[col] ? a.querySelectorAll("td")[col].innerText : "";
    const bv = b.querySelectorAll("td")[col] ? b.querySelectorAll("td")[col].innerText : "";
    const an = parseFloat(av.replace(/[+$%,]/g, ""));
    const bn = parseFloat(bv.replace(/[+$%,]/g, ""));
    if (!isNaN(an) && !isNaN(bn)) return dir ? an - bn : bn - an;
    return dir ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(r => tbody.appendChild(r));
  filterTable();
}}
</script>
</body>
</html>"""

with open("outputs/trade_results_2025_2026.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"Done! {total:,} trades saved to outputs/trade_results_2025_2026.html")
print(f"WR: {wr:.1f}% | PF: {pf:.3f} | Net PnL: {net_pnl:+,.2f}$")
