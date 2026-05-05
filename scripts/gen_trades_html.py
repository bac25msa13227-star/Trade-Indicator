"""Generate per-trade HTML log for 2025+ folds (F14-F29)."""
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "outputs" / "walkforward_trades_acc1_v14pp_profit_sim_trades.csv"
OUT_PATH = ROOT / "outputs" / "trade_log_2025.html"

FOLD_NAMES = {
    14: "F14  2025-01-15 → 02-14",
    15: "F15  2025-02-14 → 03-17",
    16: "F16  2025-03-17 → 04-16",
    17: "F17  2025-04-16 → 05-19",
    18: "F18  2025-05-19 → 06-18  ❌ LOSING",
    19: "F19  2025-06-18 → 07-18",
    20: "F20  2025-07-18 → 08-19",
    21: "F21  2025-08-19 → 09-17",
    22: "F22  2025-09-17 → 10-17",
    23: "F23  2025-10-17 → 11-18  ❌ LOSING",
    24: "F24  2025-11-18 → 12-18",
    25: "F25  2025-12-18 → 2026-01-21",
    26: "F26  2026-01-21 → 02-20",
    27: "F27  2026-02-20 → 03-23",
    28: "F28  2026-03-23 → 04-23",
    29: "F29  2026-04-23 → 04-30  [PARTIAL]",
}
LOSING_FOLDS = {18, 23}

def close_time_str(t: str, bars: int) -> str:
    """Estimate close time from open time + bars*5min."""
    dt = datetime.fromisoformat(t)
    return (dt + timedelta(minutes=bars * 5)).isoformat()


rows = []
with open(CSV_PATH, newline="") as f:
    for r in csv.DictReader(f):
        if int(r["fold"]) >= 14:
            bars = int(r["bars_held"])
            ct = close_time_str(r["time"], bars)
            rows.append({
                "time": r["time"],
                "ct": ct,          # estimated close time — used for sorting
                "side": r["side"],
                "entry": round(float(r["entry_price"]), 3),
                "rr": round(float(r["net_rr"]), 3),
                "pnl": round(float(r["pnl"]), 2),
                "rf": round(float(r["risk_fraction"]) * 100, 2),   # effective risk %
                "bb": round(float(r["balance_before"]), 2),
                "ba": round(float(r["balance_after"]), 2),
                "dd": round(float(r["drawdown"]) * 100, 2),
                "w": r["is_win"] == "True",
                "prob": round(float(r["probability"]), 3),
                "bars": bars,
                "regime": int(r["volatility_regime"]),
                "fold": int(r["fold"]),
                "ts": r["test_start"],
                "te": r["test_end"],
            })

# Sort by CLOSE time so balance sequence is always continuous
rows.sort(key=lambda x: (x["fold"], x["ct"]))

# Detect concurrent groups: trades where open-time overlaps (same fold, open < another's close)
# Mark trades that are part of a concurrent group
for i, r in enumerate(rows):
    r["conc"] = False
    if i > 0 and rows[i-1]["fold"] == r["fold"]:
        # if this trade opened BEFORE the previous trade closed → concurrent
        if r["time"] < rows[i-1]["ct"]:
            r["conc"] = True
            rows[i-1]["conc"] = True

fold_info: dict = {}
for r in rows:
    f = str(r["fold"])
    if f not in fold_info:
        fold_info[f] = {"ts": r["ts"], "te": r["te"], "trades": 0, "wins": 0, "name": FOLD_NAMES[r["fold"]]}
    fold_info[f]["trades"] += 1
    if r["w"]:
        fold_info[f]["wins"] += 1
    fold_info[f]["end_bal"] = r["ba"]

trades_js = json.dumps(rows, separators=(",", ":"))
folds_js = json.dumps(fold_info, separators=(",", ":"))
losing_js = json.dumps(list(LOSING_FOLDS))

print(f"Trades 2025+: {len(rows)}")
for fk, fi in sorted(fold_info.items(), key=lambda x: int(x[0])):
    pnl = fi["end_bal"] - 200
    print(f"  {fi['name']:40s}  {fi['trades']:3d} trades  WR={100*fi['wins']//fi['trades']}%  end=${fi['end_bal']:.2f}  P&L={'+' if pnl>=0 else ''}{pnl:.2f}")

HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Trade Log 2025 — XAUUSD ACC1</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f1117;color:#e1e4e8;font-size:13px}
h1{text-align:center;padding:20px 16px 4px;font-size:20px;color:#58a6ff}
.sub{text-align:center;color:#8b949e;font-size:11px;padding-bottom:14px}
.container{max-width:1440px;margin:0 auto;padding:0 12px 40px}
.filters{display:flex;gap:10px;flex-wrap:wrap;align-items:center;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px;margin-bottom:12px}
.filters label{color:#8b949e;font-size:11px;white-space:nowrap}
select,input[type=text]{background:#0d1117;border:1px solid #30363d;color:#e6edf3;border-radius:4px;padding:3px 7px;font-size:12px}
.btn{background:#21262d;border:1px solid #30363d;color:#e6edf3;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:12px}
.btn:hover{background:#30363d}
.btn.active{background:#1f6feb;border-color:#388bfd;color:#fff}
.summary{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.scard{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:7px 13px;min-width:90px;text-align:center}
.scard .sl{font-size:10px;color:#8b949e;text-transform:uppercase}
.scard .sv{font-size:16px;font-weight:700}
.pos{color:#3fb950}.neg{color:#f85149}.neu{color:#8b949e}
.chart-wrap{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;margin-bottom:12px}
canvas{display:block;width:100%}
.tbl-wrap{overflow-x:auto;border-radius:8px;border:1px solid #30363d}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{background:#161b22;padding:7px 8px;text-align:right;color:#8b949e;font-weight:600;white-space:nowrap;border-bottom:2px solid #30363d;position:sticky;top:0;z-index:10}
th:nth-child(1),th:nth-child(2),th:nth-child(3),th:nth-child(4){text-align:left}
td{padding:4px 8px;text-align:right;border-bottom:1px solid #21262d;white-space:nowrap}
td:nth-child(1),td:nth-child(2),td:nth-child(3),td:nth-child(4){text-align:left}
.conc-badge{display:inline-block;font-size:9px;color:#e3b341;background:#1e1a06;border:1px solid #e3b341;border-radius:2px;padding:0 3px;margin-left:3px;vertical-align:middle}
tr:hover td{background:#1c2128}
.fold-banner td{background:#0d1a2d!important;font-size:11px;color:#58a6ff;font-weight:700;padding:7px 10px;border-left:4px solid #388bfd!important;border-top:3px solid #388bfd!important}
.fold-banner.losing td{background:#1a0505!important;color:#f85149;border-left-color:#f85149!important;border-top-color:#f85149!important}
.fold-end-banner td{background:#0a1a0d!important;font-size:10px;color:#3fb950;padding:4px 10px;border-left:4px solid #3fb950!important;text-align:right}
.fold-end-banner.losing td{background:#1a0505!important;color:#f85149;border-left-color:#f85149!important}
.day-sep td{border-top:1px dashed #30363d!important;background:#0d1117!important;color:#484f58;font-size:10px;padding:2px 8px}
tr.win td:first-child{border-left:3px solid #3fb950}
tr.loss td:first-child{border-left:3px solid #f85149}
.badge{display:inline-block;border-radius:3px;padding:1px 5px;font-size:10px;font-weight:700}
.badge.buy{background:#0a2a0a;color:#3fb950}
.badge.sell{background:#2a0a0a;color:#f85149}
.r0{color:#58a6ff}.r1{color:#e6edf3}.r2{color:#e3b341}
.pages{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:10px 0}
.page-info{color:#8b949e;font-size:12px}
footer{text-align:center;color:#484f58;font-size:11px;padding:14px;border-top:1px solid #21262d;margin-top:16px}
</style>
</head>
<body>
<h1>Trade Log — XAUUSD ACC1 | 2025 → Apr 2026</h1>
<div class="sub">16 folds (F14–F29) &nbsp;|&nbsp; $200 start/fold (no-compound) &nbsp;|&nbsp; 3,331 trades &nbsp;|&nbsp; Config: acc1_v14pp_profit ptp2.5 tp5.5 risk5%WF</div>

<div class="container">
<div class="filters">
  <label>Fold:</label>
  <select id="foldSel"><option value="all">All Folds</option></select>
  <label>Side:</label>
  <select id="sideSel">
    <option value="all">All</option><option value="buy">Buy</option><option value="sell">Sell</option>
  </select>
  <label>Result:</label>
  <select id="resultSel">
    <option value="all">All</option><option value="win">Win</option><option value="loss">Loss</option>
  </select>
  <label>Regime:</label>
  <select id="regimeSel">
    <option value="all">All</option><option value="0">Sideway</option><option value="1">Normal</option><option value="2">Volatile</option>
  </select>
  <label>Date filter:</label>
  <input type="text" id="searchInput" placeholder="e.g. 2025-03" style="width:120px">
  <button class="btn" onclick="resetFilters()">Reset</button>
  <span id="countLabel" class="neu" style="margin-left:auto;font-size:11px"></span>
</div>

<div class="summary" id="summaryBar"></div>
<div class="chart-wrap"><canvas id="eqChart" height="100"></canvas></div>

<div class="note" style="font-size:11px;color:#8b949e;padding:6px 10px;margin-bottom:6px;background:#161b22;border:1px solid #30363d;border-radius:6px">
  ℹ️ Trades sorted by <strong>close time</strong> (= open + bars×5min) for correct balance continuity.
  <span style="color:#e3b341">⚡CONC</span> = concurrent position (overlapping hold period).
  P&amp;L = Bal_before × Risk% × Net_RR — Risk% varies with lot-snap &amp; ATR stop.
</div>
<div class="tbl-wrap">
<table>
<thead>
<tr>
  <th>#</th><th>Open (UTC)</th><th>Close≈</th><th>Side</th>
  <th>Entry</th><th>Net RR</th><th>P&amp;L $</th><th>Risk%</th>
  <th>Bal Before</th><th>Bal After</th><th>DD%</th>
  <th>Prob</th><th>Bars</th><th>Fold</th>
</tr>
</thead>
<tbody id="tblBody"></tbody>
</table>
</div>
<div class="pages" id="pagesBar"></div>

<footer>
  Config: acc1_v14pp_profit_candidate_ptp25_tp55.yaml &nbsp;|&nbsp; risk 5% WF / 3% live &nbsp;|&nbsp;
  Model: acc1_combo133_202604 (FROZEN Apr 26 2026) &nbsp;|&nbsp; Generated May 2 2026
</footer>
</div>

<script>
const TRADES=TRADES_PLACEHOLDER;
const FOLD_INFO=FOLD_INFO_PLACEHOLDER;
const LOSING_FOLDS=new Set(LOSING_PLACEHOLDER);
const FOLD_NAMES={14:"F14 2025-01-15→02-14",15:"F15 2025-02-14→03-17",16:"F16 2025-03-17→04-16",17:"F17 2025-04-16→05-19",18:"F18 2025-05-19→06-18 ❌",19:"F19 2025-06-18→07-18",20:"F20 2025-07-18→08-19",21:"F21 2025-08-19→09-17",22:"F22 2025-09-17→10-17",23:"F23 2025-10-17→11-18 ❌",24:"F24 2025-11-18→12-18",25:"F25 2025-12-18→2026-01-21",26:"F26 2026-01-21→02-20",27:"F27 2026-02-20→03-23",28:"F28 2026-03-23→04-23",29:"F29 2026-04-23→04-30 [PARTIAL]"};

const PAGE_SIZE=150;
let filtered=[], page=0;

// Build fold select
const fsel=document.getElementById('foldSel');
for(let f=14;f<=29;f++){
  const o=document.createElement('option');
  o.value=f; o.textContent=FOLD_NAMES[f]; fsel.appendChild(o);
}

function getFilters(){
  return {
    fold:document.getElementById('foldSel').value,
    side:document.getElementById('sideSel').value,
    result:document.getElementById('resultSel').value,
    regime:document.getElementById('regimeSel').value,
    search:document.getElementById('searchInput').value.trim()
  };
}

function applyFilters(){
  const f=getFilters();
  filtered=TRADES.filter(t=>{
    if(f.fold!=='all'&&t.fold!=+f.fold)return false;
    if(f.side!=='all'&&t.side!==f.side)return false;
    if(f.result==='win'&&!t.w)return false;
    if(f.result==='loss'&&t.w)return false;
    if(f.regime!=='all'&&t.regime!=+f.regime)return false;
    if(f.search&&!t.time.includes(f.search)&&!t.ct.includes(f.search))return false;
    return true;
  });
  page=0;
  renderSummary();
  renderChart();
  renderTable();
  renderPages();
}

function resetFilters(){
  ['foldSel','sideSel','resultSel','regimeSel'].forEach(id=>document.getElementById(id).value='all');
  document.getElementById('searchInput').value='';
  applyFilters();
}

function renderSummary(){
  if(!filtered.length){
    document.getElementById('summaryBar').innerHTML='<div class="scard"><div class="sv neu">No trades</div></div>';
    document.getElementById('countLabel').textContent='0 trades';
    return;
  }
  const wins=filtered.filter(t=>t.w).length;
  const tpnl=filtered.reduce((s,t)=>s+t.pnl,0);
  const worstDd=Math.min(...filtered.map(t=>t.dd));
  const avgRr=filtered.reduce((s,t)=>s+t.rr,0)/filtered.length;
  const wr=wins/filtered.length;
  document.getElementById('summaryBar').innerHTML=`
    <div class="scard"><div class="sl">Trades</div><div class="sv neu">${filtered.length.toLocaleString()}</div></div>
    <div class="scard"><div class="sl">Win / Loss</div><div class="sv"><span class="pos">${wins}</span> / <span class="neg">${filtered.length-wins}</span></div></div>
    <div class="scard"><div class="sl">Win Rate</div><div class="sv ${wr>=0.4?'pos':'neg'}">${(wr*100).toFixed(1)}%</div></div>
    <div class="scard"><div class="sl">Total P&amp;L</div><div class="sv ${tpnl>=0?'pos':'neg'}">${tpnl>=0?'+':''}$${tpnl.toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2})}</div></div>
    <div class="scard"><div class="sl">Avg RR</div><div class="sv ${avgRr>=0?'pos':'neg'}">${avgRr>=0?'+':''}${avgRr.toFixed(3)}</div></div>
    <div class="scard"><div class="sl">Worst DD</div><div class="sv ${Math.abs(worstDd)<=15?'pos':Math.abs(worstDd)<=18?'neu':'neg'}">${worstDd.toFixed(2)}%</div></div>
  `;
  document.getElementById('countLabel').textContent=filtered.length.toLocaleString()+' trades';
}

function renderChart(){
  const canvas=document.getElementById('eqChart');
  const W=canvas.parentElement.clientWidth-20;
  canvas.width=W; canvas.height=100;
  const ctx=canvas.getContext('2d');
  ctx.fillStyle='#161b22';ctx.fillRect(0,0,W,100);
  if(!filtered.length)return;

  const bals=filtered.map(t=>t.ba);
  const minB=Math.min(...bals,150);
  const maxB=Math.max(...bals);
  const rangeB=maxB-minB||1;
  const PAD_L=36,PAD_T=8,H=85;

  // Fold color bands + dividers
  const folds=[...new Set(filtered.map(t=>t.fold))].sort((a,b)=>a-b);
  folds.forEach(f=>{
    const idxs=filtered.reduce((a,t,i)=>t.fold===f?[...a,i]:a,[]);
    if(!idxs.length)return;
    const x1=PAD_L+(idxs[0]/(filtered.length-1||1))*(W-PAD_L);
    const x2=PAD_L+(idxs[idxs.length-1]/(filtered.length-1||1))*(W-PAD_L);
    ctx.fillStyle=LOSING_FOLDS.has(f)?'rgba(248,81,73,0.08)':'rgba(56,139,253,0.05)';
    ctx.fillRect(x1,PAD_T,x2-x1,H-PAD_T);
    if(idxs[0]>0){
      const x=PAD_L+(idxs[0]/(filtered.length-1||1))*(W-PAD_L);
      ctx.strokeStyle=LOSING_FOLDS.has(f)?'rgba(248,81,73,0.6)':'rgba(56,139,253,0.5)';
      ctx.lineWidth=1;ctx.setLineDash([3,3]);
      ctx.beginPath();ctx.moveTo(x,PAD_T);ctx.lineTo(x,H);ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle=LOSING_FOLDS.has(f)?'#f85149':'#58a6ff';
      ctx.font='9px monospace';
      ctx.fillText('F'+f,x+2,PAD_T+9);
    }
  });

  // Zero/200 baseline
  const y200=H-(0-minB)/rangeB*(H-PAD_T);
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;ctx.setLineDash([2,4]);
  ctx.beginPath();ctx.moveTo(PAD_L,y200);ctx.lineTo(W,y200);ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#484f58';ctx.font='9px monospace';
  ctx.fillText('$200',2,y200+3);

  // Y-axis labels
  ctx.fillStyle='#484f58';ctx.font='9px monospace';
  const labels=[200,1000,5000,10000].filter(v=>v<=maxB*1.1);
  labels.forEach(v=>{
    const y=H-(v-minB)/rangeB*(H-PAD_T);
    if(y>=PAD_T&&y<=H){
      ctx.fillText(v>=1000?'$'+(v/1000).toFixed(0)+'k':'$'+v,2,y+3);
    }
  });

  // Balance line
  ctx.strokeStyle='#3fb950';ctx.lineWidth=1.5;
  ctx.beginPath();
  filtered.forEach((t,i)=>{
    const x=PAD_L+(i/(filtered.length-1||1))*(W-PAD_L);
    const y=H-(t.ba-minB)/rangeB*(H-PAD_T);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();
}

function renderTable(){
  const tbody=document.getElementById('tblBody');
  tbody.innerHTML='';
  const start=page*PAGE_SIZE;
  const slice=filtered.slice(start,start+PAGE_SIZE);
  let prevFold=null,prevDay=null,num=start;

  slice.forEach(t=>{
    num++;
    const day=t.ct.substring(0,10);  // group by CLOSE date
    const newFold=t.fold!==prevFold;
    const newDay=!newFold&&day!==prevDay;

    if(newFold){
      const fi=FOLD_INFO[t.fold];
      const pnl=fi.end_bal-200;
      const isL=LOSING_FOLDS.has(t.fold);
      const wr=fi.trades?Math.round(100*fi.wins/fi.trades):0;
      const br=document.createElement('tr');
      br.className='fold-banner'+(isL?' losing':'');
      br.innerHTML=`<td colspan="14">
        ${isL?'❌':'▶'}&nbsp;<strong>${fi.name}</strong>
        &nbsp;&nbsp;|&nbsp;&nbsp;Start <strong>$200.00</strong>
        &nbsp;|&nbsp; ${fi.trades} trades &nbsp;|&nbsp; WR ${wr}%
        &nbsp;|&nbsp; End <strong>$${fi.end_bal.toLocaleString('en',{minimumFractionDigits:2})}</strong>
        &nbsp;|&nbsp; P&amp;L <strong>${pnl>=0?'+':''}$${Math.abs(pnl).toFixed(2)}</strong>
        ${isL?'&nbsp;❌ LOSING FOLD':'&nbsp;✅ WIN FOLD'}
      </td>`;
      tbody.appendChild(br);
      prevFold=t.fold;prevDay=null;
    } else if(newDay){
      const dr=document.createElement('tr');
      dr.className='day-sep';
      dr.innerHTML=`<td colspan="14">── ${day}</td>`;
      tbody.appendChild(dr);
    }
    prevDay=day;

    const tr=document.createElement('tr');
    tr.className=t.w?'win':'loss';
    const pnlC=t.pnl>=0?'pos':'neg';
    const rrC=t.rr>0?'pos':t.rr<-0.5?'neg':'neu';
    const ddC=Math.abs(t.dd)>15?'neg':Math.abs(t.dd)>10?'neu':'pos';
    const rfC=t.rf>7?'neg':t.rf>5?'neu':'pos';  // flag unusually high effective risk
    const timeOpen=t.time.replace('T',' ').replace('+00:00','').substring(5); // MM-DD HH:MM
    const timeClose=t.ct.replace('T',' ').replace('+00:00','').substring(11,16); // HH:MM
    const concBadge=t.conc?'<span class="conc-badge">⚡CONC</span>':'';
    tr.innerHTML=`
      <td style="color:#484f58;font-size:10px">${num}</td>
      <td style="font-family:monospace;font-size:11px">${timeOpen}${concBadge}</td>
      <td style="font-family:monospace;font-size:10px;color:#484f58">${timeClose}</td>
      <td><span class="badge ${t.side}">${t.side.toUpperCase()}</span></td>
      <td>${t.entry.toLocaleString('en',{minimumFractionDigits:3})}</td>
      <td class="${rrC}">${t.rr>=0?'+':''}${t.rr.toFixed(3)}</td>
      <td class="${pnlC}">${t.pnl>=0?'+':''}${t.pnl.toFixed(2)}</td>
      <td class="${rfC}" style="font-size:11px">${t.rf.toFixed(2)}%</td>
      <td class="neu" style="font-size:11px">$${t.bb.toLocaleString('en',{minimumFractionDigits:2})}</td>
      <td><strong class="${t.ba>=200?'pos':'neg'}">$${t.ba.toLocaleString('en',{minimumFractionDigits:2})}</strong></td>
      <td class="${ddC}">${t.dd.toFixed(2)}%</td>
      <td class="neu">${(t.prob*100).toFixed(1)}%</td>
      <td class="neu">${t.bars}</td>
      <td class="neu" style="font-size:10px">F${t.fold}</td>
    `;
    tbody.appendChild(tr);
  });

  // Fold-end banner if last trade of a fold is in this page
  if(slice.length){
    const lastFold=slice[slice.length-1].fold;
    const nextTrade=filtered[start+slice.length];
    if(!nextTrade||nextTrade.fold!==lastFold){
      const fi=FOLD_INFO[lastFold];
      const pnl=fi.end_bal-200;
      const isL=LOSING_FOLDS.has(lastFold);
      const er=document.createElement('tr');
      er.className='fold-end-banner'+(isL?' losing':'');
      er.innerHTML=`<td colspan="14">
        ◀ END F${lastFold} &nbsp;|&nbsp; Final Balance <strong>$${fi.end_bal.toLocaleString('en',{minimumFractionDigits:2})}</strong>
        &nbsp;|&nbsp; P&amp;L <strong>${pnl>=0?'+':''}$${Math.abs(pnl).toFixed(2)}</strong>
        &nbsp;|&nbsp; ${isL?'❌ LOSING FOLD':'✅ WIN FOLD'}
      </td>`;
      tbody.appendChild(er);
    }
  }
}

function renderPages(){
  const total=Math.ceil(filtered.length/PAGE_SIZE);
  const bar=document.getElementById('pagesBar');
  bar.innerHTML='';
  const info=document.createElement('span');
  info.className='page-info';
  info.textContent=`${filtered.length.toLocaleString()} trades  |  Page ${page+1} / ${total}  |  `;
  bar.appendChild(info);

  const addBtn=(label,pi)=>{
    const b=document.createElement('button');
    b.className='btn'+(pi===page?' active':'');
    b.textContent=label;
    b.onclick=()=>{page=pi;renderTable();renderPages();window.scrollTo(0,350)};
    bar.appendChild(b);
  };
  if(page>0) addBtn('◀ Prev',page-1);
  const s=Math.max(0,page-4),e=Math.min(total,s+10);
  for(let i=s;i<e;i++) addBtn(i+1,i);
  if(page<total-1) addBtn('Next ▶',page+1);
}

['foldSel','sideSel','resultSel','regimeSel'].forEach(id=>{
  document.getElementById(id).addEventListener('change',applyFilters);
});
document.getElementById('searchInput').addEventListener('input',applyFilters);

applyFilters();
window.addEventListener('resize',()=>{if(filtered.length)renderChart()});
</script>
</body>
</html>"""

# Inject data
HTML = HTML.replace("TRADES_PLACEHOLDER", trades_js)
HTML = HTML.replace("FOLD_INFO_PLACEHOLDER", folds_js)
HTML = HTML.replace("LOSING_PLACEHOLDER", losing_js)

OUT_PATH.write_text(HTML, encoding="utf-8")
size_kb = OUT_PATH.stat().st_size / 1024
print(f"\nWritten: {OUT_PATH}")
print(f"Size: {size_kb:.1f} KB")
