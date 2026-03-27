"""Write upgraded dashboard HTML to correct path."""
import pathlib

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>XAU/USD Trade Monitor</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" crossorigin="anonymous"/>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    :root{
      --bg:#0d1117;--card:#161b22;--card2:#1c2129;--bdr:#30363d;
      --accent:#58a6ff;--green:#3fb950;--red:#f85149;--yellow:#d29922;
      --purple:#bc8cff;--orange:#ffa657;--text:#e6edf3;--muted:#8b949e;
    }
    *{box-sizing:border-box}
    body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;margin:0}
    /* ─── Account tabs ─── */
    .acct-tabs{display:flex;gap:4px;border-bottom:1px solid var(--bdr);padding:12px 16px 0;background:var(--card)}
    .acct-tab{padding:7px 20px;border-radius:6px 6px 0 0;border:1px solid transparent;cursor:pointer;font-weight:600;font-size:.82rem;color:var(--muted);background:transparent;transition:all .15s}
    .acct-tab.active{background:var(--bg);color:var(--accent);border-color:var(--bdr) var(--bdr) var(--bg)}
    .acct-tab:hover:not(.active){color:var(--text);background:rgba(255,255,255,.04)}
    /* ─── Section tabs ─── */
    .sec-tabs{display:flex;gap:2px;padding:10px 16px 0;border-bottom:1px solid var(--bdr);background:var(--card)}
    .sec-tab{padding:5px 14px;border-radius:5px 5px 0 0;cursor:pointer;font-size:.78rem;font-weight:500;color:var(--muted);border:1px solid transparent;background:transparent;transition:all .15s}
    .sec-tab.active{background:var(--bg);color:var(--text);border-color:var(--bdr) var(--bdr) var(--bg)}
    .sec-tab:hover:not(.active){color:var(--text);background:rgba(255,255,255,.04)}
    /* ─── Cards / layout ─── */
    .section-body{padding:14px 16px}
    .kcard{background:var(--card);border:1px solid var(--bdr);border-radius:8px;padding:12px 14px;margin-bottom:0}
    .kcard2{background:var(--card2);border:1px solid var(--bdr);border-radius:6px;padding:10px 12px}
    .slabel{font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:2px}
    .sval{font-size:1.3rem;font-weight:700;line-height:1.15}
    .ssub{font-size:.68rem;color:var(--muted);margin-top:2px}
    .sval-sm{font-size:.97rem;font-weight:700}
    /* ─── Badges ─── */
    .b-buy{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.66rem;font-weight:700;background:rgba(63,185,80,.15);color:var(--green);border:1px solid var(--green)}
    .b-sell{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.66rem;font-weight:700;background:rgba(248,81,73,.15);color:var(--red);border:1px solid var(--red)}
    .b-none{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.66rem;font-weight:700;background:rgba(139,148,158,.1);color:var(--muted);border:1px solid var(--bdr)}
    .b-ok{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.7rem;font-weight:700;background:rgba(63,185,80,.12);color:var(--green);border:1px solid var(--green)}
    .b-warn{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.7rem;font-weight:700;background:rgba(248,81,73,.12);color:var(--red);border:1px solid var(--red)}
    /* ─── Confidence bar ─── */
    .cbar-wrap{background:var(--bdr);border-radius:3px;height:9px;position:relative;overflow:hidden;margin-top:5px}
    .cbar{height:100%;border-radius:3px;transition:width .4s}
    .cbar-thresh{position:absolute;top:0;width:2px;height:100%;background:var(--yellow)}
    /* ─── Tables ─── */
    .tbl{width:100%;border-collapse:collapse;font-size:.76rem}
    .tbl th{color:var(--muted);font-weight:500;padding:5px 8px;border-bottom:1px solid var(--bdr);white-space:nowrap}
    .tbl td{padding:5px 8px;border-bottom:1px solid rgba(48,54,61,.5);vertical-align:middle}
    .tbl tr:last-child td{border-bottom:none}
    .tbl tr:hover td{background:rgba(88,166,255,.03)}
    .win{color:var(--green)}.loss{color:var(--red)}.muted{color:var(--muted)}
    /* ─── Signal big card ─── */
    .sig-big{display:flex;align-items:flex-start;gap:16px;padding:4px 0}
    .sig-side-big{font-size:2.2rem;font-weight:900;line-height:1;min-width:90px}
    .sig-side-big.buy{color:var(--green)}.sig-side-big.sell{color:var(--red)}.sig-side-big.none{color:var(--muted)}
    .sig-details{flex:1;min-width:0}
    /* ─── Grid helpers ─── */
    .g2{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
    .g3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
    .g4{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
    .g6{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}
    @media(max-width:900px){.g6{grid-template-columns:repeat(3,1fr)}.g4{grid-template-columns:repeat(2,1fr)}}
    @media(max-width:500px){.g6{grid-template-columns:repeat(2,1fr)}.g4{grid-template-columns:repeat(2,1fr)}}
    /* ─── Misc ─── */
    .scroll-y{overflow-y:auto;max-height:360px}
    .chart-wrap{position:relative;height:240px}
    .chart-wrap-lg{position:relative;height:300px}
    #conn-badge{font-size:.68rem;padding:2px 9px;border-radius:12px;margin-left:10px}
    .conn-live{background:rgba(63,185,80,.15);color:var(--green);border:1px solid var(--green)}
    .conn-dead{background:rgba(248,81,73,.15);color:var(--red);border:1px solid var(--red)}
    .hdr{display:flex;align-items:center;padding:10px 16px;background:var(--card);border-bottom:1px solid var(--bdr);gap:8px}
    .pulse{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);animation:pulse 1.5s infinite}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
    .sec-ttl{font-size:.72rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px}
    .fold-ok td:first-child{border-left:3px solid var(--green)}
    .fold-bad td:first-child{border-left:3px solid var(--red)}
    .fold-neu td:first-child{border-left:3px solid var(--yellow)}
  </style>
</head>
<body>

<div class="hdr">
  <div class="pulse" id="dot"></div>
  <span style="font-weight:700;font-size:.95rem">XAU/USD Trade Monitor</span>
  <span id="conn-badge" class="conn-dead">&#9679; CONNECTING</span>
  <span style="margin-left:auto;font-size:.68rem;color:var(--muted)">Updated: <span id="last-ts">&#8212;</span></span>
</div>

<div class="acct-tabs" id="acct-tabs">
  <button class="acct-tab active" data-acct="acc1">ACC1 &middot; ICT+Wyckoff</button>
  <button class="acct-tab"        data-acct="acc2">ACC2 &middot; Model v2</button>
</div>

<div id="ap-acc1">
  <div class="sec-tabs" id="st-acc1">
    <button class="sec-tab active" data-acct="acc1" data-sec="overview">&#128202; Overview</button>
    <button class="sec-tab"        data-acct="acc1" data-sec="trading">&#9889; Live</button>
    <button class="sec-tab"        data-acct="acc1" data-sec="pnl">&#128176; P&amp;L</button>
    <button class="sec-tab"        data-acct="acc1" data-sec="backtest">&#128300; Backtest</button>
    <button class="sec-tab"        data-acct="acc1" data-sec="walkforward">&#128260; WalkForward</button>
  </div>
  <div id="sb-acc1-overview"    class="section-body"></div>
  <div id="sb-acc1-trading"     class="section-body" style="display:none"></div>
  <div id="sb-acc1-pnl"         class="section-body" style="display:none"></div>
  <div id="sb-acc1-backtest"    class="section-body" style="display:none"></div>
  <div id="sb-acc1-walkforward" class="section-body" style="display:none"></div>
</div>

<div id="ap-acc2" style="display:none">
  <div class="sec-tabs" id="st-acc2">
    <button class="sec-tab active" data-acct="acc2" data-sec="overview">&#128202; Overview</button>
    <button class="sec-tab"        data-acct="acc2" data-sec="trading">&#9889; Live</button>
    <button class="sec-tab"        data-acct="acc2" data-sec="pnl">&#128176; P&amp;L</button>
    <button class="sec-tab"        data-acct="acc2" data-sec="backtest">&#128300; Backtest</button>
    <button class="sec-tab"        data-acct="acc2" data-sec="walkforward">&#128260; WalkForward</button>
  </div>
  <div id="sb-acc2-overview"    class="section-body"></div>
  <div id="sb-acc2-trading"     class="section-body" style="display:none"></div>
  <div id="sb-acc2-pnl"         class="section-body" style="display:none"></div>
  <div id="sb-acc2-backtest"    class="section-body" style="display:none"></div>
  <div id="sb-acc2-walkforward" class="section-body" style="display:none"></div>
</div>

<script>
/* ─ helpers ─ */
const el  = id => document.getElementById(id);
const fmt = (v,d=2) => (v==null||v==='')?'&#8212;':(+v).toFixed(d);
const pct = v => v==null?'&#8212;':(+v*100).toFixed(1)+'%';
const pctR= v => v==null?'&#8212;':(+v).toFixed(1)+'%';
const dol = v => { if(v==null) return '&#8212;'; const n=+v; return (n>=0?'$':'-$')+Math.abs(n).toFixed(2); };
const q   = v => (v==null||v===''||v===undefined)?'&#8212;':v;
const clr = v => v==null?'':(+v>0?'win':+v<0?'loss':'');

function badge(side){
  if(!side) return '<span class="b-none">&#8212;</span>';
  const s=String(side).toUpperCase();
  if(s==='BUY'||s==='LONG')  return '<span class="b-buy">&#9650; BUY</span>';
  if(s==='SELL'||s==='SHORT') return '<span class="b-sell">&#9660; SELL</span>';
  return `<span class="b-none">${s}</span>`;
}

function confBar(conf, thresh){
  const p=Math.min(100,Math.max(0,(+(conf||0))*100));
  const tp=Math.min(100,Math.max(0,(+(thresh||0))*100));
  const over=conf!=null&&thresh!=null&&+conf>=+thresh;
  const c=over?'var(--green)':'var(--red)';
  return `<div style="display:flex;justify-content:space-between;font-size:.7rem">
    <span style="font-weight:600">${fmt(conf,3)}</span>
    <span class="muted">threshold ${fmt(thresh,3)}</span></div>
  <div class="cbar-wrap">
    <div class="cbar" style="width:${p}%;background:${c}"></div>
    <div class="cbar-thresh" style="left:${tp}%"></div>
  </div>`;
}

function statBox(lbl,val,cls='',sub=''){
  return `<div class="kcard2"><div class="slabel">${lbl}</div><div class="sval-sm ${cls}">${val}</div>${sub?`<div class="ssub">${sub}</div>`:''}</div>`;
}

/* ─ charts ─ */
const charts={};
Chart.defaults.color='#8b949e';

function renderLine(id,labels,values,lbl,color){
  const c=el(id); if(!c)return;
  if(charts[id]){charts[id].destroy();delete charts[id];}
  const n=labels.length;
  charts[id]=new Chart(c.getContext('2d'),{
    type:'line',
    data:{labels,datasets:[{label:lbl,data:values,borderColor:color,backgroundColor:color+'1a',fill:true,tension:0.3,pointRadius:n>80?0:n>30?1:2,borderWidth:2}]},
    options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
      plugins:{legend:{labels:{color:'#8b949e',boxWidth:10,font:{size:10}}}},
      scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:8,font:{size:10},maxRotation:0},grid:{color:'#1c2129'}},
              y:{ticks:{color:'#8b949e',font:{size:10}},grid:{color:'#1c2129'}}}}
  });
}

function renderBars(id,labels,values){
  const c=el(id); if(!c)return;
  if(charts[id]){charts[id].destroy();delete charts[id];}
  charts[id]=new Chart(c.getContext('2d'),{
    type:'bar',
    data:{labels,datasets:[{label:'PnL',data:values,
      backgroundColor:values.map(v=>v>=0?'rgba(63,185,80,.5)':'rgba(248,81,73,.5)'),
      borderColor:values.map(v=>v>=0?'#3fb950':'#f85149'),borderWidth:1}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:8,font:{size:9},maxRotation:0},grid:{color:'#1c2129'}},
              y:{ticks:{color:'#8b949e',font:{size:10}},grid:{color:'#1c2129'}}}}
  });
}

/* ─ section renderers ─ */
function renderOverview(acct,d){
  const s=d.status||{};const mod=d.model||{};const bt=d.backtest||{};const day=d.daily||{};
  const trades=d.recent_trades||[];const sig=(d.recent_signals||[]).slice(-1)[0]||{};
  const cbActive=day.killed||day.cooldown_bars>0;
  const cbTxt=day.killed?'KILL SWITCH':day.cooldown_bars>0?`Cooldown ${day.cooldown_bars} bars`:'NORMAL';
  const liveSide=s.side;const liveConf=s.confidence;const shouldTrade=s.should_trade;
  const sigSide=sig.direction??sig.signal??sig.side??null;
  const sigConf=sig.confidence??sig.conf??null;

  const miniRows=trades.slice(-5).reverse().map(t=>{
    const pnl=t.pnl??t.profit??null;
    return `<tr><td class="muted">${String(t.time||t.ts||'').slice(5,16)}</td>
      <td>${badge(t.side||t.direction||'')}</td>
      <td class="${clr(pnl)}">${dol(pnl)}</td>
      <td class="muted">${q(t.close_type||t.reason||'')}</td></tr>`;
  }).join('');

  el(`sb-${acct}-overview`).innerHTML=`
  <div class="g6" style="margin-bottom:12px">
    <div class="kcard2"><div class="slabel">Balance</div><div class="sval ${(d.total_pnl||0)>=0?'win':'loss'}">${dol(s.account_balance)}</div><div class="ssub">Peak ${dol(d.peak_balance)}</div></div>
    <div class="kcard2"><div class="slabel">Total Live PnL</div><div class="sval ${(d.total_pnl||0)>=0?'win':'loss'}">${dol(d.total_pnl)}</div><div class="ssub">${trades.length} trades</div></div>
    <div class="kcard2"><div class="slabel">Win Rate</div><div class="sval">${pct(d.win_rate)}</div><div class="ssub">last ${trades.length} trades</div></div>
    <div class="kcard2"><div class="slabel">Drawdown</div><div class="sval ${(d.drawdown_pct||0)>5?'loss':''}">${d.drawdown_pct!=null?d.drawdown_pct+'%':'&#8212;'}</div><div class="ssub">from peak</div></div>
    <div class="kcard2"><div class="slabel">Open Positions</div><div class="sval">${s.open_positions??'0'}<span style="font-size:.8rem;font-weight:400;color:var(--muted)"> / ${s.max_positions??'&#8212;'}</span></div><div class="ssub">${s.volatility_regime!=null?'Vol regime '+s.volatility_regime:''}</div></div>
    <div class="kcard2"><div class="slabel">Circuit Breaker</div><div class="sval" style="font-size:.85rem;padding-top:3px"><span class="${cbActive?'b-warn':'b-ok'}">${cbTxt}</span></div><div class="ssub">${day.consecutive_losses||0} consec losses</div></div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">
    <div class="kcard">
      <div class="sec-ttl">Current Live Signal</div>
      <div class="sig-big">
        <div class="sig-side-big ${liveSide?liveSide.toLowerCase():'none'}">${liveSide?( liveSide.toLowerCase()==='buy'?'&#9650; BUY':'&#9660; SELL'):'&#8212;'}</div>
        <div class="sig-details">
          ${confBar(liveConf, mod.threshold)}
          <div style="margin-top:7px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            <span class="${shouldTrade?'b-ok':'b-warn'}" style="font-size:.66rem">${shouldTrade?'TRADING':'BLOCKED'}</span>
            <span class="muted" style="font-size:.68rem">${s.bar_time?String(s.bar_time).slice(0,16):''}</span>
          </div>
          <div style="font-size:.72rem;color:var(--muted);margin-top:5px;line-height:1.4">${q(s.reason||'')}</div>
        </div>
      </div>
      <div style="border-top:1px solid var(--bdr);margin-top:10px;padding-top:8px;font-size:.71rem;color:var(--muted)">
        Latest paper signal: ${String(sig.time||sig.ts||'').slice(0,16)} &nbsp;${badge(sigSide)}&nbsp; conf ${fmt(sigConf,3)}
      </div>
    </div>

    <div class="kcard">
      <div class="sec-ttl">Model &amp; Backtest Snapshot</div>
      <div class="g4" style="margin-bottom:8px">
        ${statBox('Precision', pct(mod.precision))}
        ${statBox('Threshold', fmt(mod.threshold,3))}
        ${statBox('ROC-AUC', fmt(mod.roc_auc,3))}
        ${statBox('WF Prec avg', pct(mod.wf_precision_avg))}
      </div>
      <div class="g4">
        ${statBox('BT Return', pctR(bt.return_pct), (bt.return_pct||0)>=0?'win':'loss')}
        ${statBox('Profit Factor', fmt(bt.profit_factor))}
        ${statBox('BT Win Rate', pct(bt.win_rate))}
        ${statBox('BT Max DD', pctR(bt.max_drawdown_pct), 'loss')}
      </div>
    </div>
  </div>

  <div class="kcard">
    <div class="sec-ttl">Recent Trades (last 5)</div>
    <table class="tbl"><thead><tr><th>Time</th><th>Side</th><th>PnL</th><th>Close Type</th></tr></thead>
    <tbody>${miniRows||'<tr><td colspan="4" class="muted" style="text-align:center;padding:14px">No trades yet</td></tr>'}</tbody></table>
  </div>`;
}

function renderTrading(acct,d){
  const s=d.status||{};const mod=d.model||{};
  const trades=d.recent_trades||[];const signals=d.recent_signals||[];

  const trRows=trades.slice().reverse().map(t=>{
    const pnl=t.pnl??t.profit??null;
    return `<tr><td class="muted">${String(t.time||t.close_time||t.ts||'').slice(0,16)}</td>
      <td>${badge(t.side||t.direction||'')}</td>
      <td class="muted">${q(t.open_price||t.entry_price||'')}</td>
      <td class="muted">${q(t.close_price||t.exit_price||'')}</td>
      <td class="${clr(pnl)}">${dol(pnl)}</td>
      <td class="muted">${q(t.close_type||t.exit_reason||'')}</td>
      <td class="muted" style="font-size:.68rem">${q(t.ticket||t.session_id||'')}</td>
    </tr>`;
  }).join('');

  const sgRows=signals.slice().reverse().map(sg=>{
    const c=sg.confidence??sg.conf??null;
    const thr=mod.threshold;
    const over=c!=null&&thr!=null?+c>=+thr:null;
    return `<tr><td class="muted">${String(sg.time||sg.ts||sg.timestamp||'').slice(0,16)}</td>
      <td>${badge(sg.direction||sg.signal||sg.side||'')}</td>
      <td class="${over===true?'win':over===false?'loss':''}">${fmt(c,3)}</td>
      <td class="muted">${q(sg.regime||sg.market_regime||sg.volatility_regime||'')}</td>
      <td class="muted" style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${q((sg.reason||sg.signal_reason||'').slice(0,80))}</td>
    </tr>`;
  }).join('');

  el(`sb-${acct}-trading`).innerHTML=`
  <div class="kcard" style="margin-bottom:12px">
    <div class="sec-ttl">Live Status</div>
    <div class="g4" style="margin-bottom:8px">
      ${statBox('Bar Time', String(s.bar_time||'&#8212;').slice(0,16))}
      ${statBox('Balance', dol(s.account_balance))}
      ${statBox('Open / Max Pos', (s.open_positions??'&#8212;')+' / '+(s.max_positions??'&#8212;'))}
      ${statBox('Vol Regime', s.volatility_regime??'&#8212;')}
    </div>
    <div class="g4">
      ${statBox('Confidence', fmt(s.confidence,3), s.should_trade?'win':'loss')}
      ${statBox('Threshold', fmt(mod.threshold,3))}
      ${statBox('Side', badge(s.side))}
      ${statBox('Should Trade', `<span class="${s.should_trade?'b-ok':'b-warn'}">${s.should_trade?'YES':'NO'}</span>`)}
    </div>
    <div style="margin-top:8px;font-size:.74rem;color:var(--muted)">Reason: ${q(s.reason)}</div>
  </div>

  <div style="display:grid;grid-template-columns:5fr 7fr;gap:10px">
    <div class="kcard">
      <div class="sec-ttl">Recent Signals (${signals.length})</div>
      <div class="scroll-y"><table class="tbl">
        <thead><tr><th>Time</th><th>Side</th><th>Conf</th><th>Regime</th><th>Reason</th></tr></thead>
        <tbody>${sgRows||'<tr><td colspan="5" class="muted" style="text-align:center;padding:14px">No signals yet</td></tr>'}</tbody>
      </table></div>
    </div>
    <div class="kcard">
      <div class="sec-ttl">Live Trades (${trades.length})</div>
      <div class="scroll-y"><table class="tbl">
        <thead><tr><th>Time</th><th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Type</th><th>ID</th></tr></thead>
        <tbody>${trRows||'<tr><td colspan="7" class="muted" style="text-align:center;padding:14px">No trades yet</td></tr>'}</tbody>
      </table></div>
    </div>
  </div>`;
}

function renderPnl(acct,d){
  const trades=d.recent_trades||[];
  const series=d.pnl_series||{labels:[],values:[],individual:[]};
  const W=trades.filter(t=>String(t.is_win||'').toLowerCase()==='true'||t.is_win===true||t.is_win===1||t.is_win==='1').length;
  const L=trades.length-W;
  const wTrades=trades.filter(t=>String(t.is_win||'').toLowerCase()==='true'||t.is_win===true);
  const lTrades=trades.filter(t=>+(t.pnl??t.profit??0)<0);
  const avgWin=W>0?wTrades.reduce((a,t)=>a+parseFloat(t.pnl??t.profit??0),0)/W:0;
  const avgLoss=L>0&&lTrades.length>0?lTrades.reduce((a,t)=>a+parseFloat(t.pnl??t.profit??0),0)/lTrades.length:0;

  const trRows=trades.slice().reverse().map((t,i)=>{
    const pnl=t.pnl??t.profit??null;
    const cumIdx=series.values.length-1-(trades.length-1-i);
    const cum=series.values[series.values.length-1-i];
    return `<tr><td class="muted">${String(t.time||t.close_time||'').slice(0,16)}</td>
      <td>${badge(t.side||t.direction||'')}</td>
      <td class="${clr(pnl)}">${dol(pnl)}</td>
      <td class="${(cum??0)>=0?'win':'loss'}">${dol(cum)}</td>
      <td class="muted">${q(t.close_type||'')}</td></tr>`;
  }).join('');

  el(`sb-${acct}-pnl`).innerHTML=`
  <div class="g6" style="margin-bottom:12px">
    ${statBox('Total PnL', dol(d.total_pnl), (d.total_pnl||0)>=0?'win':'loss')}
    ${statBox('Trades', trades.length)}
    ${statBox('Wins / Losses', W+' <span class="muted">/</span> '+L)}
    ${statBox('Win Rate', pct(d.win_rate))}
    ${statBox('Avg Win', '$'+avgWin.toFixed(2), 'win')}
    ${statBox('Avg Loss', '$'+Math.abs(avgLoss).toFixed(2), 'loss')}
  </div>
  <div class="kcard" style="margin-bottom:12px">
    <div class="sec-ttl">Live Equity Curve</div>
    <div class="chart-wrap-lg"><canvas id="ch-${acct}-pnl"></canvas></div>
  </div>
  <div class="kcard" style="margin-bottom:12px">
    <div class="sec-ttl">Per-Trade PnL</div>
    <div class="chart-wrap"><canvas id="ch-${acct}-ptrade"></canvas></div>
  </div>
  <div class="kcard">
    <div class="sec-ttl">Trade Log (${trades.length} trades)</div>
    <div class="scroll-y"><table class="tbl">
      <thead><tr><th>Time</th><th>Side</th><th>PnL</th><th>Cumulative</th><th>Close Type</th></tr></thead>
      <tbody>${trRows||'<tr><td colspan="5" class="muted" style="text-align:center;padding:14px">No trades yet</td></tr>'}</tbody>
    </table></div>
  </div>`;
  requestAnimationFrame(()=>{
    renderLine(`ch-${acct}-pnl`,series.labels,series.values,'Cumulative PnL','#58a6ff');
    renderBars(`ch-${acct}-ptrade`,series.labels,series.individual);
  });
}

function renderBacktest(acct,d){
  const bt=d.backtest||{};
  const eq=d.backtest_equity||{labels:[],values:[],individual:[]};
  el(`sb-${acct}-backtest`).innerHTML=`
  <div class="kcard" style="margin-bottom:12px">
    <div class="sec-ttl">Backtest Report &nbsp;<span class="muted" style="font-size:.68rem;text-transform:none">${String(bt.trade_start||'').slice(0,10)} &#8594; ${String(bt.trade_end||'').slice(0,10)} &nbsp;&#183;&nbsp; ${bt.trade_days??'?'} days</span></div>
    <div class="g4" style="margin-bottom:8px">
      ${statBox('Starting Balance', dol(bt.starting_balance))}
      ${statBox('Ending Balance',   dol(bt.ending_balance), 'win')}
      ${statBox('Net Profit', dol(bt.net_profit), (bt.net_profit||0)>=0?'win':'loss')}
      ${statBox('Return', pctR(bt.return_pct), (bt.return_pct||0)>=0?'win':'loss')}
    </div>
    <div class="g4" style="margin-bottom:8px">
      ${statBox('Win Rate', pct(bt.win_rate))}
      ${statBox('Profit Factor', fmt(bt.profit_factor))}
      ${statBox('Sharpe-like', fmt(bt.sharpe_like))}
      ${statBox('Max Drawdown', pctR(bt.max_drawdown_pct), 'loss')}
    </div>
    <div class="g4" style="margin-bottom:8px">
      ${statBox('Trades', bt.trades??'&#8212;')}
      ${statBox('Wins', bt.wins??'&#8212;', 'win')}
      ${statBox('Losses', bt.losses??'&#8212;', 'loss')}
      ${statBox('Draws', bt.draws??0)}
    </div>
    <div class="g4" style="margin-bottom:8px">
      ${statBox('Avg Win', dol(bt.avg_win), 'win')}
      ${statBox('Avg Loss', dol(bt.avg_loss), 'loss')}
      ${statBox('Best Trade', dol(bt.best_trade), 'win')}
      ${statBox('Worst Trade', dol(bt.worst_trade), 'loss')}
    </div>
    <div class="g4">
      ${statBox('Avg Hold Bars', fmt(bt.avg_holding_bars,1))}
      ${statBox('Train Days', bt.train_days??'&#8212;')}
      ${statBox('Compound Cap', dol(bt.compound_cap))}
      ${statBox('Friction RR', fmt(bt.friction_rr,2))}
    </div>
  </div>
  <div class="kcard" style="margin-bottom:12px">
    <div class="sec-ttl">Backtest Equity Curve (${eq.labels.length} trades)</div>
    <div class="chart-wrap-lg"><canvas id="ch-${acct}-bteq"></canvas></div>
  </div>
  <div class="kcard">
    <div class="sec-ttl">Backtest Per-Trade PnL</div>
    <div class="chart-wrap"><canvas id="ch-${acct}-btbar"></canvas></div>
  </div>`;
  requestAnimationFrame(()=>{
    renderLine(`ch-${acct}-bteq`,eq.labels,eq.values,'Equity','#d29922');
    renderBars(`ch-${acct}-btbar`,eq.labels,eq.individual);
  });
}

function renderWalkforward(acct,d){
  const wf=d.walkforward||{};const agg=wf.aggregate||{};const cfg=wf.config||{};const folds=wf.folds||[];
  const sim=agg.concurrent_sim||{};

  const fRows=folds.map(f=>{
    const rp=f.return_pct;const cls=rp>=0?'fold-ok':rp<-10?'fold-bad':'fold-neu';
    return `<tr class="${cls}"><td style="font-weight:600">${f.fold}</td>
      <td class="muted" style="font-size:.72rem">${f.train_start}&#8594;${f.train_end}</td>
      <td class="muted" style="font-size:.72rem">${f.test_start}&#8594;${f.test_end}</td>
      <td>${fmt(f.threshold,3)}</td>
      <td>${pct(f.precision)}</td>
      <td>${fmt(f.roc_auc,4)}</td>
      <td class="muted">${f.n_signals??'&#8212;'}</td>
      <td class="${f.win_rate>=0.5?'win':'loss'}">${pct(f.win_rate)}</td>
      <td>${fmt(f.profit_factor,2)}</td>
      <td class="${clr(rp)}">${rp!=null?(+rp).toFixed(1)+'%':'&#8212;'}</td>
      <td class="loss">${f.max_dd!=null?(+f.max_dd).toFixed(1)+'%':'&#8212;'}</td>
      <td class="muted">${f.trades??'&#8212;'}</td>
    </tr>`;
  }).join('');

  el(`sb-${acct}-walkforward`).innerHTML=`
  <div class="kcard" style="margin-bottom:12px">
    <div class="sec-ttl">Walk-Forward Configuration</div>
    <div class="g4">
      ${statBox('Folds', cfg.n_folds??folds.length)}
      ${statBox('Train Bars', cfg.train_bars??'&#8212;')}
      ${statBox('Test Bars', cfg.test_bars??'&#8212;')}
      <div class="kcard2"><div class="slabel">Model</div><div style="font-size:.68rem;color:var(--muted);margin-top:2px;line-height:1.4">${q(cfg.model)}</div></div>
    </div>
  </div>

  <div class="kcard" style="margin-bottom:12px">
    <div class="sec-ttl">Aggregate Metrics (all folds)</div>
    <div class="g4" style="margin-bottom:8px">
      ${statBox('Avg ROC-AUC', fmt(agg.avg_roc_auc,4), '', '&#963;='+fmt(agg.std_roc_auc,4))}
      ${statBox('Avg Precision', pct(agg.avg_precision), '', '&#963;='+pct(agg.std_precision))}
      ${statBox('Signal Win Rate', pct(agg.signal_win_rate), agg.signal_win_rate>=0.5?'win':'loss')}
      ${statBox('Total Signals', agg.total_signals??'&#8212;')}
    </div>
    <div class="sec-ttl">Concurrent Simulation Average</div>
    <div class="g4">
      ${statBox('Avg Win Rate', pct(sim.avg_win_rate), (sim.avg_win_rate||0)>=0.5?'win':'loss')}
      ${statBox('Avg Profit Factor', fmt(sim.avg_profit_factor,3))}
      ${statBox('Avg Return', (sim.avg_return_pct!=null?(+sim.avg_return_pct).toFixed(1)+'%':'&#8212;'), (sim.avg_return_pct||0)>=0?'win':'loss')}
      ${statBox('Avg Max DD', (sim.avg_max_drawdown_pct!=null?(+sim.avg_max_drawdown_pct).toFixed(1)+'%':'&#8212;'), 'loss')}
    </div>
  </div>

  <div class="kcard">
    <div class="sec-ttl">Per-Fold Results (${folds.length} folds &nbsp;&#8212;&nbsp; green border = profitable)</div>
    <div style="overflow-x:auto"><table class="tbl">
      <thead><tr><th>Fold</th><th>Train</th><th>Test</th><th>Thresh</th><th>Prec</th><th>ROC-AUC</th><th>Signals</th><th>Win%</th><th>PF</th><th>Return</th><th>Max DD</th><th>Trades</th></tr></thead>
      <tbody>${fRows||'<tr><td colspan="12" class="muted" style="text-align:center;padding:14px">No fold data</td></tr>'}</tbody>
    </table></div>
  </div>`;
}

/* ─ state & routing ─ */
const State={data:null,acct:'acc1',sec:{acc1:'overview',acc2:'overview'}};

function renderSection(acct,sec){
  const d=State.data?.accounts?.[acct]; if(!d)return;
  State.sec[acct]=sec;
  ['overview','trading','pnl','backtest','walkforward'].forEach(s=>{
    const div=el(`sb-${acct}-${s}`); if(div) div.style.display=s===sec?'':'none';
  });
  document.querySelectorAll(`#st-${acct} .sec-tab`).forEach(b=>b.classList.toggle('active',b.dataset.sec===sec));
  if(sec==='overview')    renderOverview(acct,d);
  if(sec==='trading')     renderTrading(acct,d);
  if(sec==='pnl')         renderPnl(acct,d);
  if(sec==='backtest')    renderBacktest(acct,d);
  if(sec==='walkforward') renderWalkforward(acct,d);
}

function updateUI(data){
  State.data=data;
  el('last-ts').textContent=(data.ts||'').replace('T',' ').slice(0,19)+' UTC';
  ['acc1','acc2'].forEach(a=>{ if((data.accounts||{})[a]) renderSection(a,State.sec[a]); });
}

/* account tab switch */
document.querySelectorAll('.acct-tab').forEach(btn=>{
  btn.addEventListener('click',()=>{
    State.acct=btn.dataset.acct;
    document.querySelectorAll('.acct-tab').forEach(b=>b.classList.toggle('active',b===btn));
    ['acc1','acc2'].forEach(a=>{ const p=el(`ap-${a}`); if(p) p.style.display=a===State.acct?'':'none'; });
  });
});

/* section tab switch */
document.addEventListener('click',e=>{
  const btn=e.target.closest('.sec-tab'); if(!btn)return;
  renderSection(btn.dataset.acct,btn.dataset.sec);
});

/* WebSocket */
(function connectWS(){
  const proto=location.protocol==='https:'?'wss':'ws';
  const ws=new WebSocket(`${proto}://${location.host}/ws/live`);
  const badge=el('conn-badge'); const dot=el('dot');
  ws.onopen=()=>{ badge.textContent='&#9679; LIVE'; badge.className='conn-live'; dot.style.display=''; };
  ws.onmessage=e=>{ try{ const data=JSON.parse(e.data); if(data.ping)return; updateUI(data); }catch(_){} };
  let delay=1500;
  ws.onclose=ws.onerror=()=>{
    badge.textContent='&#9679; RECONNECTING'; badge.className='conn-dead'; dot.style.display='none';
    setTimeout(()=>{ delay=Math.min(delay*2,30000); connectWS(); },delay);
  };
})();
</script>
</body>
</html>'''

p = pathlib.Path(r'C:\Users\Administrator\Documents\Trade-Indicator\src\xauusd_ai\dashboard\static\index.html')
p.write_text(HTML, encoding='utf-8')
print(f"Written {p.stat().st_size} bytes, {len(HTML.splitlines())} lines")
