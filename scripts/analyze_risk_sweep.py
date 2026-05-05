"""Analyze risk-pct sweep results and compare configs."""
import json, re, sys
from pathlib import Path

BASE = Path(__file__).parent.parent

def parse_fold_log(filepath):
    """Parse PnL from WF text log file."""
    try:
        lines = Path(filepath).read_text().splitlines()
    except FileNotFoundError:
        return None, 0
    bal_lines = [l for l in lines if 'Bal:' in l and '→' in l]
    if not bal_lines:
        return None, 0
    total_pnl = 0
    for l in bal_lines:
        m = re.search(r'→\$([0-9,]+\.?[0-9]*)', l)
        if m:
            total_pnl += float(m.group(1).replace(',', '')) - 200
    return total_pnl, len(bal_lines)

def load_json_report(filepath):
    """Load JSON report for DD stats."""
    try:
        with open(filepath) as f:
            r = json.load(f)
        folds = r.get('folds', [])
        if not folds:
            return None
        total_pnl = sum(f['concurrent_sim']['ending_balance'] - f['concurrent_sim']['starting_balance'] for f in folds)
        worst_dd = min(f['concurrent_sim']['max_drawdown_pct'] for f in folds)
        avg_dd = r['aggregate']['concurrent_sim']['avg_max_drawdown_pct']
        avg_wr = r['aggregate']['concurrent_sim']['avg_win_rate']
        avg_pf = r['aggregate']['concurrent_sim']['avg_profit_factor']
        trades = sum(f['concurrent_sim']['trades'] for f in folds)
        losing = sum(1 for f in folds if f['concurrent_sim']['ending_balance'] < f['concurrent_sim']['starting_balance'])
        return dict(pnl=total_pnl, avg_dd=avg_dd, worst_dd=worst_dd, wr=avg_wr, pf=avg_pf, trades=trades, losing_folds=losing, folds=len(folds))
    except (FileNotFoundError, KeyError):
        return None

configs = [
    # (label, risk, json_report, text_log)
    ('baseline  risk=3.0%', 0.030, 'outputs/walkforward_report_acc1_v14pp_profit.json', None),
    ('sweep     risk=3.5%', 0.035, None, 'outputs/wf_risk0035_verify.txt'),
    ('sweep     risk=4.0%', 0.040, None, 'outputs/wf_risk0040_verify.txt'),
    ('sweep     risk=4.5%', 0.045, None, 'outputs/wf_risk0045_verify.txt'),
    ('highEV    risk=3.0%', 0.030, 'outputs/walkforward_report_acc1_highEV.json', None),
    ('dd_prot   risk=4.0%', 0.040, 'outputs/walkforward_report_acc1_dd_protected.json', None),
]

print(f"\n{'Config':<25} {'PnL':>12} {'vs_base':>8} {'avgDD':>7} {'worstDD':>9} {'WR':>6} {'PF':>6} {'Trades':>7} {'Loss_F':>6}  DD<20?")
print('─' * 100)

base_pnl = None
for label, risk, json_path, txt_path in configs:
    result = None
    if json_path:
        result = load_json_report(BASE / json_path)
    if result is None and txt_path:
        pnl, n = parse_fold_log(BASE / txt_path)
        if pnl is not None:
            result = dict(pnl=pnl, avg_dd=None, worst_dd=None, wr=None, pf=None, trades=None, losing_folds=None, folds=n)

    if result is None:
        print(f'{label:<25} {"NOT DONE":>12}')
        continue

    if base_pnl is None:
        base_pnl = result['pnl']

    pnl = result['pnl']
    vs = f"+{((pnl/base_pnl)-1)*100:.1f}%" if base_pnl else ''
    avg_dd = f"{result['avg_dd']:.2f}%" if result['avg_dd'] else 'N/A'
    worst_dd = f"{result['worst_dd']:.2f}%" if result['worst_dd'] else 'est.'
    wr = f"{result['wr']:.1%}" if result['wr'] else 'N/A'
    pf = f"{result['pf']:.3f}" if result['pf'] else 'N/A'
    trades = str(result['trades']) if result['trades'] else 'N/A'
    lf = str(result['losing_folds']) if result['losing_folds'] is not None else 'N/A'
    dd_ok = '✓' if result['worst_dd'] and result['worst_dd'] > -20.0 else ('?' if not result['worst_dd'] else '✗')

    print(f'{label:<25} ${pnl:>11,.0f} {vs:>8} {avg_dd:>7} {worst_dd:>9} {wr:>6} {pf:>6} {trades:>7} {lf:>6}  {dd_ok}')

print('─' * 100)
print("DD<20% constraint: worst single-fold drawdown must be > -20%")
print("Baseline worstDD = -17.78% (2.22% buffer before 20% breach)")
