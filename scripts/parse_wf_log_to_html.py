#!/usr/bin/env python3
"""
Parse WF log file and generate HTML report with fold-by-fold results
"""
import re
import sys
from pathlib import Path
from datetime import datetime

def parse_fold_line(line):
    """Parse a fold result line from WF log"""
    # Example: Fold  1/268        Test: 2022-12-01 -> 2023-01-04 | AUC=0.6544  Prec=0.2030  Recall=0.6067  F1=0.3042  Thr=0.60  Sigs=2493/6000  Bal: $500→$744 (+48.7%)  (32.2s)
    pattern = r'Fold\s+(\d+)/(\d+)\s+Test:\s+([\d-]+)\s+->\s+([\d-]+)\s+\|\s+AUC=([\d.]+)\s+Prec=([\d.]+)\s+Recall=([\d.]+)\s+F1=([\d.]+)\s+Thr=([\d.]+)\s+Sigs=(\d+)/(\d+)\s+Bal:\s+\$(\d+)→\$(\d+(?:,\d+)?)\s+\(([+-][\d.]+)%\)'
    
    match = re.search(pattern, line)
    if not match:
        return None
    
    return {
        'fold': int(match.group(1)),
        'total_folds': int(match.group(2)),
        'start_date': match.group(3),
        'end_date': match.group(4),
        'auc': float(match.group(5)),
        'precision': float(match.group(6)),
        'recall': float(match.group(7)),
        'f1': float(match.group(8)),
        'threshold': float(match.group(9)),
        'signals': int(match.group(10)),
        'total_bars': int(match.group(11)),
        'start_balance': int(match.group(12)),
        'end_balance': int(match.group(13).replace(',', '')),
        'pct_change': float(match.group(14))
    }

def generate_html_report(config_name, folds, output_path):
    """Generate HTML report from parsed folds"""
    
    # Calculate cumulative P&L
    cumulative_pnl = []
    total_pnl = 0
    for fold in folds:
        pnl = fold['end_balance'] - fold['start_balance']
        total_pnl += pnl
        cumulative_pnl.append(total_pnl)
    
    # Calculate summary stats
    winning_folds = len([f for f in folds if f['pct_change'] > 0])
    total_folds = len(folds)
    fold_win_rate = winning_folds / total_folds * 100 if total_folds > 0 else 0
    
    avg_pct = sum(f['pct_change'] for f in folds) / total_folds if total_folds > 0 else 0
    avg_auc = sum(f['auc'] for f in folds) / total_folds if total_folds > 0 else 0
    avg_precision = sum(f['precision'] for f in folds) / total_folds if total_folds > 0 else 0
    avg_recall = sum(f['recall'] for f in folds) / total_folds if total_folds > 0 else 0
    
    total_signals = sum(f['signals'] for f in folds)
    
    # Extract config params from name
    parts = config_name.replace('grid_', '').replace('_th', ' th=').replace('_r', ' r=').replace('_mp', ' mp=').replace('_notrail', ' NO-TRAIL').replace('_trail', ' TRAIL').replace('_slip', ' slip=')
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config_name} - Walk-Forward Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .subtitle {{
            color: #666;
            font-size: 1.1em;
            margin-bottom: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        .metric-card h3 {{
            font-size: 0.85em;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        .metric-card .value {{
            font-size: 2.2em;
            font-weight: 700;
        }}
        .chart-container {{
            margin: 40px 0;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }}
        .chart-title {{
            font-size: 1.5em;
            color: #667eea;
            margin-bottom: 20px;
            font-weight: 600;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 30px 0;
            font-size: 0.95em;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-radius: 8px;
            overflow: hidden;
        }}
        thead {{
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 16px 12px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        td {{
            padding: 14px 12px;
            border-bottom: 1px solid #e8e8e8;
        }}
        tbody tr:hover {{
            background: #f8f9fa;
        }}
        tbody tr:nth-child(even) {{
            background: #fafbfc;
        }}
        tbody tr:nth-child(even):hover {{
            background: #f0f2f5;
        }}
        .positive {{
            color: #10b981;
            font-weight: 600;
        }}
        .negative {{
            color: #ef4444;
            font-weight: 600;
        }}
        .neutral {{
            color: #6b7280;
        }}
        .date {{
            color: #6b7280;
            font-size: 0.9em;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-win {{
            background: #d1fae5;
            color: #065f46;
        }}
        .badge-loss {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .note {{
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 20px;
            border-radius: 8px;
            margin: 30px 0;
        }}
        .note h3 {{
            color: #92400e;
            margin-bottom: 10px;
        }}
        .note p {{
            color: #78350f;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>
            <span>📊</span>
            <span>{config_name}</span>
        </h1>
        <div class="subtitle">
            Walk-Forward Validation Report • {parts} • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        
        <div class="summary-grid">
            <div class="metric-card">
                <h3>Total Folds</h3>
                <div class="value">{total_folds}</div>
            </div>
            <div class="metric-card">
                <h3>Fold Win Rate</h3>
                <div class="value">{fold_win_rate:.1f}%</div>
            </div>
            <div class="metric-card">
                <h3>Avg Return/Fold</h3>
                <div class="value">{avg_pct:+.1f}%</div>
            </div>
            <div class="metric-card">
                <h3>Total Signals</h3>
                <div class="value">{total_signals:,}</div>
            </div>
            <div class="metric-card">
                <h3>Avg AUC</h3>
                <div class="value">{avg_auc:.3f}</div>
            </div>
            <div class="metric-card">
                <h3>Avg Precision</h3>
                <div class="value">{avg_precision:.3f}</div>
            </div>
            <div class="metric-card">
                <h3>Avg Recall</h3>
                <div class="value">{avg_recall:.3f}</div>
            </div>
            <div class="metric-card">
                <h3>Net P&L</h3>
                <div class="value">${total_pnl:+,.0f}</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Cumulative P&L Across Folds</div>
            <canvas id="pnlChart"></canvas>
        </div>
        
        <div class="note">
            <h3>⚠️ Note on Trade Details</h3>
            <p>
                This report shows <strong>fold-by-fold summary metrics</strong> parsed from the WF log file.
                Individual trade details are not included because the WF script overwrites <code>walkforward_trades_sim_trades.csv</code> on each run.
                To see individual trades, re-run WF for this config with unique output filenames (~30 minutes).
            </p>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Fold</th>
                    <th>Period</th>
                    <th>AUC</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>Threshold</th>
                    <th>Signals</th>
                    <th>Start Bal</th>
                    <th>End Bal</th>
                    <th>Return %</th>
                    <th>P&L</th>
                    <th>Result</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for i, fold in enumerate(folds):
        pnl = fold['end_balance'] - fold['start_balance']
        pnl_class = 'positive' if pnl > 0 else 'negative' if pnl < 0 else 'neutral'
        result_badge = 'badge-win' if pnl > 0 else 'badge-loss'
        result_text = '✅ Win' if pnl > 0 else '❌ Loss'
        
        html += f"""
                <tr>
                    <td><strong>{fold['fold']}</strong></td>
                    <td class="date">{fold['start_date']} → {fold['end_date']}</td>
                    <td>{fold['auc']:.4f}</td>
                    <td>{fold['precision']:.4f}</td>
                    <td>{fold['recall']:.4f}</td>
                    <td>{fold['f1']:.4f}</td>
                    <td>{fold['threshold']:.2f}</td>
                    <td>{fold['signals']:,}</td>
                    <td>${fold['start_balance']:,}</td>
                    <td>${fold['end_balance']:,}</td>
                    <td class="{pnl_class}">{fold['pct_change']:+.1f}%</td>
                    <td class="{pnl_class}">${pnl:+,}</td>
                    <td><span class="badge {result_badge}">{result_text}</span></td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
        
        <script>
            const ctx = document.getElementById('pnlChart').getContext('2d');
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: """ + str([f"Fold {f['fold']}" for f in folds]) + """,
                    datasets: [{
                        label: 'Cumulative P&L ($)',
                        data: """ + str(cumulative_pnl) + """,
                        borderColor: 'rgb(102, 126, 234)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        pointBackgroundColor: 'rgb(102, 126, 234)',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 3,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top'
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            padding: 12,
                            titleFont: { size: 14, weight: 'bold' },
                            bodyFont: { size: 13 },
                            displayColors: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            },
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toLocaleString();
                                },
                                font: { size: 12 }
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                font: { size: 11 },
                                maxRotation: 45,
                                minRotation: 45
                            }
                        }
                    },
                    interaction: {
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }
                }
            });
        </script>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_wf_log_to_html.py <log_file>")
        return 1
    
    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return 1
    
    config_name = log_path.stem
    
    print(f"📖 Parsing {log_path.name}...")
    print("="*70)
    
    # Parse fold results
    folds = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Fold' in line and 'Test:' in line:
                fold_data = parse_fold_line(line)
                if fold_data:
                    folds.append(fold_data)
    
    if not folds:
        print("❌ No fold results found in log file")
        return 1
    
    print(f"✅ Parsed {len(folds)} folds")
    print()
    
    # Generate HTML
    output_path = Path("outputs") / f"{config_name}_report.html"
    generate_html_report(config_name, folds, output_path)
    
    print(f"✅ HTML report generated: {output_path}")
    print(f"📊 {len(folds)} folds • {sum(f['signals'] for f in folds):,} total signals")
    print(f"💰 Net P&L: ${sum(f['end_balance'] - f['start_balance'] for f in folds):+,.0f}")
    print()
    print("="*70)
    print(f"🎉 Open in browser: {output_path.absolute()}")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
