#!/usr/bin/env python3
"""
Generate comprehensive HTML report from WF results with trade-by-trade details
"""
import sys
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

def parse_fold_line(line):
    """Parse a fold result line from WF log"""
    pattern = r'Fold\s+(\d+)/(\d+)\s+Test:\s+([\d-]+)\s+->\s+([\d-]+)\s+\|\s+AUC=([\d.]+)\s+Prec=([\d.]+)\s+Recall=([\d.]+)\s+F1=([\d.]+)\s+Thr=([\d.]+)\s+Sigs=(\d+)/(\d+)\s+Bal:\s+\$(\d+)→\$(\d+(?:,\d+)?)\s+\(([+-][\d.]+)%\)'
    
    match = re.search(pattern, line)
    if not match:
        return None
    
    return {
        'fold': int(match.group(1)),
        'start_date': match.group(3),
        'end_date': match.group(4),
        'auc': float(match.group(5)),
        'precision': float(match.group(6)),
        'recall': float(match.group(7)),
        'f1': float(match.group(8)),
        'threshold': float(match.group(9)),
        'signals': int(match.group(10)),
        'start_balance': int(match.group(12)),
        'end_balance': int(match.group(13).replace(',', '')),
        'pct_change': float(match.group(14))
    }

def load_trades_for_fold(trades_df, fold_start, fold_end):
    """Filter trades within fold date range"""
    # Convert time column to datetime if needed
    if 'time' not in trades_df.columns:
        return pd.DataFrame()
    
    # Make a copy to avoid modifying original
    df = trades_df.copy()
    
    # Convert time to datetime if string
    if df['time'].dtype == 'object' or df['time'].dtype.name == 'string':
        df['time'] = pd.to_datetime(df['time'])
    
    fold_start_dt = pd.to_datetime(fold_start)
    fold_end_dt = pd.to_datetime(fold_end)
    
    # Remove timezone from time column if present
    if hasattr(df['time'].dtype, 'tz') and df['time'].dtype.tz is not None:
        df['time'] = df['time'].dt.tz_localize(None)
    
    fold_trades = df[
        (df['time'] >= fold_start_dt) & 
        (df['time'] < fold_end_dt)
    ].copy()
    
    return fold_trades

def generate_html(config_name, folds, trades_df, output_path):
    """Generate comprehensive HTML report"""
    
    # Calculate summary stats
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    trade_wr = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    total_pnl = trades_df['pnl'].sum()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if total_trades - winning_trades > 0 else 0
    
    profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) if trades_df[trades_df['pnl'] <= 0]['pnl'].sum() != 0 else 0
    
    # Calculate cumulative P&L
    cumulative_pnl = []
    total = 0
    for fold in folds:
        pnl = fold['end_balance'] - fold['start_balance']
        total += pnl
        cumulative_pnl.append(total)
    
    avg_fold_return = sum(f['pct_change'] for f in folds) / len(folds) if folds else 0
    
    # Parse config name for params
    parts = config_name.replace('grid_', '').replace('_th', ' th=').replace('_r', ' r=').replace('_mp', ' mp=').replace('_notrail', ' NO-TRAIL').replace('_trail', ' TRAIL').replace('_slip', ' slip=')
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config_name} - Comprehensive Report</title>
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
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #667eea;
            font-size: 2.8em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .subtitle {{
            color: #666;
            font-size: 1.1em;
            margin-bottom: 30px;
            line-height: 1.6;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);
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
        .chart-section {{
            margin: 40px 0;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }}
        .chart-title {{
            font-size: 1.6em;
            color: #667eea;
            margin-bottom: 20px;
            font-weight: 600;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 30px 0;
            font-size: 0.9em;
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
            padding: 14px 10px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8em;
            letter-spacing: 0.5px;
        }}
        td {{
            padding: 12px 10px;
            border-bottom: 1px solid #e8e8e8;
        }}
        tbody tr:hover {{
            background: #f0f2f5;
        }}
        tbody tr:nth-child(even) {{
            background: #fafbfc;
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
            font-size: 0.85em;
        }}
        .fold-section {{
            margin: 40px 0;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
        }}
        .fold-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 18px 25px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .fold-header:hover {{
            opacity: 0.95;
        }}
        .fold-header h3 {{
            margin: 0;
            font-size: 1.2em;
        }}
        .fold-stats {{
            display: flex;
            gap: 25px;
            font-size: 0.9em;
        }}
        .fold-stats span {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .fold-content {{
            padding: 25px;
            max-height: 600px;
            overflow-y: auto;
            background: white;
        }}
        .fold-content.collapsed {{
            display: none;
        }}
        .toggle-btn {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            font-size: 1.3em;
            cursor: pointer;
            padding: 5px 12px;
            border-radius: 6px;
            transition: background 0.2s;
        }}
        .toggle-btn:hover {{
            background: rgba(255,255,255,0.3);
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        .badge-buy {{
            background: #dbeafe;
            color: #1e40af;
        }}
        .badge-sell {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .summary-box {{
            background: #f0f9ff;
            border-left: 4px solid #0284c7;
            padding: 20px;
            border-radius: 8px;
            margin: 30px 0;
        }}
        .summary-box h3 {{
            color: #0c4a6e;
            margin-bottom: 15px;
        }}
        .summary-box .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .stat-item {{
            background: white;
            padding: 12px;
            border-radius: 6px;
        }}
        .stat-item strong {{
            color: #0284c7;
            display: block;
            font-size: 0.85em;
            margin-bottom: 5px;
        }}
        .stat-item .value {{
            font-size: 1.3em;
            font-weight: 600;
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
            <strong>Walk-Forward Validation Report</strong> • {parts}<br>
            Date Range: {folds[0]['start_date']} → {folds[-1]['end_date']} • {len(folds)} Folds • {total_trades:,} Executed Trades<br>
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        
        <div class="summary-box">
            <h3>📈 Executive Summary</h3>
            <div class="stats">
                <div class="stat-item">
                    <strong>Total Trades</strong>
                    <div class="value">{total_trades:,}</div>
                </div>
                <div class="stat-item">
                    <strong>Trade Win Rate</strong>
                    <div class="value {'positive' if trade_wr >= 50 else 'negative'}">{trade_wr:.1f}%</div>
                </div>
                <div class="stat-item">
                    <strong>Profit Factor</strong>
                    <div class="value {'positive' if profit_factor >= 1.3 else 'negative'}">{profit_factor:.3f}</div>
                </div>
                <div class="stat-item">
                    <strong>Net P&L</strong>
                    <div class="value {'positive' if total_pnl > 0 else 'negative'}">${total_pnl:+,.2f}</div>
                </div>
                <div class="stat-item">
                    <strong>Avg Win</strong>
                    <div class="value positive">${avg_win:.2f}</div>
                </div>
                <div class="stat-item">
                    <strong>Avg Loss</strong>
                    <div class="value negative">${avg_loss:.2f}</div>
                </div>
                <div class="stat-item">
                    <strong>Avg Fold Return</strong>
                    <div class="value {'positive' if avg_fold_return > 0 else 'negative'}">{avg_fold_return:+.1f}%</div>
                </div>
                <div class="stat-item">
                    <strong>Daily P&L Est.</strong>
                    <div class="value positive">${total_pnl / (len(folds) * 30):.2f}/day</div>
                </div>
            </div>
        </div>
        
        <div class="chart-section">
            <div class="chart-title">Cumulative P&L Across Folds</div>
            <canvas id="pnlChart"></canvas>
        </div>
        
        <div class="chart-section">
            <div class="chart-title">Fold-by-Fold Performance</div>
            <table>
                <thead>
                    <tr>
                        <th>Fold</th>
                        <th>Period</th>
                        <th>AUC</th>
                        <th>Prec</th>
                        <th>Recall</th>
                        <th>F1</th>
                        <th>Signals</th>
                        <th>Start</th>
                        <th>End</th>
                        <th>Return %</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for fold in folds:
        pnl = fold['end_balance'] - fold['start_balance']
        pnl_class = 'positive' if pnl > 0 else 'negative' if pnl < 0 else 'neutral'
        
        html += f"""
                    <tr>
                        <td><strong>{fold['fold']}</strong></td>
                        <td class="date">{fold['start_date']} → {fold['end_date']}</td>
                        <td>{fold['auc']:.4f}</td>
                        <td>{fold['precision']:.4f}</td>
                        <td>{fold['recall']:.4f}</td>
                        <td>{fold['f1']:.4f}</td>
                        <td>{fold['signals']:,}</td>
                        <td>${fold['start_balance']:,}</td>
                        <td>${fold['end_balance']:,}</td>
                        <td class="{pnl_class}">{fold['pct_change']:+.1f}%</td>
                        <td class="{pnl_class}">${pnl:+,}</td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
        </div>
        
        <h2 style="color: #667eea; margin: 40px 0 20px 0; font-size: 2em;">📋 Trade-by-Trade Details</h2>
"""
    
    # Generate fold sections with trades
    for fold in folds:
        fold_trades = load_trades_for_fold(trades_df, fold['start_date'], fold['end_date'])
        
        if len(fold_trades) == 0:
            continue
        
        fold_wins = len(fold_trades[fold_trades['pnl'] > 0])
        fold_losses = len(fold_trades) - fold_wins
        fold_wr = fold_wins / len(fold_trades) * 100 if len(fold_trades) > 0 else 0
        fold_pnl = fold_trades['pnl'].sum()
        
        html += f"""
        <div class="fold-section">
            <div class="fold-header" onclick="toggleFold('fold-{fold['fold']}')">
                <h3>Fold {fold['fold']} • {fold['start_date']} → {fold['end_date']}</h3>
                <div class="fold-stats">
                    <span>📊 {len(fold_trades)} trades</span>
                    <span class="positive">✅ {fold_wins} wins</span>
                    <span class="negative">❌ {fold_losses} losses</span>
                    <span>WR: {fold_wr:.1f}%</span>
                    <span>P&L: ${fold_pnl:+.2f}</span>
                </div>
                <button class="toggle-btn">▼</button>
            </div>
            <div id="fold-{fold['fold']}" class="fold-content {'collapsed' if fold['fold'] > 3 else ''}">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Side</th>
                            <th>Entry</th>
                            <th>Exit</th>
                            <th>SL</th>
                            <th>TP</th>
                            <th>RR Target</th>
                            <th>RR Realized</th>
                            <th>P&L</th>
                            <th>Result</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for _, trade in fold_trades.iterrows():
            pnl = trade.get('pnl', 0)
            pnl_class = 'positive' if pnl > 0 else 'negative'
            result = '✅ Win' if pnl > 0 else '❌ Loss'
            side = trade.get('side', 'N/A')
            side_class = 'badge-buy' if side == 'BUY' else 'badge-sell'
            
            html += f"""
                        <tr>
                            <td class="date">{trade.get('time', 'N/A')}</td>
                            <td><span class="badge {side_class}">{side}</span></td>
                            <td>{trade.get('entry_price', 0):.2f}</td>
                            <td>{trade.get('exit_price', 0):.2f}</td>
                            <td>{trade.get('sl_price', 0):.2f}</td>
                            <td>{trade.get('tp_price', 0):.2f}</td>
                            <td>{trade.get('rr_target', 0):.2f}</td>
                            <td class="{pnl_class}">{trade.get('realized_rr', 0):.2f}</td>
                            <td class="{pnl_class}">${pnl:.2f}</td>
                            <td class="{pnl_class}">{result}</td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
"""
    
    html += f"""
        <script>
            const ctx = document.getElementById('pnlChart').getContext('2d');
            const chart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {[f"Fold {f['fold']}" for f in folds]},
                    datasets: [{{
                        label: 'Cumulative P&L ($)',
                        data: {cumulative_pnl},
                        borderColor: 'rgb(102, 126, 234)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        pointBackgroundColor: 'rgb(102, 126, 234)',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 3,
                    plugins: {{
                        legend: {{
                            display: true,
                            position: 'top',
                            labels: {{
                                font: {{
                                    size: 14,
                                    weight: 'bold'
                                }}
                            }}
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false,
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            padding: 12,
                            titleFont: {{ size: 14, weight: 'bold' }},
                            bodyFont: {{ size: 13 }},
                            displayColors: false,
                            callbacks: {{
                                label: function(context) {{
                                    return 'P&L: $' + context.parsed.y.toLocaleString();
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            grid: {{
                                color: 'rgba(0, 0, 0, 0.05)'
                            }},
                            ticks: {{
                                callback: function(value) {{
                                    return '$' + value.toLocaleString();
                                }},
                                font: {{ size: 12 }}
                            }}
                        }},
                        x: {{
                            grid: {{
                                display: false
                            }},
                            ticks: {{
                                font: {{ size: 11 }},
                                maxRotation: 45,
                                minRotation: 45
                            }}
                        }}
                    }},
                    interaction: {{
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }}
                }}
            }});
            
            function toggleFold(foldId) {{
                const content = document.getElementById(foldId);
                content.classList.toggle('collapsed');
                const btn = content.previousElementSibling.querySelector('.toggle-btn');
                btn.textContent = content.classList.contains('collapsed') ? '▶' : '▼';
            }}
        </script>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path

def main():
    config_name = "grid_0014_th70_r5_mp4_notrail_slip15"
    
    log_path = Path(f"outputs/grid_search/{config_name}.log")
    trades_path = Path(f"outputs/grid_0014_trades_sim.csv")
    
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return 1
    
    if not trades_path.exists():
        print(f"❌ Trades CSV not found: {trades_path}")
        return 1
    
    print(f"📖 Parsing {log_path.name}...")
    print(f"📊 Loading {trades_path.name}...")
    print("="*70)
    
    # Parse fold results from log
    folds = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Fold' in line and 'Test:' in line:
                fold_data = parse_fold_line(line)
                if fold_data:
                    folds.append(fold_data)
    
    print(f"✅ Parsed {len(folds)} folds from log")
    
    # Load trades CSV
    trades_df = pd.read_csv(trades_path)
    # Force convert time column to datetime immediately
    if 'time' in trades_df.columns:
        trades_df['time'] = pd.to_datetime(trades_df['time'], errors='coerce')
    print(f"✅ Loaded {len(trades_df)} trades from CSV")
    print()
    
    # Generate HTML
    output_path = Path(f"outputs/grid_0014_comprehensive_report.html")
    generate_html(config_name, folds, trades_df, output_path)
    
    print(f"✅ HTML report generated: {output_path}")
    print(f"📊 {len(folds)} folds • {len(trades_df):,} executed trades")
    print(f"💰 Net P&L: ${trades_df['pnl'].sum():+,.2f}")
    print()
    print("="*70)
    print(f"🎉 Open in browser: {output_path.absolute()}")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
