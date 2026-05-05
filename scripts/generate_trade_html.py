#!/usr/bin/env python3
"""
Generate interactive HTML report for walk-forward trades.
Shows all executed trades with color coding, filters, and statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_trade_html(csv_path: str, output_path: str = None) -> str:
    """Generate HTML report from trade CSV."""
    
    # Load data
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # Calculate statistics
    total_trades = len(df)
    wins = len(df[df['pnl'] > 0])
    losses = len(df[df['pnl'] < 0])
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    
    total_pnl = df['pnl'].sum()
    avg_win = df[df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losses > 0 else 0
    profit_factor = abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()) if losses > 0 else 0
    
    # Format display columns
    df_display = df.copy()
    df_display['time'] = df_display['time'].dt.strftime('%Y-%m-%d %H:%M')
    df_display['side'] = df_display['side'].map({1: 'BUY', -1: 'SELL', 'buy': 'BUY', 'sell': 'SELL'})
    df_display['entry_price'] = df_display['entry_price'].round(2)
    df_display['realized_rr'] = df_display['realized_rr'].round(2)
    df_display['net_rr'] = df_display['net_rr'].round(2)
    df_display['pnl'] = df_display['pnl'].round(2)
    df_display['balance_after'] = df_display['balance_after'].round(2)
    df_display['probability'] = (df_display['probability'] * 100).round(1)
    df_display['drawdown'] = (df_display['drawdown'] * 100).round(2)
    
    # Select key columns for display
    display_cols = [
        'time', 'side', 'entry_price', 'realized_rr', 'net_rr', 
        'friction_rr', 'bars_held', 'probability', 'pnl', 
        'balance_after', 'drawdown', 'fold', 'test_start', 'test_end'
    ]
    df_display = df_display[display_cols]
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Walk-Forward Trade Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        h1 {{
            color: #58a6ff;
            margin-bottom: 10px;
            font-size: 2em;
        }}
        
        .subtitle {{
            color: #8b949e;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px;
        }}
        
        .stat-label {{
            color: #8b949e;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        
        .stat-value.positive {{
            color: #3fb950;
        }}
        
        .stat-value.negative {{
            color: #f85149;
        }}
        
        .filters {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .filter-group {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .filter-group label {{
            color: #8b949e;
            margin-right: 5px;
        }}
        
        .filter-group select, .filter-group input {{
            background: #0d1117;
            border: 1px solid #30363d;
            color: #c9d1d9;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.95em;
        }}
        
        .filter-group button {{
            background: #238636;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.95em;
            font-weight: 500;
        }}
        
        .filter-group button:hover {{
            background: #2ea043;
        }}
        
        .table-container {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        
        thead {{
            background: #0d1117;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        
        th {{
            padding: 12px 10px;
            text-align: left;
            font-weight: 600;
            color: #8b949e;
            border-bottom: 2px solid #30363d;
            cursor: pointer;
            user-select: none;
        }}
        
        th:hover {{
            background: #21262d;
        }}
        
        th.sortable::after {{
            content: ' ↕';
            color: #30363d;
        }}
        
        th.sort-asc::after {{
            content: ' ↑';
            color: #58a6ff;
        }}
        
        th.sort-desc::after {{
            content: ' ↓';
            color: #58a6ff;
        }}
        
        tbody tr {{
            border-bottom: 1px solid #21262d;
            transition: background 0.2s;
        }}
        
        tbody tr:hover {{
            background: #0d1117;
        }}
        
        td {{
            padding: 10px;
        }}
        
        .side-buy {{
            color: #3fb950;
            font-weight: 600;
        }}
        
        .side-sell {{
            color: #f85149;
            font-weight: 600;
        }}
        
        .pnl-positive {{
            color: #3fb950;
            font-weight: 600;
        }}
        
        .pnl-negative {{
            color: #f85149;
            font-weight: 600;
        }}
        
        .rr-positive {{
            color: #3fb950;
        }}
        
        .rr-negative {{
            color: #f85149;
        }}
        
        .fold-badge {{
            background: #1f6feb;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        
        .hidden {{
            display: none !important;
        }}
        
        .summary {{
            margin-top: 20px;
            padding: 15px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            text-align: center;
            color: #8b949e;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Walk-Forward Trade Report</h1>
        <div class="subtitle">
            {total_trades:,} trades executed | {wins:,} wins ({win_rate:.1f}%) | {losses:,} losses
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total P&L</div>
                <div class="stat-value {'positive' if total_pnl > 0 else 'negative'}">${total_pnl:,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value {'positive' if win_rate >= 50 else 'negative'}">{win_rate:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Profit Factor</div>
                <div class="stat-value {'positive' if profit_factor > 1 else 'negative'}">{profit_factor:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Win</div>
                <div class="stat-value positive">${avg_win:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Loss</div>
                <div class="stat-value negative">${avg_loss:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Final Balance</div>
                <div class="stat-value">${df['balance_after'].iloc[-1]:,.2f}</div>
            </div>
        </div>
        
        <div class="filters">
            <div class="filter-group">
                <label>Filter:</label>
                <select id="sideFilter">
                    <option value="all">All Sides</option>
                    <option value="BUY">BUY only</option>
                    <option value="SELL">SELL only</option>
                </select>
                
                <select id="outcomeFilter">
                    <option value="all">All Outcomes</option>
                    <option value="win">Wins only</option>
                    <option value="loss">Losses only</option>
                </select>
                
                <select id="foldFilter">
                    <option value="all">All Folds</option>
                    {generate_fold_options(df)}
                </select>
                
                <input type="text" id="searchInput" placeholder="Search time, price...">
                
                <button onclick="resetFilters()">Reset Filters</button>
            </div>
        </div>
        
        <div class="table-container">
            <table id="tradeTable">
                <thead>
                    <tr>
                        <th class="sortable" onclick="sortTable(0)">Time</th>
                        <th class="sortable" onclick="sortTable(1)">Side</th>
                        <th class="sortable" onclick="sortTable(2)">Entry</th>
                        <th class="sortable" onclick="sortTable(3)">Realized RR</th>
                        <th class="sortable" onclick="sortTable(4)">Net RR</th>
                        <th class="sortable" onclick="sortTable(5)">Friction</th>
                        <th class="sortable" onclick="sortTable(6)">Bars</th>
                        <th class="sortable" onclick="sortTable(7)">Prob %</th>
                        <th class="sortable" onclick="sortTable(8)">P&L</th>
                        <th class="sortable" onclick="sortTable(9)">Balance</th>
                        <th class="sortable" onclick="sortTable(10)">DD %</th>
                        <th class="sortable" onclick="sortTable(11)">Fold</th>
                        <th>Period</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Generate table rows
    for idx, row in df_display.iterrows():
        side_class = 'side-buy' if row['side'] == 'BUY' else 'side-sell'
        pnl_class = 'pnl-positive' if row['pnl'] > 0 else 'pnl-negative'
        rr_class = 'rr-positive' if row['net_rr'] > 0 else 'rr-negative'
        
        html += f"""
                    <tr data-side="{row['side']}" data-pnl="{'win' if row['pnl'] > 0 else 'loss'}" data-fold="{row['fold']}">
                        <td>{row['time']}</td>
                        <td class="{side_class}">{row['side']}</td>
                        <td>${row['entry_price']}</td>
                        <td class="{rr_class}">{row['realized_rr']:+.2f}R</td>
                        <td class="{rr_class}">{row['net_rr']:+.2f}R</td>
                        <td>{row['friction_rr']:.3f}R</td>
                        <td>{row['bars_held']}</td>
                        <td>{row['probability']:.1f}%</td>
                        <td class="{pnl_class}">${row['pnl']:+.2f}</td>
                        <td>${row['balance_after']:,.2f}</td>
                        <td>{row['drawdown']:+.2f}%</td>
                        <td><span class="fold-badge">{row['fold']}</span></td>
                        <td>{row['test_start']}</td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
        </div>
        
        <div class="summary" id="summary">
            Showing <span id="visibleCount">0</span> of {total_trades:,} trades
        </div>
    </div>
    
    <script>
        // Filter functionality
        function applyFilters() {{
            const sideFilter = document.getElementById('sideFilter').value;
            const outcomeFilter = document.getElementById('outcomeFilter').value;
            const foldFilter = document.getElementById('foldFilter').value;
            const searchText = document.getElementById('searchInput').value.toLowerCase();
            
            const rows = document.querySelectorAll('#tradeTable tbody tr');
            let visibleCount = 0;
            
            rows.forEach(row => {{
                const side = row.dataset.side;
                const pnl = row.dataset.pnl;
                const fold = row.dataset.fold;
                const text = row.textContent.toLowerCase();
                
                const sideMatch = sideFilter === 'all' || side === sideFilter;
                const outcomeMatch = outcomeFilter === 'all' || pnl === outcomeFilter;
                const foldMatch = foldFilter === 'all' || fold === foldFilter;
                const searchMatch = searchText === '' || text.includes(searchText);
                
                if (sideMatch && outcomeMatch && foldMatch && searchMatch) {{
                    row.classList.remove('hidden');
                    visibleCount++;
                }} else {{
                    row.classList.add('hidden');
                }}
            }});
            
            document.getElementById('visibleCount').textContent = visibleCount.toLocaleString();
        }}
        
        function resetFilters() {{
            document.getElementById('sideFilter').value = 'all';
            document.getElementById('outcomeFilter').value = 'all';
            document.getElementById('foldFilter').value = 'all';
            document.getElementById('searchInput').value = '';
            applyFilters();
        }}
        
        // Attach filter listeners
        document.getElementById('sideFilter').addEventListener('change', applyFilters);
        document.getElementById('outcomeFilter').addEventListener('change', applyFilters);
        document.getElementById('foldFilter').addEventListener('change', applyFilters);
        document.getElementById('searchInput').addEventListener('input', applyFilters);
        
        // Sorting functionality
        let sortColumn = -1;
        let sortAscending = true;
        
        function sortTable(columnIndex) {{
            const table = document.getElementById('tradeTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr:not(.hidden)'));
            
            // Update sort direction
            if (sortColumn === columnIndex) {{
                sortAscending = !sortAscending;
            }} else {{
                sortAscending = true;
                sortColumn = columnIndex;
            }}
            
            // Update header classes
            document.querySelectorAll('th').forEach((th, idx) => {{
                th.classList.remove('sort-asc', 'sort-desc');
                if (idx === columnIndex) {{
                    th.classList.add(sortAscending ? 'sort-asc' : 'sort-desc');
                }}
            }});
            
            // Sort rows
            rows.sort((a, b) => {{
                const aValue = a.children[columnIndex].textContent.replace(/[$,%R]/g, '').trim();
                const bValue = b.children[columnIndex].textContent.replace(/[$,%R]/g, '').trim();
                
                const aNum = parseFloat(aValue);
                const bNum = parseFloat(bValue);
                
                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return sortAscending ? aNum - bNum : bNum - aNum;
                }}
                
                return sortAscending ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
            }});
            
            // Reorder rows
            rows.forEach(row => tbody.appendChild(row));
        }}
        
        // Initialize
        applyFilters();
    </script>
</body>
</html>
""".format(total_trades=total_trades)
    
    # Write to file
    if output_path is None:
        output_path = csv_path.replace('.csv', '.html')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path


def generate_fold_options(df):
    """Generate fold filter options."""
    folds = sorted(df['fold'].unique())
    options = ''.join([f'<option value="{fold}">Fold {fold}</option>' for fold in folds])
    return options


if __name__ == '__main__':
    import sys
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/walkforward_trades_acc1_v14pp_profit_sim_trades.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = generate_trade_html(csv_path, output_path)
    print(f"✅ Generated HTML report: {result}")
    print(f"📊 Open in browser: file://{Path(result).absolute()}")
