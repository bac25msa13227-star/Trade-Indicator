#!/usr/bin/env python3
"""
Generate detailed Markdown trade report for Combo #133
Includes: fold summary, monthly breakdown, all 11,668 trades table
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timezone

CSV_PATH = Path("outputs/combo133_trades.csv")
OUT_PATH = Path("outputs/combo133_trade_report.md")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["time"] = pd.to_datetime(df["time"], utc=True)
df["date"] = df["time"].dt.date
df["month"] = df["time"].dt.strftime("%Y-%m")

# Outcome labels
def outcome(row):
    rr = row["realized_rr"]
    if row["is_win"]:
        if rr >= 2.0:   return "✅ TP Full"
        elif rr >= 1.2: return "✅ TP Partial"
        else:           return "✅ Win"
    elif row["is_loss"]:
        if rr == -1.0:  return "❌ SL Hit"
        else:           return "❌ Loss"
    else:
        return "➖ Hòa"

def sl_moved(row):
    rr = row["realized_rr"]
    # Trail activates at RR=1.0, breakeven at RR=0.5
    if row["is_win"] and rr >= 1.0:
        return "Có (Trail)"
    elif row["is_win"] and rr >= 0.5:
        return "Có (BE)"
    elif row["is_loss"] and rr > -1.0:
        return "Có (Partial)"
    else:
        return "Không"

df["outcome"]   = df.apply(outcome, axis=1)
df["sl_moved"]  = df.apply(sl_moved, axis=1)
df["side_vn"]   = df["side"].map({"buy": "MUA", "sell": "BÁN"})

# ── Fold map (from saved fold column) ────────────────────────────────────────
fold_summary = (
    df.groupby("fold")
    .agg(
        tu_ngay    = ("time", "min"),
        den_ngay   = ("time", "max"),
        so_lenh    = ("pnl",  "count"),
        wr         = ("is_win", "mean"),
        tong_pnl   = ("pnl",  "sum"),
        von_dau    = ("balance_before", "first"),
        du_cuoi    = ("balance_after",  "last"),
    )
    .reset_index()
)
fold_summary["tu_ngay"] = fold_summary["tu_ngay"].dt.strftime("%Y-%m-%d")
fold_summary["den_ngay"] = fold_summary["den_ngay"].dt.strftime("%Y-%m-%d")

# ── Monthly summary ───────────────────────────────────────────────────────────
monthly = (
    df.groupby("month")
    .agg(
        so_lenh  = ("pnl", "count"),
        wr       = ("is_win", "mean"),
        tong_pnl = ("pnl", "sum"),
        du_cuoi  = ("balance_after", "last"),
    )
    .reset_index()
)
monthly["cum_pnl"] = monthly["tong_pnl"].cumsum()

# ── Write MD ──────────────────────────────────────────────────────────────────
lines = []
A = lines.append

A("# Báo Cáo Giao Dịch — Combo #133 (XAUUSD AI)")
A("")
A("> **Cấu hình:** min_confidence=0.70 | require_trend=False | blocked_hours=[3,15,17,22,23]  ")
A("> **Model:** VotingClassifier (HGB×1 + RF×1 + ET×1, weights 3:2:1)  ")
A("> **RR mục tiêu:** 3.5:1 (Volatile: 5.0:1) | **Partial TP:** 1.2R (50%) | **Trail:** kích hoạt tại 1.0R  ")
A("> **Vốn ban đầu mỗi fold:** $200 | **Dữ liệu:** XAUUSD M5 Dukascopy  ")
A(f"> **Tổng lệnh:** {len(df):,} | **Kỳ:** Jan 2024 – Apr 2026  ")
A("")

# ─ Overall stats ─
total_pnl   = df["pnl"].sum()
total_wins  = df["is_win"].sum()
total_loss  = df["is_loss"].sum()
total_draw  = df["is_draw"].sum()
overall_wr  = total_wins / len(df)
avg_win     = df.loc[df["is_win"],  "pnl"].mean()
avg_loss    = df.loc[df["is_loss"], "pnl"].mean()
pf          = abs(df.loc[df["is_win"], "pnl"].sum() / df.loc[df["is_loss"], "pnl"].sum()) if df["is_loss"].sum() > 0 else float("inf")
avg_rr      = df.loc[df["is_win"], "realized_rr"].mean()

A("## 1. Tổng Quan")
A("")
A("| Chỉ số | Giá trị |")
A("|--------|---------|")
A(f"| Tổng số lệnh | {len(df):,} |")
A(f"| Tổng lãi/lỗ | **+${total_pnl:,.2f}** |")
A(f"| Số fold (tháng) | {df['fold'].nunique()} |")
A(f"| Win Rate tổng | **{overall_wr:.1%}** |")
A(f"| Lệnh thắng (Win) | {int(total_wins):,} ({overall_wr:.1%}) |")
A(f"| Lệnh thua (Loss) | {int(total_loss):,} ({total_loss/len(df):.1%}) |")
A(f"| Lệnh hòa (Draw) | {int(total_draw):,} |")
A(f"| Lãi TB / lệnh thắng | +${avg_win:.2f} |")
A(f"| Lỗ TB / lệnh thua | ${avg_loss:.2f} |")
A(f"| Profit Factor | {pf:.2f} |")
A(f"| RR TB lệnh thắng | {avg_rr:.2f}R |")
A(f"| SL bị hit | {(df['realized_rr']==-1.0).sum():,} lệnh |")
A(f"| TP Full (≥2R) | {(df['is_win'] & (df['realized_rr']>=2.0)).sum():,} lệnh |")
A(f"| Thoát trailing | {(df['is_win'] & (df['realized_rr']<2.0)).sum():,} lệnh |")
A("")

# ─ Fold summary ─
A("---")
A("")
A("## 2. Tóm Tắt Theo Fold (~3 tháng/fold)")
A("")
A("| Fold | Từ ngày | Đến ngày | Số lệnh | Win Rate | Tổng P&L | Vốn đầu | Số dư cuối | Kết quả |")
A("|------|---------|----------|---------|----------|----------|---------|------------|---------|")
for _, r in fold_summary.iterrows():
    flag = "✅" if r["tong_pnl"] >= 0 else "❌"
    partial = "\\*" if r["fold"] == fold_summary["fold"].max() else ""
    A(f"| {int(r['fold'])}{partial} | {r['tu_ngay']} | {r['den_ngay']} | {int(r['so_lenh']):,} | {r['wr']:.1%} | "
      f"{'+'if r['tong_pnl']>=0 else ''}${r['tong_pnl']:,.2f} | "
      f"${r['von_dau']:,.2f} | ${r['du_cuoi']:,.2f} | {flag} |")
A("")
A("> \\* = fold partial (dữ liệu chưa đủ 1 fold đầy đủ)")
A("")

# ─ Monthly summary ─
A("---")
A("")
A("## 3. Tóm Tắt Theo Tháng")
A("")
A("| Tháng | Số lệnh | Win Rate | P&L tháng | Cộng dồn | Số dư cuối |")
A("|-------|---------|----------|-----------|----------|------------|")
for _, r in monthly.iterrows():
    flag = "🚀" if r["tong_pnl"] > 5000 else ("✅" if r["tong_pnl"] >= 0 else "❌")
    A(f"| {r['month']} | {int(r['so_lenh']):,} | {r['wr']:.1%} | "
      f"{'+'if r['tong_pnl']>=0 else ''}${r['tong_pnl']:,.2f} | "
      f"${r['cum_pnl']:,.2f} | ${r['du_cuoi']:,.2f} | {flag} |")
A("")

# ─ All trades grouped by day ─
A("---")
A("")
A("## 4. Toàn Bộ Lệnh Giao Dịch (Phân Theo Ngày)")
A("")
A("> **Ghi chú cột:**  ")
A("> - **Giá vào**: Entry price (USD/oz)  ")
A("> - **RR đạt**: Realized RR (−1.0 = hit SL; ≥2.0 = hit TP; 0.5–1.9 = thoát trailing)  ")
A("> - **Dịch SL**: Trail = SL dịch trailing @ 1.0R | BE = breakeven @ 0.5R | Không = SL giữ nguyên  ")
A("> - **Tin cậy**: Xác suất model (0.70–1.00)  ")
A("")

regime_map = {0: "Normal", 1: "Volatile", 2: "Sideway", 3: "Strong Vol"}

# Group by date
global_trade_num = 0
for day, day_df in df.groupby("date"):
    day_wins  = day_df["is_win"].sum()
    day_loss  = day_df["is_loss"].sum()
    day_pnl   = day_df["pnl"].sum()
    day_wr    = day_wins / len(day_df)
    bal_end   = day_df["balance_after"].iloc[-1]
    pnl_str   = f"+${day_pnl:.2f}" if day_pnl >= 0 else f"-${abs(day_pnl):.2f}"
    day_flag  = "🚀" if day_pnl > 500 else ("✅" if day_pnl >= 0 else "❌")
    fold_no   = int(day_df["fold"].iloc[0])

    A(f"### {day} — {len(day_df)} lệnh | WR {day_wr:.0%} | P&L {pnl_str} | Số dư: ${bal_end:.2f} {day_flag} | Fold {fold_no}")
    A("")
    A("| # | Giờ (UTC) | Chiều | Giá vào | RR đạt | P&L ($) | Số dư sau | Kết quả | Dịch SL | Tin cậy | Regime |")
    A("|---|-----------|-------|---------|--------|---------|-----------|---------|---------|---------|--------|")

    for _, row in day_df.iterrows():
        global_trade_num += 1
        time_str = row["time"].strftime("%H:%M")
        regime   = regime_map.get(int(row.get("volatility_regime", 0)), "?")
        p        = f"+${row['pnl']:.2f}" if row["pnl"] >= 0 else f"-${abs(row['pnl']):.2f}"
        A(f"| {global_trade_num} | {time_str} | {row['side_vn']} | {row['entry_price']:.2f} | "
          f"{row['realized_rr']:.3f} | {p} | ${row['balance_after']:.2f} | "
          f"{row['outcome']} | {row['sl_moved']} | {row['probability']:.3f} | {regime} |")
    A("")
A("---")
A("")
A("## 5. Ghi Chú Kỹ Thuật")
A("")
A("| Tham số | Giá trị |")
A("|---------|---------|")
A("| Config file | `configs/acc1_v14pp_profit.yaml` |")
A("| Instrument | XAUUSD (Gold Spot) |")
A("| Timeframe | M5 (5 phút) |")
A("| Train bars / fold | 30,000 bars (~3 tháng) |")
A("| Test bars / fold | 6,000 bars (~3 tuần) |")
A("| SL | ATR × 1.5 từ entry |")
A("| TP normal | RR 3.5:1 |")
A("| TP volatile | RR 5.0:1 |")
A("| Partial TP | 50% vị thế tại RR 1.2 |")
A("| Breakeven SL | Kích hoạt tại RR 0.5 |")
A("| Trailing SL | Kích hoạt tại RR 1.0, trail ATR×1.0 |")
A("| Min confidence | 0.70 |")
A("| Blocked hours UTC | 3h, 15h, 17h, 22h, 23h |")
A("| Risk/trade | Biến động theo anti-martingale |")
A("")

# ── Write file ────────────────────────────────────────────────────────────────
text = "\n".join(lines)
OUT_PATH.write_text(text, encoding="utf-8")
print(f"✅ Đã tạo: {OUT_PATH}")
print(f"   Kích thước: {OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")
print(f"   Số dòng: {len(lines):,}")
print(f"   Tổng lệnh trong báo cáo: {len(df):,}")
