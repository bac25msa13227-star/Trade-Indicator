from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "outputs"

# ── Colour palette ──────────────────────────────────────────────────────────
GREEN = "#26a69a"
RED = "#ef5350"
AMBER = "#ffa726"
BLUE = "#42a5f5"
GREY = "#90a4ae"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _colour(val: float, good_positive: bool = True) -> str:
    if good_positive:
        return GREEN if val > 0 else (RED if val < 0 else GREY)
    return RED if val > 0 else (GREEN if val < 0 else GREY)


def _pct(val: float | None, decimals: int = 1) -> str:
    if val is None:
        return "n/a"
    return f"{float(val):.{decimals}%}"


def _round(val, decimals: int = 4):
    try:
        return round(float(val), decimals)
    except Exception:
        return val


def _threshold_bar(label: str, value: float, threshold: float, reverse: bool = False) -> None:
    """Renders a labelled progress bar showing value vs threshold."""
    pct = min(abs(value) / max(abs(threshold) * 2, 1e-9), 1.0)
    passed = (value >= threshold) if not reverse else (value <= threshold)
    colour = GREEN if passed else RED
    icon = "✅" if passed else "❌"
    st.markdown(
        f"""
        <div style="margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;font-size:0.85rem">
            <span>{icon} <b>{label}</b></span>
            <span style="color:{colour}"><b>{value:.4f}</b> / threshold {threshold:.4f}</span>
          </div>
          <div style="background:#1e1e2e;border-radius:4px;height:8px">
            <div style="width:{pct*100:.1f}%;background:{colour};height:8px;border-radius:4px"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _signal_card(title: str, value: str, sub: str, colour: str) -> None:
    st.markdown(
        f"""
        <div style="background:{colour}22;border-left:4px solid {colour};
                    padding:12px 16px;border-radius:6px;margin-bottom:8px">
          <div style="font-size:0.75rem;color:{colour};text-transform:uppercase;letter-spacing:.05em">{title}</div>
          <div style="font-size:1.6rem;font-weight:700;color:#fff">{value}</div>
          <div style="font-size:0.8rem;color:#aaa">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summarize_trades(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty or "pnl" not in trades.columns:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_profit": 0.0,
        }

    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
    return {
        "trades": int(len(trades)),
        "wins": int((trades["pnl"] > 0).sum()),
        "losses": int((trades["pnl"] < 0).sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": float(trades["pnl"].sum()),
    }


def summarize_daily_performance(trades: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    if trades.empty:
        return {
            "mean_daily_return": 0.0,
            "median_daily_return": 0.0,
            "best_day_return": 0.0,
            "worst_day_return": 0.0,
            "share_ge_2": 0.0,
            "share_ge_5": 0.0,
            "share_ge_10": 0.0,
        }, pd.DataFrame()

    daily = trades.copy()
    daily["time"] = pd.to_datetime(daily["time"], utc=True, errors="coerce")
    daily = daily.dropna(subset=["time"]).sort_values("time")
    daily["date"] = daily["time"].dt.date
    daily_summary = daily.groupby("date").agg(
        trades=("pnl", "size"),
        net_pnl=("pnl", "sum"),
        gross_profit=("pnl", lambda series: series[series > 0].sum()),
        gross_loss=("pnl", lambda series: -series[series < 0].sum()),
        start_balance=("balance_before", "first"),
        end_balance=("balance_after", "last"),
    )
    daily_summary["return_pct"] = (daily_summary["end_balance"] / daily_summary["start_balance"] - 1.0) * 100.0

    summary = {
        "mean_daily_return": float(daily_summary["return_pct"].mean()),
        "median_daily_return": float(daily_summary["return_pct"].median()),
        "best_day_return": float(daily_summary["return_pct"].max()),
        "worst_day_return": float(daily_summary["return_pct"].min()),
        "share_ge_2": float((daily_summary["return_pct"] >= 2.0).mean()),
        "share_ge_5": float((daily_summary["return_pct"] >= 5.0).mean()),
        "share_ge_10": float((daily_summary["return_pct"] >= 10.0).mean()),
    }
    return summary, daily_summary.reset_index()


def enrich_trades_with_dataset_context(trades: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or dataset.empty or "time" not in trades.columns or "time" not in dataset.columns:
        return trades

    enriched_trades = trades.copy()
    enriched_trades["time"] = pd.to_datetime(enriched_trades["time"], utc=True, errors="coerce")

    context_columns = [column for column in ["time", "volatility_regime", "strategy_score", "trade_side"] if column in dataset.columns]
    if len(context_columns) <= 1:
        return enriched_trades

    context = dataset[context_columns].copy()
    context["time"] = pd.to_datetime(context["time"], utc=True, errors="coerce")
    context = context.dropna(subset=["time"]).sort_values("time")
    enriched_trades = pd.merge_asof(enriched_trades.sort_values("time"), context, on="time", direction="backward")

    if "volatility_regime" in enriched_trades.columns:
        regime_map = {0: "sideway", 1: "normal", 2: "strong_volatility"}
        enriched_trades["regime_label"] = enriched_trades["volatility_regime"].map(regime_map).fillna("unknown")

    return enriched_trades


def summarize_trades(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty or "pnl" not in trades.columns:
        return {"trades": 0, "wins": 0, "losses": 0, "gross_profit": 0.0, "gross_loss": 0.0, "net_profit": 0.0}
    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
    return {
        "trades": int(len(trades)),
        "wins": int((trades["pnl"] > 0).sum()),
        "losses": int((trades["pnl"] < 0).sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": float(trades["pnl"].sum()),
    }


def summarize_daily_performance(trades: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    if trades.empty:
        return {
            "mean_daily_return": 0.0, "median_daily_return": 0.0,
            "best_day_return": 0.0, "worst_day_return": 0.0,
            "share_ge_2": 0.0, "share_ge_5": 0.0, "share_ge_10": 0.0,
        }, pd.DataFrame()
    daily = trades.copy()
    daily["time"] = pd.to_datetime(daily["time"], utc=True, errors="coerce")
    daily = daily.dropna(subset=["time"]).sort_values("time")
    daily["date"] = daily["time"].dt.date
    daily_summary = daily.groupby("date").agg(
        trades=("pnl", "size"),
        net_pnl=("pnl", "sum"),
        gross_profit=("pnl", lambda s: s[s > 0].sum()),
        gross_loss=("pnl", lambda s: -s[s < 0].sum()),
        start_balance=("balance_before", "first"),
        end_balance=("balance_after", "last"),
    )
    daily_summary["return_pct"] = (daily_summary["end_balance"] / daily_summary["start_balance"] - 1.0) * 100.0
    summary = {
        "mean_daily_return": float(daily_summary["return_pct"].mean()),
        "median_daily_return": float(daily_summary["return_pct"].median()),
        "best_day_return": float(daily_summary["return_pct"].max()),
        "worst_day_return": float(daily_summary["return_pct"].min()),
        "share_ge_2": float((daily_summary["return_pct"] >= 2.0).mean()),
        "share_ge_5": float((daily_summary["return_pct"] >= 5.0).mean()),
        "share_ge_10": float((daily_summary["return_pct"] >= 10.0).mean()),
    }
    return summary, daily_summary.reset_index()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="XAUUSD AI Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for dark-theme cards
st.markdown(
    """
    <style>
    .block-container{padding-top:1rem}
    div[data-testid="metric-container"]{
        background:#1e1e2e;border-radius:8px;padding:10px 14px;
        border-left:3px solid #42a5f5
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 XAUUSD AI Bot — Live Dashboard")

# ── Public URL banner ──────────────────────────────────────────────────────
_tunnel_url_path = OUTPUTS / "tunnel_url.txt"
_pub_url = _tunnel_url_path.read_text(encoding="utf-8").strip() if _tunnel_url_path.exists() else None
if _pub_url:
    st.markdown(
        f"""
        <div style="background:#0d2137;border:1px solid #42a5f5;border-radius:8px;
                    padding:10px 18px;margin-bottom:8px;display:flex;align-items:center;gap:16px">
          <span style="font-size:1.3rem">🌐</span>
          <div>
            <span style="color:#90caf9;font-size:0.8rem;text-transform:uppercase;letter-spacing:.08em">Public URL — truy cập mọi nơi</span><br>
            <a href="{_pub_url}" target="_blank"
               style="color:#42a5f5;font-size:1.05rem;font-weight:700;text-decoration:none">{_pub_url}</a>
          </div>
          <span style="margin-left:auto;background:#1565c0;color:#fff;padding:4px 10px;
                       border-radius:4px;font-size:0.78rem">🟢 LIVE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_live, tab_analysis, tab_backtest, tab_walkforward, tab_paper = st.tabs([
    "🟢 Live Monitor", "🔍 Signal Analysis", "📊 Backtest", "🔄 Walk-Forward", "📝 Paper Trade"
])

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED DATA (loaded once, used in multiple tabs)
# ═══════════════════════════════════════════════════════════════════════════════
paper_log_path  = OUTPUTS / "paper_trade_signals.csv"
live_log_path   = OUTPUTS / "live_learning_log.jsonl"
model_meta_path = OUTPUTS / "model_meta.json"
model_path      = OUTPUTS / "model.pkl"

live_signals = load_csv(paper_log_path)
if not live_signals.empty:
    live_signals["time"] = pd.to_datetime(live_signals["time"], utc=True, errors="coerce")
    live_signals = live_signals.sort_values("time", ascending=False).reset_index(drop=True)

model_meta = load_json(model_meta_path)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.header("🟢 Live Bot Monitor")

    # ── Top bar: model status + auto-refresh ──────────────────────────────────
    hdr_left, hdr_right = st.columns([3, 1])
    with hdr_left:
        if model_path.exists():
            mtime = dt.datetime.fromtimestamp(model_path.stat().st_mtime)
            st.success(f"✅ Model aktif — dilatih terakhir: **{mtime.strftime('%Y-%m-%d %H:%M:%S')}**")
        else:
            st.error("❌ Chưa có model — chạy `train` trước")
    with hdr_right:
        auto_refresh = st.toggle("🔄 Auto-refresh 30s", value=True)

    # ── Model performance metrics ─────────────────────────────────────────────
    if model_meta:
        st.subheader("🧠 Model Performance")
        mm1, mm2, mm3, mm4 = st.columns(4)
        threshold_val = model_meta.get("decision_threshold") or model_meta.get("selected_threshold") or 0.5
        mm1.metric("🎯 Decision Threshold", f"{float(threshold_val):.2f}")
        mm2.metric("📐 Precision", _pct(model_meta.get("precision")))
        mm3.metric("🔁 Recall",    _pct(model_meta.get("recall")))
        mm4.metric("⚖️ F1 Score",  _pct(model_meta.get("f1")))

    st.divider()

    # ── Latest signal big cards ───────────────────────────────────────────────
    if not live_signals.empty:
        latest = live_signals.iloc[0]
        conf   = float(latest.get("confidence", 0))
        side   = str(latest.get("side", "flat"))
        score  = float(latest.get("strategy_score", 0))
        regime = int(latest.get("volatility_regime", 1))
        traded = bool(latest.get("should_trade", False))
        reason = str(latest.get("reason", ""))

        st.subheader("📡 Tín hiệu mới nhất")

        # Hero signal cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            side_label = {"buy": "🟢 BUY", "sell": "🔴 SELL"}.get(side, "⚪ FLAT")
            side_colour = {"buy": GREEN, "sell": RED}.get(side, GREY)
            _signal_card("Tín hiệu", side_label, f"lúc {str(latest.get('time',''))[:16]}", side_colour)
        with c2:
            conf_colour = GREEN if conf >= 0.55 else (AMBER if conf >= 0.45 else RED)
            _signal_card("Confidence (ML)", f"{conf:.1%}", f"ngưỡng: {float(threshold_val):.0%}", conf_colour)
        with c3:
            score_colour = GREEN if abs(score) >= 0.3 else (AMBER if abs(score) >= 0.1 else RED)
            _signal_card("Strategy Score", f"{score:+.3f}", "ICT + Wyckoff + RSI/MACD", score_colour)
        with c4:
            regime_map = {0: ("😴 Sideway", AMBER), 1: ("📊 Normal", BLUE), 2: ("⚡ Strong Vol", GREEN)}
            r_label, r_colour = regime_map.get(regime, ("❓ Unknown", GREY))
            _signal_card("Volatility Regime", r_label, "0=Sideway  1=Normal  2=Strong", r_colour)
        with c5:
            trade_colour = GREEN if traded else RED
            trade_label  = "✅ VÀO LỆNH" if traded else "🚫 KHÔNG VÀO"
            _signal_card("Quyết định", trade_label, reason[:40], trade_colour)

        # Entry / SL / TP
        if traded or float(latest.get("entry_price", 0)) > 0:
            st.divider()
            ep_c, sl_c, tp_c, rr_c = st.columns(4)
            entry = float(latest.get("entry_price", 0))
            sl    = float(latest.get("stop_loss", 0))
            tp    = float(latest.get("take_profit", 0))
            rr    = abs((tp - entry) / (entry - sl)) if abs(entry - sl) > 0 else 0
            ep_c.metric("💰 Entry Price", f"{entry:,.3f}")
            sl_c.metric("🛑 Stop Loss",   f"{sl:,.3f}", delta=f"{sl-entry:+.3f}")
            tp_c.metric("🎯 Take Profit", f"{tp:,.3f}", delta=f"{tp-entry:+.3f}")
            rr_c.metric("⚖️ Risk/Reward", f"1 : {rr:.2f}")

        # ── Why no trade? — Lý do KHÔNG vào lệnh ────────────────────────────
        if not traded:
            st.divider()
            st.subheader("❓ Tại sao Bot KHÔNG vào lệnh?")
            st.markdown(
                f"""
                <div style="background:#2d1b1b;border:1px solid {RED};border-radius:8px;padding:16px;margin-bottom:12px">
                  <span style="font-size:1.1rem;font-weight:700;color:{RED}">🚫 Lý do chính: </span>
                  <span style="font-size:1.1rem;color:#fff">{reason}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            reason_explanations = {
                "Model confidence below threshold": (
                    "ML model dự đoán xác suất tăng/giảm chưa đủ mạnh. "
                    f"Confidence hiện tại **{conf:.1%}** < ngưỡng tối thiểu **{float(threshold_val):.0%}**. "
                    "Chưa đủ pattern rõ ràng từ dữ liệu lịch sử."
                ),
                "Strategy consensus is weak": (
                    f"Strategy Score = **{score:+.4f}** quá thấp. "
                    "ICT, Wyckoff và Momentum chưa đồng thuận đủ mạnh. "
                    "Bot cần ít nhất 1-2 trong 3 hệ thống kỹ thuật xác nhận tín hiệu."
                ),
                "Higher timeframe trend is misaligned": (
                    "Trend D1 và H1 đang đi ngược chiều nhau. "
                    "Bot yêu cầu bias ngày và bias giờ phải cùng hướng trước khi vào lệnh."
                ),
                "strategy_score_too_weak": (
                    f"Strategy Score = **{score:+.4f}**. Với volatility regime hiện tại ({r_label}), "
                    "ngưỡng tối thiểu chưa được đáp ứng."
                ),
                "blocked_hour": (
                    "Giờ này bị chặn theo lịch. Bot tránh các khung giờ biến động do tin tức "
                    "(thường là 7h, 10h, 11h, 22h UTC)."
                ),
                "trend_misaligned": (
                    "Trend D1 (daily_bias) và H1 (hourly_bias) đang ngược chiều. "
                    "Không an toàn để vào lệnh khi trend đa khung thời gian mâu thuẫn."
                ),
                "confidence_below_floor": (
                    f"Confidence {conf:.1%} dưới ngưỡng tối thiểu tuyệt đối (min_confidence). "
                    "Ngay cả khi strategy score tốt, ML model phải xác nhận trước."
                ),
            }
            expl = reason_explanations.get(reason, f"Điều kiện chưa đáp ứng: `{reason}`")
            st.info(f"💡 **Giải thích chi tiết:** {expl}")

            # Checklist 5 điều kiện
            st.markdown("#### ✅ Checklist 5 điều kiện để Bot vào lệnh")
            chk1 = conf >= float(threshold_val)
            chk2 = abs(score) >= 0.3
            chk3 = regime in (1, 2)
            chk4 = "blocked" not in reason.lower() and "misaligned" not in reason.lower()
            chk5 = traded  # nếu vào được thì 5/5

            rows = [
                (chk1, f"ML Confidence ≥ ngưỡng",     f"{conf:.1%} / {float(threshold_val):.0%}"),
                (chk2, "Strategy Score đủ mạnh",        f"{score:+.4f}"),
                (chk3, "Volatility regime không sideway",f"{r_label}"),
                (chk4, "Không bị chặn giờ / ngày",      reason if not chk4 else "✓ Giờ được phép"),
                (chk5, "Tất cả điều kiện đều pass",      "🎯 Vào lệnh!" if chk5 else "Chờ cơ hội..."),
            ]
            for ok, lbl, val in rows:
                icon  = "✅" if ok else "❌"
                color = GREEN if ok else RED
                st.markdown(
                    f'<div style="padding:6px 12px;margin:4px 0;border-radius:5px;'
                    f'background:{"#1a2e1a" if ok else "#2e1a1a"};border-left:3px solid {color}">'
                    f'{icon} <b>{lbl}</b> <span style="float:right;color:{color}">{val}</span></div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Confidence chart ──────────────────────────────────────────────────
        chart_df = live_signals.set_index("time")[["confidence"]].dropna().sort_index()
        st.subheader(f"📈 Confidence theo thời gian (last {len(chart_df)} ticks)")
        if not chart_df.empty:
            # Add threshold line
            chart_df["threshold"] = float(threshold_val)
            st.line_chart(chart_df, color=[BLUE, RED], height=200)

        # ── Strategy score history ────────────────────────────────────────────
        if "strategy_score" in live_signals.columns:
            st.subheader("⚡ Strategy Score theo thời gian")
            score_df = live_signals.set_index("time")["strategy_score"].dropna().sort_index()
            st.bar_chart(score_df, height=180)

        # ── Signal history table ──────────────────────────────────────────────
        st.subheader("📋 Lịch sử tín hiệu (50 gần nhất)")
        display_cols = [c for c in ["time", "side", "confidence", "entry_price", "stop_loss",
                                    "take_profit", "strategy_score", "volatility_regime",
                                    "should_trade", "reason"] if c in live_signals.columns]
        st.dataframe(
            live_signals[display_cols].head(50).style.map(
                lambda v: f"color: {GREEN}" if v is True else (f"color: {RED}" if v is False else ""),
                subset=["should_trade"] if "should_trade" in display_cols else [],
            ),
            use_container_width=True,
        )

    else:
        st.info("⏳ Chưa có tín hiệu nào. Bot đang chạy và giám sát thị trường...")

    # ── Live learning events ──────────────────────────────────────────────────
    if live_log_path.exists():
        st.subheader("🧠 Live Learning Events (retrain tự động)")
        events = []
        with live_log_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
        if events:
            ev_df = pd.DataFrame(events)
            st.dataframe(ev_df.tail(15), use_container_width=True)
        else:
            st.caption("Chưa có sự kiện retrain nào.")

    if auto_refresh:
        time.sleep(30)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIGNAL ANALYSIS (Phân tích chi tiết từng yếu tố)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.header("🔍 Phân tích Chi tiết Tín hiệu")
    st.caption("Hiểu rõ mỗi yếu tố mà bot dùng để ra quyết định vào/không vào lệnh")

    if live_signals.empty:
        st.warning("Chưa có tín hiệu. Đợi bot chạy ít nhất 1 chu kỳ.")
    else:
        latest = live_signals.iloc[0]
        conf    = float(latest.get("confidence", 0))
        score   = float(latest.get("strategy_score", 0))
        regime  = int(latest.get("volatility_regime", 1))
        side    = str(latest.get("side", "flat"))
        traded  = bool(latest.get("should_trade", False))
        reason  = str(latest.get("reason", ""))
        threshold_val = float(
            model_meta.get("decision_threshold") or model_meta.get("selected_threshold") or 0.5
        )

        # ── Section 1: ML Model Analysis ─────────────────────────────────────
        st.subheader("🤖 1. Phân tích ML Model")
        ml_left, ml_right = st.columns([1, 1])

        with ml_left:
            st.markdown("**Confidence Gauge**")
            gauge_pct = int(conf * 100)
            gauge_color = GREEN if conf >= threshold_val else (AMBER if conf >= threshold_val * 0.85 else RED)
            st.markdown(
                f"""
                <div style="text-align:center;padding:20px">
                  <div style="font-size:3rem;font-weight:900;color:{gauge_color}">{conf:.1%}</div>
                  <div style="font-size:0.9rem;color:#888">ML Confidence</div>
                  <div style="background:#1e1e2e;border-radius:20px;height:20px;margin:12px 0;overflow:hidden">
                    <div style="width:{gauge_pct}%;background:{gauge_color};height:20px;
                                border-radius:20px;transition:width .5s"></div>
                  </div>
                  <div style="font-size:0.85rem;color:#aaa">
                    Ngưỡng: <b style="color:{gauge_color}">{threshold_val:.0%}</b> |
                    Gap: <b style="color:{gauge_color}">{(conf-threshold_val):+.1%}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with ml_right:
            st.markdown("**Trạng thái mô hình**")
            prec = float(model_meta.get("precision", 0))
            rec  = float(model_meta.get("recall", 0))
            f1   = float(model_meta.get("f1", 0))

            _threshold_bar("ML Confidence vs Threshold", conf, threshold_val)
            _threshold_bar("Model Precision",           prec, 0.45)
            _threshold_bar("Model Recall",              rec,  0.50)
            _threshold_bar("Model F1 Score",            f1,   0.48)

            status = "✅ PASS" if conf >= threshold_val else "❌ FAIL"
            st.markdown(
                f"""<div style="margin-top:10px;padding:10px;border-radius:6px;
                    background:{"#1a2e1a" if conf >= threshold_val else "#2e1a1a"};
                    border:1px solid {"#26a69a" if conf >= threshold_val else "#ef5350"};
                    font-size:1.1rem;font-weight:700;color:#fff;text-align:center">
                    ML Model: {status}
                </div>""",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Section 2: Strategy Components ───────────────────────────────────
        st.subheader("⚡ 2. Phân tích Strategy (ICT + Wyckoff + Momentum)")

        strat_left, strat_right = st.columns([1, 1])
        with strat_left:
            # Score decomposition bar chart using simple data
            score_abs = abs(score)
            regime_multiplier = {0: 0.5, 1: 1.0, 2: 1.2}.get(regime, 1.0)

            st.markdown("**Điểm chiến lược tổng hợp**")
            score_pct = min(score_abs / 1.2, 1.0) * 100
            score_color = GREEN if score_abs >= 0.3 else (AMBER if score_abs >= 0.1 else RED)
            score_status = "✅ ĐỦ MẠNH" if score_abs >= 0.3 else "❌ QUÁ YẾU"
            st.markdown(
                f"""
                <div style="text-align:center;padding:16px">
                  <div style="font-size:3rem;font-weight:900;color:{score_color}">{score:+.4f}</div>
                  <div style="font-size:0.9rem;color:#888">Strategy Score tổng hợp</div>
                  <div style="background:#1e1e2e;border-radius:20px;height:16px;margin:10px 0;overflow:hidden">
                    <div style="width:{score_pct:.0f}%;background:{score_color};height:16px;border-radius:20px"></div>
                  </div>
                  <div style="font-size:0.9rem;font-weight:700;color:{score_color}">{score_status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with strat_right:
            st.markdown("**Công thức tính Score**")
            st.markdown(
                f"""
                ```
                score = (
                    ICT_score   × ict_weight
                  + Wyckoff     × wyckoff_weight
                  + Momentum    × momentum_weight
                ) / total_weight × regime_multiplier

                regime_multiplier = {regime_multiplier}  ({r_label})

                Ngưỡng yêu cầu:
                  Sideway  → 0.05
                  Normal   → 0.30  ← hiện tại
                  Strong   → 0.30
                ```
                """
            )

        # Strategy components breakdown (from what we can infer)
        st.markdown("**3 thành phần chiến lược:**")
        comp_data = {
            "ICT (Liquidity Sweep + Trend Bias)": {
                "weight": "40%",
                "desc": "Phát hiện vùng thanh khoản bị quét, kiểm tra bias D1/H1",
                "requires": "liquidity_sweep=1 + trend_alignment=1",
            },
            "Wyckoff (Phase Analysis)": {
                "weight": "30%",
                "desc": "Nhận diện pha thị trường: Spring(1)=Bull, Upthrust(-1)=Bear",
                "requires": "wyckoff_phase ≠ 0",
            },
            "Momentum (RSI + MACD)": {
                "weight": "30%",
                "desc": "RSI > 55 + MACD > 0 → Bull | RSI < 45 + MACD < 0 → Bear",
                "requires": "rsi > 55 hoặc < 45, macd_hist cùng chiều",
            },
        }
        for comp_name, comp_info in comp_data.items():
            with st.expander(f"📌 {comp_name} — trọng số {comp_info['weight']}"):
                st.markdown(f"**Mô tả:** {comp_info['desc']}")
                st.markdown(f"**Điều kiện:** `{comp_info['requires']}`")

        st.divider()

        # ── Section 3: Volatility Regime ─────────────────────────────────────
        st.subheader("📊 3. Volatility Regime (Chế độ thị trường)")

        reg_left, reg_right = st.columns([1, 2])
        with reg_left:
            regime_info = {
                0: ("😴 SIDEWAY",   RED,   "Thị trường đi ngang. Score yêu cầu thấp hơn (0.05) nhưng multiplier = 0.5 → điểm bị giảm một nửa"),
                1: ("📊 NORMAL",    BLUE,  "Thị trường bình thường. Score yêu cầu ≥ 0.30, multiplier = 1.0"),
                2: ("⚡ STRONG VOL", GREEN, "Thị trường biến động mạnh. Score yêu cầu ≥ 0.30, multiplier = 1.2 → điểm được tăng 20%"),
            }
            rl, rc, rdesc = regime_info.get(regime, ("❓ Unknown", GREY, ""))
            st.markdown(
                f"""
                <div style="background:{rc}22;border:2px solid {rc};border-radius:10px;
                            padding:20px;text-align:center">
                  <div style="font-size:2rem;font-weight:800;color:{rc}">{rl}</div>
                  <div style="font-size:0.85rem;color:#aaa;margin-top:8px">{rdesc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with reg_right:
            st.markdown("**Phân bố regime trong lịch sử tín hiệu**")
            if "volatility_regime" in live_signals.columns:
                regime_counts = live_signals["volatility_regime"].map(
                    {0: "😴 Sideway", 1: "📊 Normal", 2: "⚡ Strong Vol"}
                ).value_counts()
                st.bar_chart(regime_counts, height=180)
            st.markdown(
                f"**Hiệu lực hiện tại:** multiplier = **{regime_multiplier}×** "
                f"→ Strategy score = {score:.4f} × {regime_multiplier} = **{score * regime_multiplier:.4f}**"
            )

        st.divider()

        # ── Section 4: Time Filter ────────────────────────────────────────────
        st.subheader("⏰ 4. Bộ lọc thời gian")
        now_utc = dt.datetime.utcnow()
        blocked_hours  = [7, 10, 11, 22]
        current_hour   = now_utc.hour
        is_blocked_now = current_hour in blocked_hours

        tf_left, tf_right = st.columns([1, 2])
        with tf_left:
            tf_colour = RED if is_blocked_now else GREEN
            tf_status = "🚫 GIỜ BỊ CHẶN" if is_blocked_now else "✅ GIỜ ĐƯỢC PHÉP"
            st.markdown(
                f"""
                <div style="background:{tf_colour}22;border:2px solid {tf_colour};border-radius:10px;
                            padding:20px;text-align:center">
                  <div style="font-size:1.6rem;font-weight:800;color:{tf_colour}">{tf_status}</div>
                  <div style="font-size:1rem;color:#fff;margin-top:8px">
                    Giờ UTC hiện tại: <b>{current_hour}:00</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tf_right:
            st.markdown("**Lịch giờ giao dịch trong ngày (UTC)**")
            hours_data = []
            for h in range(24):
                is_b = h in blocked_hours
                hours_data.append({
                    "Hour": f"{h:02d}:00",
                    "Status": "🚫 Blocked" if is_b else "✅ OK",
                    "Lý do": "News/Asian close/London open" if is_b else "Được phép giao dịch",
                })
            hours_df = pd.DataFrame(hours_data)
            st.dataframe(hours_df, use_container_width=True, height=250)

        st.divider()

        # ── Section 5: Decision Flowchart ─────────────────────────────────────
        st.subheader("🗺️ 5. Luồng ra quyết định của Bot")
        st.markdown(
            f"""
            ```
            📥 Nhận data MT5 (M15: 50k bars, H1: 25k, D1: 1.5k)
                          │
                          ▼
            🔧 Tính features (RSI, MACD, ATR, ICT, Wyckoff ...)
                          │
                          ▼
            🤖 ML Model dự đoán → probability = {conf:.1%}
                          │
              probability ≥ {threshold_val:.0%}?
              ├── ❌ KHÔNG → "Model confidence below threshold" → NO TRADE
              └── ✅ CÓ
                          │
                          ▼
            ⏰ Giờ có bị chặn không? (blocked_hours: {blocked_hours})
              ├── ❌ CÓ BỊ CHẶN → "blocked_hour" → NO TRADE
              └── ✅ KHÔNG
                          │
                          ▼
            📐 Trend alignment (D1 == H1 direction)?
              ├── ❌ NGƯỢC CHIỀU → "trend_misaligned" → NO TRADE
              └── ✅ CÙNG CHIỀU
                          │
                          ▼
            ⚡ Strategy Score ≥ ngưỡng regime ({score:.4f})?
              ├── ❌ YẾU → "strategy_score_too_weak" → NO TRADE
              └── ✅ ĐỦ
                          │
                          ▼
            ✅ VÀO LỆNH → Gửi Telegram + Đặt lệnh MT5
            ```
            """
        )

        # ── Section 6: History Comparison ────────────────────────────────────
        st.subheader("📊 6. So sánh tín hiệu vào lệnh vs không vào lệnh")
        if len(live_signals) > 5:
            traded_sigs    = live_signals[live_signals["should_trade"] == True]
            no_trade_sigs  = live_signals[live_signals["should_trade"] == False]

            cmp1, cmp2, cmp3, cmp4 = st.columns(4)
            cmp1.metric("Tổng tín hiệu",         len(live_signals))
            cmp2.metric("✅ Vào lệnh",            len(traded_sigs), delta=f"{len(traded_sigs)/len(live_signals):.1%}")
            cmp3.metric("🚫 Không vào",           len(no_trade_sigs))
            cmp4.metric("Conf TB (không vào)",    _pct(no_trade_sigs["confidence"].mean()) if not no_trade_sigs.empty else "n/a")

            if not traded_sigs.empty and not no_trade_sigs.empty:
                cmp_df = pd.DataFrame({
                    "Metric": ["Avg Confidence", "Avg |Strategy Score|"],
                    "Vào lệnh ✅": [
                        f"{traded_sigs['confidence'].mean():.1%}",
                        f"{traded_sigs['strategy_score'].abs().mean():.4f}" if "strategy_score" in traded_sigs.columns else "n/a",
                    ],
                    "Không vào 🚫": [
                        f"{no_trade_sigs['confidence'].mean():.1%}",
                        f"{no_trade_sigs['strategy_score'].abs().mean():.4f}" if "strategy_score" in no_trade_sigs.columns else "n/a",
                    ],
                })
                st.dataframe(cmp_df, use_container_width=True)

            # Reason breakdown
            if "reason" in live_signals.columns:
                st.markdown("**Phân bố lý do không vào lệnh:**")
                reason_counts = no_trade_sigs["reason"].value_counts() if not no_trade_sigs.empty else pd.Series(dtype=int)
                if not reason_counts.empty:
                    st.bar_chart(reason_counts, height=200)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.header("📊 Backtest Results")

    backtest_report = load_json(OUTPUTS / "backtest_report.json")
    trades          = load_csv(OUTPUTS / "backtest_trades.csv")
    dataset         = load_csv(OUTPUTS / "training_dataset.csv")
    trades          = enrich_trades_with_dataset_context(trades, dataset)
    trade_summary   = summarize_trades(trades)
    daily_summary, daily_frame = summarize_daily_performance(trades)
    training_report = load_json(OUTPUTS / "training_report.json")

    # KPI row 1
    left, middle, right = st.columns(3)
    left.metric("📈 Return %",        backtest_report.get("return_pct", "n/a"))
    middle.metric("💹 Profit Factor", backtest_report.get("profit_factor", "n/a"))
    right.metric("📉 Max Drawdown %", backtest_report.get("max_drawdown_pct", "n/a"))

    sub_left, sub_middle, sub_right = st.columns(3)
    sub_left.metric("📅 Test Days",    backtest_report.get("test_days", "n/a"))
    sub_middle.metric("📅 Trade Days", backtest_report.get("trade_days", "n/a"))
    sub_right.metric("📐 Sharpe",      backtest_report.get("sharpe_ratio", "n/a"))

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Test Start",  backtest_report.get("test_start", "n/a"))
    r2.metric("Trade End",   backtest_report.get("trade_end",  "n/a"))
    r3.metric("Test Days",   backtest_report.get("test_days",  "n/a"))
    r4.metric("Trade Days",  backtest_report.get("trade_days", "n/a"))

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Trades",         trade_summary["trades"])
    t2.metric("Wins / Losses",  f"{trade_summary['wins']} / {trade_summary['losses']}")
    t3.metric("Gross Profit",   _round(trade_summary["gross_profit"], 2))
    t4.metric("Gross Loss",     _round(trade_summary["gross_loss"],   2))

    n1, n2, n3 = st.columns(3)
    n1.metric("Net Profit",  _round(trade_summary["net_profit"], 2))
    n2.metric("Best Trade",  backtest_report.get("best_trade",  "n/a"))
    n3.metric("Worst Trade", backtest_report.get("worst_trade", "n/a"))

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Mean Daily %",   _round(daily_summary["mean_daily_return"],   3))
    d2.metric("Median Daily %", _round(daily_summary["median_daily_return"],  3))
    d3.metric("Best Day %",     _round(daily_summary["best_day_return"],      3))
    d4.metric("Worst Day %",    _round(daily_summary["worst_day_return"],     3))

    ds1, ds2, ds3 = st.columns(3)
    ds1.metric("Days ≥ 2%",  f"{daily_summary['share_ge_2']  * 100:.2f}%")
    ds2.metric("Days ≥ 5%",  f"{daily_summary['share_ge_5']  * 100:.2f}%")
    ds3.metric("Days ≥ 10%", f"{daily_summary['share_ge_10'] * 100:.2f}%")

    rc1, rc2 = st.columns(2)
    with rc1:
        st.subheader("Backtest Report")
        st.json(backtest_report)
    with rc2:
        st.subheader("Training Report")
        st.json(training_report)

    if not trades.empty:
        st.subheader("📈 Equity Curve")
        trades["time"] = pd.to_datetime(trades["time"], utc=True)
        st.line_chart(trades.set_index("time")["balance_after"])

        if not daily_frame.empty:
            st.subheader("📅 Daily Return Profile")
            daily_frame["date"] = pd.to_datetime(daily_frame["date"])
            dc1, dc2 = st.columns(2)
            with dc1:
                st.bar_chart(daily_frame.set_index("date")["return_pct"])
            with dc2:
                st.line_chart(daily_frame.set_index("date")[["net_pnl"]])
            st.dataframe(daily_frame.tail(120), use_container_width=True)

        st.subheader("📊 PnL Distribution")
        st.bar_chart(trades["pnl"].value_counts(bins=30).sort_index())

        st.subheader("📋 Trade List")
        st.dataframe(trades.tail(200), use_container_width=True)

        hourly_stats = trades.assign(hour=trades["time"].dt.hour).groupby("hour").agg(
            trades=("pnl", "size"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            win_rate=("is_win", "mean"),
        )
        if not hourly_stats.empty:
            st.subheader("⏰ Performance by Entry Hour (UTC)")
            hc1, hc2 = st.columns(2)
            with hc1:
                st.bar_chart(hourly_stats["total_pnl"])
            with hc2:
                st.bar_chart(hourly_stats["win_rate"])
            st.dataframe(hourly_stats.round(4), use_container_width=True)

        if "regime_label" in trades.columns:
            regime_stats = trades.groupby("regime_label").agg(
                trades=("pnl", "size"),
                total_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                win_rate=("is_win", "mean"),
            )
            if not regime_stats.empty:
                st.subheader("📊 Performance by Regime")
                reg1, reg2 = st.columns(2)
                with reg1:
                    st.bar_chart(regime_stats["total_pnl"])
                with reg2:
                    st.bar_chart(regime_stats["win_rate"])
                st.dataframe(regime_stats.round(4), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_walkforward:
    st.header("🔄 Walk-Forward Analysis")
    walkforward_report = load_json(OUTPUTS / "walkforward_report.json")
    walkforward_trades = load_csv(OUTPUTS / "walkforward_trades.csv")

    if walkforward_report:
        wf1, wf2, wf3, wf4 = st.columns(4)
        wf1.metric("Avg Return %",     walkforward_report.get("avg_return_pct",    "n/a"))
        wf2.metric("Avg Profit Factor",walkforward_report.get("avg_profit_factor", "n/a"))
        wf3.metric("Avg Precision",    _pct(walkforward_report.get("avg_precision")))
        wf4.metric("Avg Recall",       _pct(walkforward_report.get("avg_recall")))

        wf5, wf6, wf7 = st.columns(3)
        wf5.metric("Avg Drawdown %",  walkforward_report.get("avg_max_drawdown_pct", "n/a"))
        wf6.metric("Avg Trades/Fold", walkforward_report.get("avg_trades",           "n/a"))
        wf7.metric("Fallback Used",   "⚠️ Yes" if walkforward_report.get("fallback_used") else "✅ No")

        st.subheader("🏆 Best Params Found")
        st.json(walkforward_report.get("params", {}))

        if walkforward_report.get("folds"):
            folds = pd.DataFrame(walkforward_report["folds"])
            if "return_pct" in folds.columns:
                st.subheader("📊 Walk-Forward Fold Returns")
                st.bar_chart(folds.set_index("fold")["return_pct"])
            st.dataframe(folds, use_container_width=True)
    else:
        st.info("Chưa có dữ liệu walk-forward. Chạy lệnh `walkforward` để tạo.")

    if not walkforward_trades.empty:
        st.subheader("📋 Walk-Forward Trades")
        st.dataframe(walkforward_trades.tail(200), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PAPER TRADE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_paper:
    st.header("📝 Paper Trade Signals")
    paper_signals = live_signals.copy()  # reuse already-loaded data

    if not paper_signals.empty:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Total Signals", len(paper_signals))
        if "should_trade" in paper_signals.columns:
            trade_count = int(paper_signals["should_trade"].sum())
            p2.metric("✅ Trade Signals",    trade_count,
                      delta=f"{trade_count/len(paper_signals):.1%} trade rate")
            p3.metric("🚫 No-Trade Signals", len(paper_signals) - trade_count)
        if "confidence" in paper_signals.columns:
            p4.metric("Avg Confidence", _pct(paper_signals["confidence"].mean()))

        if "confidence" in paper_signals.columns:
            st.subheader("📈 Confidence Over Time")
            conf_df = paper_signals.set_index("time")["confidence"].dropna().sort_index()
            st.line_chart(conf_df)

        if "side" in paper_signals.columns:
            st.subheader("📊 Signal Direction Distribution")
            side_counts = paper_signals["side"].value_counts()
            st.bar_chart(side_counts)

        if "reason" in paper_signals.columns:
            st.subheader("❓ Lý do không vào lệnh — phân bố")
            no_t = paper_signals[paper_signals["should_trade"] == False]
            if not no_t.empty:
                st.bar_chart(no_t["reason"].value_counts(), height=220)

        st.subheader("📋 All Signals (200 gần nhất)")
        st.dataframe(paper_signals.head(200), use_container_width=True)
    else:
        st.info("Chưa có tín hiệu. Bot đang chạy...")
