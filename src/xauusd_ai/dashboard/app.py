"""XAUUSD AI Bot — Comprehensive Live Dashboard v2.0

Tabs:
  1. Live Monitor        — Bot status, account overview, latest signal
  2. Phan tich Chi tiet  — 6-step decision breakdown
  3. P&L & Von           — Equity curve, drawdown, win/loss streaks
  4. Hoc Lien Tuc        — Learning cycle, ROC-AUC improvement
  5. Backtest            — Historical backtest results
  6. Walk-Forward        — Walk-forward fold analysis
  7. Risk & Cai dat      — Lot calculator, position sizing
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "outputs"

GREEN  = "#26a69a"
RED    = "#ef5350"
AMBER  = "#ffa726"
BLUE   = "#42a5f5"
GREY   = "#90a4ae"
PURPLE = "#ab47bc"

_LIVE_COLS = [
    "time", "should_trade", "side", "confidence", "reason",
    "entry_price", "stop_loss", "take_profit", "volume",
    "strategy_score", "volatility_regime",
    "account_balance", "open_positions", "max_positions",
]
_PAPER_COLS = [
    "time", "should_trade", "side", "confidence", "reason",
    "entry_price", "stop_loss", "take_profit",
    "strategy_score", "volatility_regime",
]


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
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def load_signals() -> pd.DataFrame:
    """Load paper_trade_signals.csv — handle mixed 10/14-col schemas robustly."""
    path = OUTPUTS / "paper_trade_signals.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        # Need at least header + 1 data row
        data_lines = [l for l in lines[1:] if l.strip()]
        if not data_lines:
            return pd.DataFrame()
        # Detect max cols from data rows (naive split — may overcount if reason has comma)
        max_cols = max(len(l.split(",")) for l in data_lines)
        col_names = _LIVE_COLS if max_cols >= len(_LIVE_COLS) else _PAPER_COLS
        # Use header=None + skiprows=1 to avoid pandas header/names count mismatch
        df = pd.read_csv(
            path,
            header=None,
            names=col_names,
            skiprows=1,
            on_bad_lines="skip",
        )
        df = df[df["should_trade"].astype(str).str.lower().isin(["true", "false", "0", "1"])]
        df["should_trade"] = df["should_trade"].astype(str).str.lower().isin(["true", "1"])
        for col in ["confidence", "strategy_score", "entry_price", "stop_loss", "take_profit"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["volatility_regime", "open_positions", "max_positions"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(1).astype(int)
        if "account_balance" in df.columns:
            df["account_balance"] = pd.to_numeric(df["account_balance"], errors="coerce")
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"])
        return df.sort_values("time", ascending=False).reset_index(drop=True)
    except Exception as exc:
        # Surface the error as a Streamlit warning so we can debug
        try:
            import streamlit as _st
            _st.warning(f"load_signals error: {exc}")
        except Exception:
            pass
        return pd.DataFrame()


def _colour(val: float, good: bool = True) -> str:
    if good:
        return GREEN if val > 0 else (RED if val < 0 else GREY)
    return RED if val > 0 else (GREEN if val < 0 else GREY)


def _pct(val, dec: int = 1) -> str:
    if val is None:
        return "n/a"
    try:
        return f"{float(val):.{dec}%}"
    except Exception:
        return "n/a"


def _round(val, dec: int = 4):
    try:
        return round(float(val), dec)
    except Exception:
        return val


def _card(title: str, value: str, subtitle: str = "", colour: str = BLUE) -> str:
    return (
        f'<div style="background:{colour}22;border-left:4px solid {colour};'
        f'padding:12px 16px;border-radius:8px;margin-bottom:4px">'
        f'<div style="font-size:0.72rem;color:{colour};text-transform:uppercase;letter-spacing:.06em">{title}</div>'
        f'<div style="font-size:1.8rem;font-weight:700;color:#fff">{value}</div>'
        f'<div style="font-size:0.75rem;color:#aaa">{subtitle}</div>'
        f'</div>'
    )


def _threshold_bar(label: str, value: float, threshold: float, reverse: bool = False) -> None:
    pct    = min(abs(value) / max(abs(threshold) * 2, 1e-9), 1.0)
    passed = (value >= threshold) if not reverse else (value <= threshold)
    colour = GREEN if passed else RED
    icon   = "OK" if passed else "FAIL"
    st.markdown(
        f'<div style="margin-bottom:8px">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.85rem">'
        f'<span>[{icon}] <b>{label}</b></span>'
        f'<span style="color:{colour}"><b>{value:.4f}</b> / {threshold:.4f}</span>'
        f'</div>'
        f'<div style="background:#1e1e2e;border-radius:4px;height:8px">'
        f'<div style="width:{pct*100:.1f}%;background:{colour};height:8px;border-radius:4px"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _step_ok(step: int, label: str, passed: bool | None, detail: str = "") -> None:
    colour = GREEN if passed is True else (RED if passed is False else GREY)
    icon   = "PASS" if passed is True else ("FAIL" if passed is False else "INFO")
    st.markdown(
        f'<div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:10px;'
        f'padding:10px 14px;background:{colour}11;border-left:3px solid {colour};border-radius:6px">'
        f'<div style="font-size:1.2rem">[{icon}]</div>'
        f'<div>'
        f'<div style="font-size:0.78rem;color:{colour};font-weight:700">Step {step}</div>'
        f'<div style="font-size:1rem;color:#fff;font-weight:600">{label}</div>'
        f'<div style="font-size:0.82rem;color:#bbb;margin-top:2px">{detail}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def compute_streaks(is_win_series: pd.Series) -> dict:
    wins = losses = cur_win = cur_loss = max_win = max_loss = 0
    for w in is_win_series:
        if w:
            cur_win += 1; cur_loss = 0; wins += 1
        else:
            cur_loss += 1; cur_win = 0; losses += 1
        max_win  = max(max_win,  cur_win)
        max_loss = max(max_loss, cur_loss)
    return {
        "wins": wins, "losses": losses,
        "current_streak": cur_win if cur_win > 0 else -cur_loss,
        "max_win_streak": max_win,
        "max_loss_streak": max_loss,
    }


def compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd   = (equity - peak) / peak.replace(0, pd.NA) * 100
    return dd.fillna(0)


def load_learning_events() -> list[dict]:
    path = OUTPUTS / "live_learning_log.jsonl"
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except Exception:
            pass
    return events


def load_feature_importance() -> pd.DataFrame | None:
    mp = OUTPUTS / "model.pkl"
    if not mp.exists():
        return None
    try:
        with open(mp, "rb") as f:
            model = pickle.load(f)
        feature_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
        if hasattr(model, "coef_"):
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            if feature_names is None:
                feature_names = [f"feat_{i}" for i in range(len(coef))]
            df = pd.DataFrame({"feature": feature_names, "importance": coef})
            df["abs_importance"] = df["importance"].abs()
            return df.sort_values("abs_importance", ascending=False).head(25)
    except Exception:
        pass
    return None


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty or "pnl" not in trades.columns:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "gross_profit": 0.0, "gross_loss": 0.0, "net_profit": 0.0}
    wins = trades[trades["pnl"] > 0]
    losn = trades[trades["pnl"] <= 0]
    return {
        "trades":       len(trades),
        "wins":         len(wins),
        "losses":       len(losn),
        "win_rate":     len(wins) / max(len(trades), 1),
        "gross_profit": float(wins["pnl"].sum()),
        "gross_loss":   float(losn["pnl"].sum()),
        "net_profit":   float(trades["pnl"].sum()),
    }


def summarize_daily(trades: pd.DataFrame):
    empty_s = dict(mean_daily_return=0.0, median_daily_return=0.0,
                   best_day_return=0.0, worst_day_return=0.0,
                   share_ge_2=0.0, share_ge_5=0.0, share_ge_10=0.0)
    if trades.empty or "pnl" not in trades.columns:
        return empty_s, pd.DataFrame()
    d = trades.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time")
    d["date"] = d["time"].dt.date
    g = d.groupby("date").agg(
        trades=("pnl", "size"),
        net_pnl=("pnl", "sum"),
        gross_profit=("pnl", lambda s: s[s > 0].sum()),
        gross_loss=("pnl", lambda s: -s[s < 0].sum()),
        start_balance=("balance_before", "first"),
        end_balance=("balance_after", "last"),
    )
    g["return_pct"] = (g["end_balance"] / g["start_balance"] - 1.0) * 100.0
    return {
        "mean_daily_return":   float(g["return_pct"].mean()),
        "median_daily_return": float(g["return_pct"].median()),
        "best_day_return":     float(g["return_pct"].max()),
        "worst_day_return":    float(g["return_pct"].min()),
        "share_ge_2":  float((g["return_pct"] >= 2.0).mean()),
        "share_ge_5":  float((g["return_pct"] >= 5.0).mean()),
        "share_ge_10": float((g["return_pct"] >= 10.0).mean()),
    }, g.reset_index()


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="XAUUSD AI Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
.block-container{padding-top:1rem}
div[data-testid="metric-container"]{
    background:#1e1e2e;border-radius:8px;padding:10px 14px;border-left:3px solid #42a5f5}
</style>""", unsafe_allow_html=True)

st.title("📈 XAUUSD AI Bot — Live Dashboard v2")

_tunnel = OUTPUTS / "tunnel_url.txt"
if _tunnel.exists():
    _pub = _tunnel.read_text(encoding="utf-8").strip()
    if _pub:
        st.markdown(
            f'<div style="background:#0d2137;border:1px solid #42a5f5;border-radius:8px;'
            f'padding:10px 18px;margin-bottom:8px;display:flex;align-items:center;gap:16px">'
            f'<span style="font-size:1.3rem">🌐</span>'
            f'<div><span style="color:#90caf9;font-size:0.8rem">Public URL</span><br>'
            f'<a href="{_pub}" target="_blank" style="color:#42a5f5;font-size:1.05rem;font-weight:700">{_pub}</a></div>'
            f'<span style="margin-left:auto;background:#1565c0;color:#fff;padding:4px 10px;border-radius:4px;font-size:0.78rem">LIVE</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# Auto-refresh every 30 seconds when on Live Monitor tab
if _HAS_AUTOREFRESH:
    _st_autorefresh(interval=30_000, key="live_autorefresh")

(tab_live, tab_analysis, tab_pnl, tab_learning,
 tab_backtest, tab_walkforward, tab_risk) = st.tabs([
    "🟢 Live Monitor",
    "🔍 Phan tich Chi tiet",
    "📈 P&L & Von",
    "🧠 Hoc Lien Tuc",
    "📊 Backtest",
    "🔄 Walk-Forward",
    "⚙️ Risk & Cai dat",
])

# --- Shared data -------------------------------------------------------
model_meta      = load_json(OUTPUTS / "model_meta.json")
model_path      = OUTPUTS / "model.pkl"
live_signals    = load_signals()
learn_events    = load_learning_events()
backtest_report = load_json(OUTPUTS / "backtest_report.json")
training_report = load_json(OUTPUTS / "training_report.json")
trades          = load_csv(OUTPUTS / "backtest_trades.csv")
if not trades.empty and "time" in trades.columns:
    trades["time"] = pd.to_datetime(trades["time"], utc=True, errors="coerce")
    if "is_win" not in trades.columns and "pnl" in trades.columns:
        trades["is_win"] = trades["pnl"] > 0

threshold_val = float(
    model_meta.get("decision_threshold")
    or model_meta.get("selected_threshold")
    or 0.5
)

# =============================================================================
# TAB 1 — LIVE MONITOR
# =============================================================================
with tab_live:
    st.header("🟢 Live Bot Monitor")

    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        if model_path.exists():
            mtime = dt.datetime.fromtimestamp(model_path.stat().st_mtime)
            st.success(f"✅ Model aktif — trained: **{mtime.strftime('%Y-%m-%d %H:%M:%S')}**")
        else:
            st.error("❌ Model chua duoc train — chay `python main.py train` truoc")
    with hdr_r:
        if st.button("🔄 Lam moi"):
            st.rerun()

    if not live_signals.empty:
        latest  = live_signals.iloc[0]
        conf    = float(latest.get("confidence", 0) or 0)
        side    = str(latest.get("side", "flat"))
        score   = float(latest.get("strategy_score", 0) or 0)
        regime  = int(float(latest.get("volatility_regime", 1) or 1))
        traded  = bool(latest.get("should_trade", False))
        reason  = str(latest.get("reason", ""))
        r_label = {0: "Sideways", 1: "Normal", 2: "Strong Vol"}.get(regime, "?")
        side_c  = {"buy": GREEN, "sell": RED}.get(side, GREY)
        conf_c  = GREEN if conf >= threshold_val else (AMBER if conf >= threshold_val * 0.7 else RED)
        score_c = GREEN if abs(score) >= 0.3 else (AMBER if abs(score) >= 0.1 else RED)
        reg_c   = {0: AMBER, 1: BLUE, 2: GREEN}.get(regime, GREY)
        dec_c   = GREEN if traded else RED

        st.subheader("📡 Tin hieu moi nhat")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(_card("Tin hieu", {"buy": "BUY", "sell": "SELL"}.get(side, "FLAT"),
                          str(latest.get("time", ""))[:16], side_c), unsafe_allow_html=True)
        c2.markdown(_card("ML Confidence", f"{conf:.1%}", f"nguong {threshold_val:.0%}", conf_c), unsafe_allow_html=True)
        c3.markdown(_card("Strategy Score", f"{score:+.4f}", ">= 0.30 de vao lenh", score_c), unsafe_allow_html=True)
        c4.markdown(_card("Regime", r_label, "thi truong", reg_c), unsafe_allow_html=True)
        c5.markdown(_card("Quyet dinh", "VAO LENH" if traded else "KHONG VAO",
                          "" if traded else reason[:40], dec_c), unsafe_allow_html=True)

    if not live_signals.empty and "account_balance" in live_signals.columns:
        latest_s = live_signals.iloc[0]
        bal = float(latest_s.get("account_balance", 0) or 0)
        opn = int(float(latest_s.get("open_positions", 0) or 0))
        mx  = int(float(latest_s.get("max_positions", 1) or 1))
        vol = float(latest_s.get("volume", 0) or 0) if "volume" in live_signals.columns else 0.0

        st.subheader("💰 Trang thai Tai khoan")
        a1, a2, a3, a4 = st.columns(4)
        bal_c = GREEN if bal >= 200 else (AMBER if bal >= 100 else RED)
        a1.markdown(_card("Balance", f"${bal:,.2f}", "MT5 live balance", bal_c), unsafe_allow_html=True)
        pos_c = RED if opn >= mx else GREEN
        a2.markdown(_card("Lenh Dang Mo", f"{opn} / {mx}", "opened / max", pos_c), unsafe_allow_html=True)
        a3.markdown(_card("Lot Size", f"{vol:.3f}", "size lenh hien tai", BLUE), unsafe_allow_html=True)
        bar_pct = int((opn / max(mx, 1)) * 100)
        bar_c = GREEN if bar_pct < 70 else (AMBER if bar_pct < 100 else RED)
        a4.markdown(
            f'<div style="padding:12px 16px;background:#1e1e2e;border-radius:8px">'
            f'<div style="font-size:0.72rem;color:#aaa">Position Utilization</div>'
            f'<div style="margin-top:8px;background:#333;border-radius:6px;height:14px">'
            f'<div style="width:{bar_pct}%;background:{bar_c};height:14px;border-radius:6px"></div>'
            f'</div>'
            f'<div style="font-size:0.9rem;margin-top:4px;color:{bar_c}">{bar_pct}% su dung</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if opn >= mx:
            st.warning(f"Da day lenh: {opn}/{mx} — Bot khong mo them lenh moi")

    if model_meta:
        st.divider()
        st.subheader("🧠 Model Performance")
        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("Threshold", f"{threshold_val:.2f}")
        mm2.metric("Precision", _pct(model_meta.get("precision")))
        mm3.metric("Recall",    _pct(model_meta.get("recall")))
        mm4.metric("F1 Score",  _pct(model_meta.get("f1")))

    if not live_signals.empty and bool(live_signals.iloc[0].get("should_trade", False)):
        latest = live_signals.iloc[0]
        entry = float(latest.get("entry_price", 0) or 0)
        sl    = float(latest.get("stop_loss", 0) or 0)
        tp    = float(latest.get("take_profit", 0) or 0)
        rr    = abs((tp - entry) / (entry - sl)) if (sl and tp and entry and sl != entry) else 0
        st.divider()
        st.subheader("📍 Entry Plan")
        ep1, ep2, ep3, ep4 = st.columns(4)
        ep1.metric("Entry",       f"{entry:.3f}")
        ep2.metric("Stop Loss",   f"{sl:.3f}", delta=f"{sl - entry:+.2f}")
        ep3.metric("Take Profit", f"{tp:.3f}", delta=f"{tp - entry:+.2f}")
        ep4.metric("Risk/Reward", f"{rr:.2f}R")

    st.divider()
    st.subheader("📋 Lich su tin hieu gan nhat (50 muc)")
    if not live_signals.empty:
        disp = [c for c in ["time", "side", "confidence", "strategy_score",
                              "volatility_regime", "should_trade", "reason",
                              "account_balance", "open_positions"] if c in live_signals.columns]
        hist = live_signals[disp].head(50).copy()
        if "confidence" in hist.columns:
            hist["confidence"] = hist["confidence"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        st.dataframe(hist, use_container_width=True)
        if "confidence" in live_signals.columns:
            conf_ts = live_signals[["time", "confidence"]].dropna().set_index("time").sort_index()
            if not conf_ts.empty:
                st.subheader("📈 Confidence qua thoi gian")
                st.line_chart(conf_ts["confidence"], height=200)
    else:
        st.info("Chua co tin hieu. Bot dang chay...")


# =============================================================================
# TAB 2 — SIGNAL ANALYSIS
# =============================================================================
with tab_analysis:
    st.header("🔍 Phan tich Chi tiet Tin hieu")
    st.caption("Moi buoc bot can PASS de vao lenh — xem chi tiet tung dieu kien")

    if live_signals.empty:
        st.warning("Chua co tin hieu. Doi bot chay it nhat 1 chu ky.")
    else:
        latest  = live_signals.iloc[0]
        conf    = float(latest.get("confidence", 0) or 0)
        side    = str(latest.get("side", "flat"))
        score   = float(latest.get("strategy_score", 0) or 0)
        regime  = int(float(latest.get("volatility_regime", 1) or 1))
        traded  = bool(latest.get("should_trade", False))
        reason  = str(latest.get("reason", ""))
        r_label = {0: "Sideways", 1: "Normal", 2: "Strong Vol"}.get(regime, "?")
        reg_mult = {0: 0.5, 1: 1.0, 2: 1.2}.get(regime, 1.0)
        reg_thr  = {0: 0.05, 1: 0.30, 2: 0.30}.get(regime, 0.30)
        blocked_hours  = [7, 10, 11, 22]
        now_utc        = dt.datetime.utcnow()
        cur_hour       = now_utc.hour
        is_blocked_now = cur_hour in blocked_hours

        st.markdown(f"**Phan tich tai:** `{str(latest.get('time',''))[:19]}`")
        st.divider()

        # 6-step flow
        st.subheader("Luong ra quyet dinh — 6 buoc")
        fl, fr = st.columns([1, 1])
        with fl:
            ml_pass = conf >= threshold_val
            _step_ok(1, "ML Model Confidence",
                     ml_pass,
                     f"confidence = {conf:.1%} {'>=  ' if ml_pass else '< '} nguong {threshold_val:.0%}")
            time_pass = not is_blocked_now
            _step_ok(2, "Bo loc thoi gian",
                     time_pass,
                     f"UTC {cur_hour:02d}:xx — {'Gio duoc phep' if time_pass else f'Gio bi chan {blocked_hours}'}")
            _step_ok(3, "Che do thi truong",
                     None,
                     f"{r_label} — multiplier={reg_mult}x | score threshold={reg_thr}")
            score_pass = abs(score) >= reg_thr
            _step_ok(4, "Strategy Score (ICT+Wyckoff+Momentum)",
                     score_pass,
                     f"score={score:+.4f} abs={abs(score):.4f} {'>=  ' if score_pass else '< '}{reg_thr}")
            trend_ok = "trend" not in reason.lower()
            _step_ok(5, "Trend Alignment D1 == H1",
                     trend_ok if not traded else True,
                     "OK" if trend_ok else reason)
            if "account_balance" in live_signals.columns:
                opn_n = int(float(latest.get("open_positions", 0) or 0))
                mx_n  = int(float(latest.get("max_positions", 1) or 1))
                pos_ok = opn_n < mx_n
            else:
                pos_ok = "position" not in reason.lower()
            _step_ok(6, "Gioi han so lenh",
                     pos_ok if not traded else True,
                     f"{opn_n}/{mx_n}" if "account_balance" in live_signals.columns else "")

        with fr:
            dec_c = GREEN if traded else RED
            st.markdown(
                f'<div style="background:{dec_c}22;border:2px solid {dec_c};border-radius:12px;'
                f'padding:20px;text-align:center;margin-bottom:20px">'
                f'<div style="font-size:2.2rem;font-weight:900;color:{dec_c}">{"VAO LENH" if traded else "KHONG VAO"}</div>'
                f'<div style="font-size:0.9rem;color:#ccc;margin-top:8px">'
                f'{"Tat ca dieu kien da thoa man" if traded else reason}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            _threshold_bar("ML Confidence", conf, threshold_val)
            _threshold_bar("Strategy Score (abs)", abs(score), reg_thr)
            prec = float(model_meta.get("precision", 0))
            rec  = float(model_meta.get("recall", 0))
            f1v  = float(model_meta.get("f1", 0))
            _threshold_bar("Model Precision", prec, 0.45)
            _threshold_bar("Model Recall",    rec,  0.50)
            _threshold_bar("Model F1",        f1v,  0.48)

        st.divider()
        st.subheader("Strategy Components")
        sc_l, sc_r = st.columns([1, 1])
        with sc_l:
            spct = min(abs(score) / 1.2, 1.0) * 100
            sc   = GREEN if abs(score) >= 0.3 else (AMBER if abs(score) >= 0.1 else RED)
            st.markdown(
                f'<div style="text-align:center;padding:16px;background:#1e1e2e;border-radius:8px">'
                f'<div style="font-size:3rem;font-weight:900;color:{sc}">{score:+.4f}</div>'
                f'<div style="font-size:0.85rem;color:#888">Strategy Score</div>'
                f'<div style="background:#111;border-radius:20px;height:16px;margin:10px 0;overflow:hidden">'
                f'<div style="width:{spct:.0f}%;background:{sc};height:16px;border-radius:20px"></div>'
                f'</div>'
                f'<div style="font-size:0.9rem;font-weight:700;color:{sc}">{"Du manh" if abs(score) >= reg_thr else "Qua yeu"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with sc_r:
            st.code(
                f"score = (ICT x 0.40) + (Wyckoff x 0.30) + (Momentum x 0.30)\n"
                f"      x regime_multiplier\n\n"
                f"regime_multiplier = {reg_mult}  ({r_label})\n"
                f"nguong hien tai   = {reg_thr}\n\n"
                f"score hien tai = {score:.4f}\n"
                f"|score| x mult = {abs(score) * reg_mult:.4f}",
                language=None,
            )

        for cname, cweight, cdesc, ccond in [
            ("ICT (Liquidity Sweep + Trend Bias)", "40%",
             "Phat hien vung thanh khoan bi quet, kiem tra bias D1 va H1",
             "liquidity_sweep=1 + trend bias cung chieu"),
            ("Wyckoff (Phase Analysis)", "30%",
             "Spring(1)=Bullish, Upthrust(-1)=Bearish, 0=Neutral",
             "wyckoff_phase != 0"),
            ("Momentum (RSI + MACD)", "30%",
             "RSI>55 + MACD>0 = Bull | RSI<45 + MACD<0 = Bear",
             "rsi_long=55, rsi_short=45"),
        ]:
            with st.expander(f"{cname} — {cweight}"):
                st.markdown(f"**Mo ta:** {cdesc}")
                st.markdown(f"**Dieu kien:** `{ccond}`")

        st.divider()
        st.subheader("Volatility Regime")
        vr_l, vr_r = st.columns([1, 2])
        with vr_l:
            ri = {0: ("SIDEWAYS", AMBER, "Di ngang. threshold=0.05, mult=0.5"),
                  1: ("NORMAL",   BLUE,  "Binh thuong. threshold=0.30"),
                  2: ("STRONG",   GREEN, "Bien dong manh. threshold=0.30, mult=1.2")}
            rl, rc, rdesc = ri.get(regime, ("?", GREY, ""))
            st.markdown(
                f'<div style="background:{rc}22;border:2px solid {rc};border-radius:10px;'
                f'padding:20px;text-align:center">'
                f'<div style="font-size:2rem;font-weight:800;color:{rc}">{rl}</div>'
                f'<div style="font-size:0.82rem;color:#aaa;margin-top:8px">{rdesc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with vr_r:
            if "volatility_regime" in live_signals.columns:
                rc_counts = live_signals["volatility_regime"].map(
                    {0: "Sideways", 1: "Normal", 2: "Strong"}
                ).value_counts()
                if not rc_counts.empty:
                    st.bar_chart(rc_counts, height=180)
            st.markdown(f"=> Hien tai: **{r_label}** | multiplier = {reg_mult}x")

        st.divider()
        st.subheader("Bo loc thoi gian (UTC)")
        tf_l, tf_r = st.columns([1, 2])
        with tf_l:
            tf_c = RED if is_blocked_now else GREEN
            st.markdown(
                f'<div style="background:{tf_c}22;border:2px solid {tf_c};border-radius:10px;'
                f'padding:20px;text-align:center">'
                f'<div style="font-size:1.5rem;font-weight:800;color:{tf_c}">'
                f'{"Gio bi chan" if is_blocked_now else "Gio giao dich"}</div>'
                f'<div style="font-size:1rem;color:#fff;margin-top:8px">UTC {cur_hour:02d}:xx</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with tf_r:
            hours_df = pd.DataFrame([{
                "Gio UTC": f"{h:02d}:00",
                "Trang thai": "Bi chan" if h in blocked_hours else "OK",
            } for h in range(24)])
            st.dataframe(hours_df, use_container_width=True, height=240)

        st.divider()
        st.subheader("Luong quyet dinh day du")
        st.code(
            f"Data: D1=500, H1=2000, M15=5000 bars\n"
            f"=> Features: RSI, MACD, ATR, ICT, Wyckoff, Momentum\n"
            f"=> ML probability = {conf:.1%}\n"
            f"   [{('PASS' if conf >= threshold_val else 'FAIL')}] >= {threshold_val:.0%}?\n"
            f"=> Time filter UTC {cur_hour:02d}\n"
            f"   [{('PASS' if not is_blocked_now else 'FAIL')}] not in {blocked_hours}?\n"
            f"=> Strategy score = {score:.4f}\n"
            f"   [{('PASS' if abs(score) >= reg_thr else 'FAIL')}] |score| >= {reg_thr}?\n"
            f"=> Result: {'VAO LENH' if traded else 'KHONG VAO — ' + reason}",
            language=None,
        )

        st.divider()
        st.subheader("So sanh tin hieu vao vs khong vao lenh")
        if len(live_signals) >= 5:
            trd_s  = live_signals[live_signals["should_trade"] == True]
            ntrd_s = live_signals[live_signals["should_trade"] == False]
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Tong tin hieu", len(live_signals))
            cm2.metric("Da vao lenh",   len(trd_s), delta=f"{len(trd_s)/len(live_signals):.1%}")
            cm3.metric("Khong vao",     len(ntrd_s))
            cm4.metric("Conf TB khong vao", _pct(ntrd_s["confidence"].mean()) if not ntrd_s.empty else "n/a")
            if not trd_s.empty and not ntrd_s.empty:
                st.dataframe(pd.DataFrame({
                    "Metric": ["Avg Confidence", "Avg |Score|"],
                    "Vao lenh": [
                        f"{trd_s['confidence'].mean():.1%}",
                        f"{trd_s['strategy_score'].abs().mean():.4f}" if "strategy_score" in trd_s.columns else "n/a",
                    ],
                    "Khong vao": [
                        f"{ntrd_s['confidence'].mean():.1%}",
                        f"{ntrd_s['strategy_score'].abs().mean():.4f}" if "strategy_score" in ntrd_s.columns else "n/a",
                    ],
                }), use_container_width=True)
            if "reason" in live_signals.columns and not ntrd_s.empty:
                st.bar_chart(ntrd_s["reason"].value_counts(), height=200)

        # Feature importance
        st.divider()
        st.subheader("Feature Importance (Top 25 — Model Coefficients)")
        fi_df = load_feature_importance()
        if fi_df is not None and not fi_df.empty:
            st.bar_chart(fi_df.set_index("feature")["importance"].sort_values(), height=400)
            st.dataframe(fi_df[["feature", "importance", "abs_importance"]].round(6),
                         use_container_width=True)
        else:
            st.info("Chua load duoc model.pkl — can co model trained.")


# =============================================================================
# TAB 3 — P&L & VON
# =============================================================================
with tab_pnl:
    st.header("📈 P&L & Von — Loi nhuan va Thua lo")

    if not trades.empty:
        ts = summarize_trades(trades)
        ds, df = summarize_daily(trades)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Net Profit",    f"${ts['net_profit']:.2f}", delta=f"{ts['net_profit']:+.2f}")
        k2.metric("Win Rate",      f"{ts['win_rate']:.1%}")
        k3.metric("Total Trades",  ts["trades"])
        k4.metric("Profit Factor", backtest_report.get("profit_factor", "n/a"))

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Gross Profit",  f"${ts['gross_profit']:.2f}")
        k6.metric("Gross Loss",    f"${ts['gross_loss']:.2f}")
        k7.metric("Max Drawdown",  str(backtest_report.get("max_drawdown_pct", "n/a")) + "%")
        k8.metric("Sharpe Ratio",  str(backtest_report.get("sharpe_ratio", "n/a")))

        st.divider()
        st.subheader("Equity Curve & Drawdown")
        if "balance_after" in trades.columns:
            eq_df = trades.dropna(subset=["time", "balance_after"]).set_index("time").sort_index()
            if not eq_df.empty:
                eq_s = eq_df["balance_after"]
                dd_s = compute_drawdown(eq_s)
                eq_c, dd_c = st.columns([2, 1])
                with eq_c:
                    st.markdown("**Equity Curve**")
                    st.line_chart(eq_s, height=280)
                with dd_c:
                    st.markdown("**Drawdown (%)**")
                    st.area_chart(dd_s, height=280)
                min_dd = float(dd_s.min())
                st.markdown(
                    f'<div style="background:#2e1a1a;border-left:4px solid {RED};'
                    f'padding:10px 16px;border-radius:6px">'
                    f'<span style="color:{RED};font-weight:700">Max Drawdown: {min_dd:.2f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.subheader("Win / Loss Streaks")
        if "is_win" in trades.columns:
            stk = compute_streaks(trades.sort_values("time")["is_win"])
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Total Wins",      stk["wins"])
            s2.metric("Total Losses",    stk["losses"])
            cur = stk["current_streak"]
            cur_c = GREEN if cur > 0 else RED
            s3.markdown(_card("Current Streak", f"{abs(cur)}", "thang" if cur > 0 else "thua", cur_c),
                        unsafe_allow_html=True)
            s4.metric("Max Win Streak",  stk["max_win_streak"])
            s5.metric("Max Loss Streak", stk["max_loss_streak"])

            recent = trades.sort_values("time").tail(60)["is_win"].tolist()
            html = '<div style="display:flex;flex-wrap:wrap;gap:3px;padding:8px">'
            for w in recent:
                c = GREEN if w else RED
                html += f'<div style="width:18px;height:18px;background:{c};border-radius:3px" title="{"Win" if w else "Loss"}"></div>'
            html += "</div>"
            st.markdown("**Chui thang/thua 60 lenh gan nhat:**")
            st.markdown(html, unsafe_allow_html=True)

        st.divider()
        st.subheader("PnL Distribution")
        if "pnl" in trades.columns:
            pd1, pd2 = st.columns(2)
            with pd1:
                st.markdown("**Histogram PnL:**")
                st.bar_chart(trades["pnl"].value_counts(bins=20).sort_index(), height=220)
            with pd2:
                st.markdown("**PnL theo ngay:**")
                if not df.empty:
                    st.bar_chart(df.set_index("date")["net_pnl"], height=220)

        st.divider()
        st.subheader("Performance theo gio vao lenh (UTC)")
        if "time" in trades.columns:
            hourly = trades.assign(hour=trades["time"].dt.hour).groupby("hour").agg(
                trades=("pnl", "size"),
                total_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                win_rate=("is_win", "mean"),
            )
            if not hourly.empty:
                h1, h2 = st.columns(2)
                with h1:
                    st.markdown("**PnL theo gio:**")
                    st.bar_chart(hourly["total_pnl"], height=220)
                with h2:
                    st.markdown("**Win rate theo gio:**")
                    st.bar_chart(hourly["win_rate"], height=220)
                st.dataframe(hourly.round(4), use_container_width=True)

        st.divider()
        st.subheader("Danh sach lenh (200 gan nhat)")
        dcols = [c for c in ["time", "side", "entry_price", "exit_price",
                               "pnl", "is_win", "balance_after", "drawdown", "realized_rr"]
                  if c in trades.columns]
        st.dataframe(trades[dcols].tail(200), use_container_width=True)
    else:
        st.info("Chua co backtest data. Chay: python main.py backtest --config configs/settings.yaml")

    if not live_signals.empty and "account_balance" in live_signals.columns:
        st.divider()
        st.subheader("Live Balance History")
        bh = live_signals[["time", "account_balance"]].dropna().sort_values("time").set_index("time")
        if not bh.empty:
            st.area_chart(bh["account_balance"], height=220)
        if "open_positions" in live_signals.columns and "max_positions" in live_signals.columns:
            st.subheader("Open Positions History")
            ph = live_signals[["time", "open_positions", "max_positions"]].dropna().sort_values("time").set_index("time")
            st.line_chart(ph, height=180)


# =============================================================================
# TAB 4 — HOC LIEN TUC
# =============================================================================
with tab_learning:
    st.header("🧠 Hoc Lien Tuc — Self-Learning Monitor")
    st.caption("Bot tu retrain model khi co du du lieu moi, so sanh model moi vs cu, chi giu neu tot hon")

    # Learning cycle diagram
    st.subheader("Vong lap hoc lien tuc (Live Learning Cycle)")
    st.markdown(
        '<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:20px;font-family:monospace">'
        '<div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap">'
        '<div style="background:#1565c022;border:1px solid #1565c0;border-radius:8px;padding:10px 14px;text-align:center;min-width:110px">'
        '<div style="font-size:1.2rem">📥</div><div style="color:#90caf9;font-weight:700;font-size:0.78rem">BUOC 1</div>'
        '<div style="color:#fff;font-size:0.82rem">Fetch Data</div><div style="color:#888;font-size:0.7rem">MT5/CSV/yfinance</div></div>'
        '<div style="color:#555;font-size:1.4rem;padding:0 4px">→</div>'
        '<div style="background:#1b5e2022;border:1px solid #2e7d32;border-radius:8px;padding:10px 14px;text-align:center;min-width:110px">'
        '<div style="font-size:1.2rem">🔧</div><div style="color:#81c784;font-weight:700;font-size:0.78rem">BUOC 2</div>'
        '<div style="color:#fff;font-size:0.82rem">Feature Eng.</div><div style="color:#888;font-size:0.7rem">RSI/MACD/ATR/ICT</div></div>'
        '<div style="color:#555;font-size:1.4rem;padding:0 4px">→</div>'
        '<div style="background:#4a148c22;border:1px solid #7b1fa2;border-radius:8px;padding:10px 14px;text-align:center;min-width:110px">'
        '<div style="font-size:1.2rem">🤖</div><div style="color:#ce93d8;font-weight:700;font-size:0.78rem">BUOC 3</div>'
        '<div style="color:#fff;font-size:0.82rem">Retrain Model</div><div style="color:#888;font-size:0.7rem">LogisticRegression</div></div>'
        '<div style="color:#555;font-size:1.4rem;padding:0 4px">→</div>'
        '<div style="background:#e65100 22;border:1px solid #e65100;border-radius:8px;padding:10px 14px;text-align:center;min-width:110px">'
        '<div style="font-size:1.2rem">📊</div><div style="color:#ffba79;font-weight:700;font-size:0.78rem">BUOC 4</div>'
        '<div style="color:#fff;font-size:0.82rem">Evaluate</div><div style="color:#888;font-size:0.7rem">ROC-AUC / F1</div></div>'
        '<div style="color:#555;font-size:1.4rem;padding:0 4px">→</div>'
        '<div style="background:#26a69a22;border:1px solid #26a69a;border-radius:8px;padding:10px 14px;text-align:center;min-width:110px">'
        '<div style="font-size:1.2rem">💾</div><div style="color:#80cbc4;font-weight:700;font-size:0.78rem">BUOC 5</div>'
        '<div style="color:#fff;font-size:0.82rem">Deploy/Skip</div><div style="color:#888;font-size:0.7rem">Neu AUC tot hon</div></div>'
        '</div>'
        '<div style="color:#666;font-size:0.8rem;margin-top:12px;text-align:center">'
        'Lap lai sau moi 12 nen M15 moi (= 3 gio giao dich). Data bo sung tu yfinance moi 1 gio.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    if training_report:
        st.subheader("Ket qua Training gan nhat")
        tr1, tr2, tr3, tr4 = st.columns(4)
        tr1.metric("ROC-AUC",   _round(training_report.get("roc_auc"), 4))
        tr2.metric("Precision", _pct(training_report.get("precision")))
        tr3.metric("Recall",    _pct(training_report.get("recall")))
        tr4.metric("F1",        _pct(training_report.get("f1")))
        tr5, tr6 = st.columns(2)
        tr5.metric("Train Rows", training_report.get("train_rows", "n/a"))
        tr6.metric("Test Rows",  training_report.get("test_rows",  "n/a"))

    st.divider()

    sl_e = [e for e in learn_events if e.get("event") in ("self_learn", "live_retrain")]
    if sl_e:
        le1, le2, le3, le4 = st.columns(4)
        le1.metric("Tong lan retrain",       len(sl_e))
        improved = [e for e in sl_e if e.get("status") == "improved"]
        le2.metric("Lan cai thien model",    len(improved),
                   delta=f"{len(improved)/len(sl_e):.0%} ty le")
        best_roc = max((e.get("roc_auc", 0) for e in sl_e), default=0)
        le3.metric("Best ROC-AUC",           f"{best_roc:.4f}")
        last_ts = sl_e[-1].get("timestamp", "")[:16]
        le4.metric("Lan hoc cuoi",           last_ts)

        roc_data    = [e.get("roc_auc", 0) for e in sl_e]
        prec_data   = [e.get("precision", 0) for e in sl_e]
        recall_data = [e.get("recall", 0) for e in sl_e]
        f1_data     = [e.get("f1", 0) for e in sl_e]
        learn_df = pd.DataFrame({
            "ROC-AUC":   roc_data,
            "Precision": prec_data,
            "Recall":    recall_data,
            "F1":        f1_data,
        })
        st.markdown("**Tien trinh hoc: ROC-AUC / Precision / Recall qua cac lan retrain**")
        st.line_chart(learn_df, height=280)

        row_data = [e.get("dataset_rows", 0) for e in sl_e]
        if any(r > 0 for r in row_data):
            st.markdown("**Kich thuoc dataset qua cac lan retrain:**")
            st.bar_chart(pd.DataFrame({"dataset_rows": row_data}), height=180)

        status_counts = pd.Series([e.get("status", "unknown") for e in sl_e]).value_counts()
        st.markdown("**Ty le cai thien model:**")
        st.bar_chart(status_counts, height=160)

        st.subheader("Chi tiet cac lan tu hoc (30 gan nhat)")
        ev_df = pd.DataFrame(sl_e[-30:])
        dcols = [c for c in ["timestamp", "event", "status", "dataset_rows",
                               "roc_auc", "best_roc_auc", "precision", "recall", "f1",
                               "retrain_count"] if c in ev_df.columns]
        st.dataframe(ev_df[dcols] if dcols else ev_df, use_container_width=True)
    else:
        st.info(
            "He thong tu hoc chua kich hoat.\n\n"
            "Dieu kien: live_learning_enabled=true + bot chay live mode\n"
            "+ du 12 nen M15 moi (3 gio) + dataset > 1000 hang"
        )

    st.divider()
    st.subheader("Feature Importance — Dieu model dang hoc")
    fi_df = load_feature_importance()
    if fi_df is not None and not fi_df.empty:
        pos = fi_df[fi_df["importance"] > 0].sort_values("importance", ascending=False).head(12)
        neg = fi_df[fi_df["importance"] < 0].sort_values("importance").head(12)
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown(f"**Top BUY signals (coef > 0) — {len(pos)} features:**")
            if not pos.empty:
                st.bar_chart(pos.set_index("feature")["importance"], height=300)
        with fc2:
            st.markdown(f"**Top SELL signals (coef < 0) — {len(neg)} features:**")
            if not neg.empty:
                st.bar_chart(neg.set_index("feature")["importance"], height=300)
        with st.expander("Bang day du feature importance"):
            st.dataframe(fi_df[["feature", "importance", "abs_importance"]].round(6),
                         use_container_width=True)
    else:
        st.info("Chua co model. Chay training de xem feature importance.")

    st.divider()
    st.subheader("Cau hinh Self-Learning hien tai")
    st.code("""
# settings.yaml
training:
  live_learning_enabled: true          # Bat/tat tu hoc
  live_learning_min_new_bars: 12       # Toi thieu 12 nen M15 moi (3 gio)
  live_learning_min_rows: 1000         # Dataset toi thieu 1000 hang
  live_learning_interval_hours: 1      # Tan suat fetch yfinance

# Nguon data bo sung: yfinance XAUUSD=X / GC=F / GLD
# Gop voi data MT5 lich su -> dataset lon dan -> model hoc duoc nhieu hon
""", language="yaml")


# =============================================================================
# TAB 5 — BACKTEST
# =============================================================================
with tab_backtest:
    st.header("📊 Backtest Results")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Return %",        backtest_report.get("return_pct", "n/a"))
    b2.metric("Profit Factor",   backtest_report.get("profit_factor", "n/a"))
    b3.metric("Max Drawdown %",  backtest_report.get("max_drawdown_pct", "n/a"))
    b4.metric("Sharpe Ratio",    backtest_report.get("sharpe_ratio", "n/a"))

    b5, b6, b7, b8 = st.columns(4)
    b5.metric("Test Start",   backtest_report.get("test_start", "n/a"))
    b6.metric("Trade End",    backtest_report.get("trade_end",  "n/a"))
    b7.metric("Test Days",    backtest_report.get("test_days",  "n/a"))
    b8.metric("Trade Days",   backtest_report.get("trade_days", "n/a"))

    if not trades.empty:
        ts2 = summarize_trades(trades)
        ds2, df2 = summarize_daily(trades)

        bt1, bt2, bt3, bt4 = st.columns(4)
        bt1.metric("Total Trades",   ts2["trades"])
        bt2.metric("Wins / Losses",  f"{ts2['wins']} / {ts2['losses']}")
        bt3.metric("Win Rate",       f"{ts2['win_rate']:.1%}")
        bt4.metric("Net Profit",     f"${ts2['net_profit']:.2f}")

        st.divider()
        if "balance_after" in trades.columns:
            st.subheader("Backtest Equity Curve")
            eq2 = trades.dropna(subset=["time", "balance_after"]).set_index("time").sort_index()
            st.line_chart(eq2["balance_after"], height=280)

        if not df2.empty:
            st.subheader("Daily Return Profile")
            dd1, dd2 = st.columns(2)
            with dd1:
                st.bar_chart(df2.set_index("date")["return_pct"], height=200)
            with dd2:
                st.line_chart(df2.set_index("date")["net_pnl"], height=200)
            kd1, kd2, kd3, kd4 = st.columns(4)
            kd1.metric("Mean Daily %",   f"{ds2['mean_daily_return']:.3f}%")
            kd2.metric("Best Day %",     f"{ds2['best_day_return']:.3f}%")
            kd3.metric("Worst Day %",    f"{ds2['worst_day_return']:.3f}%")
            kd4.metric("Days >= 2%",     f"{ds2['share_ge_2'] * 100:.1f}%")

        if "pnl" in trades.columns:
            st.subheader("PnL Distribution")
            st.bar_chart(trades["pnl"].value_counts(bins=30).sort_index(), height=200)

        with st.expander("Backtest Report JSON"):
            st.json(backtest_report)
        with st.expander("Training Report JSON"):
            st.json(training_report)

        st.subheader("Trade List (tail 200)")
        dcols2 = [c for c in ["time", "side", "entry_price", "exit_price",
                                "pnl", "is_win", "balance_after", "drawdown", "realized_rr"]
                   if c in trades.columns]
        st.dataframe(trades[dcols2].tail(200), use_container_width=True)
    else:
        st.info("Chua co backtest data.")


# =============================================================================
# TAB 6 — WALK-FORWARD
# =============================================================================
with tab_walkforward:
    st.header("🔄 Walk-Forward Analysis")
    wf_r = load_json(OUTPUTS / "walkforward_report.json")
    wf_t = load_csv(OUTPUTS / "walkforward_trades.csv")

    if wf_r:
        wf1, wf2, wf3, wf4 = st.columns(4)
        wf1.metric("Avg Return %",       wf_r.get("avg_return_pct",    "n/a"))
        wf2.metric("Avg Profit Factor",  wf_r.get("avg_profit_factor", "n/a"))
        wf3.metric("Avg Precision",      _pct(wf_r.get("avg_precision")))
        wf4.metric("Avg Recall",         _pct(wf_r.get("avg_recall")))

        wf5, wf6, wf7 = st.columns(3)
        wf5.metric("Avg Drawdown %",   wf_r.get("avg_max_drawdown_pct", "n/a"))
        wf6.metric("Avg Trades/Fold",  wf_r.get("avg_trades",           "n/a"))
        wf7.metric("Fallback Used",    "Yes" if wf_r.get("fallback_used") else "No")

        st.subheader("Best Params")
        st.json(wf_r.get("params", {}))
        if wf_r.get("folds"):
            fd = pd.DataFrame(wf_r["folds"])
            if "return_pct" in fd.columns:
                st.bar_chart(fd.set_index("fold")["return_pct"])
            st.dataframe(fd, use_container_width=True)
    else:
        st.info("Chua co walk-forward data. Chay: python main.py walkforward --config configs/settings.yaml")

    if not wf_t.empty:
        st.subheader("Walk-Forward Trades")
        st.dataframe(wf_t.tail(200), use_container_width=True)


# =============================================================================
# TAB 7 — RISK & CAI DAT
# =============================================================================
with tab_risk:
    st.header("⚙️ Risk Management & Cai dat")

    st.subheader("Dynamic Position Sizing")
    st.markdown("""
| Balance            | Max Lenh | Risk/Lenh | Ghi chu          |
|--------------------|----------|-----------|------------------|
| < $200             | **1**    | 0.65%     | Micro account    |
| $200 – $500        | **2**    | 0.65%     | Small account    |
| $500 – $2,000      | **3**    | 0.65%     | Medium-low       |
| $2,000 – $10,000   | **5**    | 0.65%     | Medium           |
| $10,000 – $50,000  | **8**    | 0.65%     | Large account    |
| > $50,000          | **15**   | 0.65%     | Whale            |

**Dieu chinh Regime:**
- Sideways (regime=0): so lenh / 2
- Strong Volatile (regime=2): so lenh x 0.7
- Normal (regime=1): giu nguyen
""")

    st.divider()
    st.subheader("Lot Size Calculator")
    rc_l, rc_r = st.columns([1, 1])
    with rc_l:
        cb = st.number_input("Balance (USD)", min_value=10.0, max_value=1_000_000.0, value=200.0, step=50.0)
        cs = st.number_input("Stop Distance (USD)", min_value=0.1, max_value=200.0, value=3.6, step=0.5)
        cp = st.slider("Risk per Trade (%)", min_value=0.1, max_value=3.0, value=0.65, step=0.05)
    with rc_r:
        ra    = cb * (cp / 100)
        cl    = max(0.01, round((ra / (100.0 * cs)) / 0.01) * 0.01)
        ml    = cl * 100 * cs
        lc    = GREEN if cl <= 0.1 else AMBER
        st.markdown(_card("Lot Size", f"{cl:.2f}", "XAUUSD", lc), unsafe_allow_html=True)
        st.metric("Risk Amount",    f"${ra:.2f}")
        st.metric("Max Loss (SL)",  f"${ml:.2f}")
        st.metric("Risk/Balance",   f"{(ml/cb)*100:.2f}%")

    st.divider()
    if not live_signals.empty and "account_balance" in live_signals.columns:
        st.subheader("Live Balance History")
        bh2 = live_signals[["time", "account_balance"]].dropna().sort_values("time").set_index("time")
        if not bh2.empty:
            st.area_chart(bh2["account_balance"], height=200)

    st.divider()
    sl_ev = [e for e in learn_events if e.get("event") in ("self_learn", "live_retrain")]
    st.subheader("Self-Learning Events Summary")
    if sl_ev:
        sk1, sk2, sk3 = st.columns(3)
        sk1.metric("Total Retrains",  len(sl_ev))
        imp2 = sum(1 for e in sl_ev if e.get("status") == "improved")
        sk2.metric("Model Improved",  imp2)
        brv  = max((e.get("roc_auc", 0) for e in sl_ev), default=0)
        sk3.metric("Best ROC-AUC",    f"{brv:.4f}")
        if len(sl_ev) >= 2:
            st.line_chart(pd.DataFrame({"ROC-AUC": [e.get("roc_auc", 0) for e in sl_ev]}), height=180)
        ev2 = pd.DataFrame(sl_ev[-20:])
        dc2 = [c for c in ["timestamp", "status", "dataset_rows", "roc_auc",
                             "precision", "recall", "f1"] if c in ev2.columns]
        st.dataframe(ev2[dc2] if dc2 else ev2, use_container_width=True)
    else:
        st.info("Chua co su kien tu hoc. Bat live_learning_enabled=true va chay bot live.")
