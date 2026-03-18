"""XAUUSD AI Bot — Comprehensive Live Dashboard v3.0 ICT+Wyckoff

Model: HistGradientBoostingClassifier, 28 features
Features: D1(1) + H4(9: ICT) + H1(3: Wyckoff) + M15(15: execution)
Threshold: 0.55 | Walk-Forward: 19 folds, precision avg 54.7%, AUC std 0.0123

Tabs:
  1. Live Monitor        — Bot status, account overview, latest signal
  2. Phan tich Chi tiet  — 6-step decision breakdown (ICT→Wyckoff→Execution)
  3. P&L & Von           — Equity curve, drawdown, win/loss streaks
  4. Hoc Lien Tuc        — Learning cycle, ROC-AUC improvement, win/loss log
  5. Backtest            — Historical backtest results (ICT+Wyckoff model)
  6. Walk-Forward        — 19-fold walk-forward analysis
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


def load_signals(filename: str = "paper_trade_signals.csv") -> pd.DataFrame:
    """Load paper_trade_signals CSV — handle mixed 10/14-col schemas robustly."""
    path = OUTPUTS / filename
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
        # Keep the LAST written row per bar-timestamp (most up-to-date open_positions)
        df = df.drop_duplicates(subset=["time"], keep="last")
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


def load_learning_events(filename: str = "live_learning_log.jsonl") -> list[dict]:
    path = OUTPUTS / filename
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except Exception:
            pass
    return events


def load_wf_signals() -> pd.DataFrame:
    """Load walk-forward signals from walk-forward analysis."""
    path = OUTPUTS / "walkforward_signals_ict_wyckoff.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def load_live_closed_trades(filename: str = "live_closed_trades.csv") -> pd.DataFrame:
    """Load live closed trades logged by the live trading loop."""
    path = OUTPUTS / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        if "is_win" not in df.columns and "pnl" in df.columns:
            df["is_win"] = df["pnl"] > 0
        return df.sort_values("time").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def simulate_signal_outcomes(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Với mỗi tín hiệu (kể cả bị lọc), scan M15 price data về sau để xác định:
      - Nếu vào lệnh tại entry_price, TP hay SL hit trước?
      - P&L ước tính dựa trên volume (nếu có) hoặc 0.01 lot mặc định.
    Trả về signals DataFrame với thêm các cột: sim_outcome, sim_pnl, sim_bars.
    """
    M15_PATH = ROOT / "src" / "xauusd_ai" / "real_data" / "XAUUSDm_M15.csv"
    if not M15_PATH.exists() or signals.empty:
        return signals.copy()

    try:
        price = pd.read_csv(M15_PATH, parse_dates=["time"], on_bad_lines="skip")
        price["time"] = pd.to_datetime(price["time"], utc=True, errors="coerce")
        price = price.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    except Exception:
        return signals.copy()

    result = signals.copy()
    for col in ["sim_outcome", "sim_pnl_per_lot", "sim_pnl", "sim_bars"]:
        result[col] = None

    for idx, row in result.iterrows():
        try:
            entry = float(row.get("entry_price") or 0)
            sl    = float(row.get("stop_loss") or 0)
            tp    = float(row.get("take_profit") or 0)
            side  = str(row.get("side", "")).lower()
            vol   = float(row.get("volume") or 0.01)
            sig_time = pd.to_datetime(row.get("time"), utc=True, errors="coerce")
            if entry <= 0 or sl <= 0 or tp <= 0 or side not in ("buy", "sell") or pd.isna(sig_time):
                continue

            future = price[price["time"] > sig_time].head(200)  # max 200 bars ~50h
            if future.empty:
                continue

            outcome = None
            bars_taken = 0
            for _, bar in future.iterrows():
                bars_taken += 1
                h, l = float(bar["high"]), float(bar["low"])
                if side == "buy":
                    if l <= sl:
                        outcome = "loss"
                        break
                    if h >= tp:
                        outcome = "win"
                        break
                else:  # sell
                    if h >= sl:
                        outcome = "loss"
                        break
                    if l <= tp:
                        outcome = "win"
                        break

            if outcome is None:
                result.at[idx, "sim_outcome"] = "open"
                continue

            # P&L per lot: (TP/SL - entry) × contract_oz
            if side == "buy":
                pnl_per_lot = (tp - entry) * 100.0 if outcome == "win" else (sl - entry) * 100.0
            else:
                pnl_per_lot = (entry - tp) * 100.0 if outcome == "win" else (entry - sl) * 100.0

            lot = max(vol, 0.01)
            result.at[idx, "sim_outcome"]     = outcome
            result.at[idx, "sim_pnl_per_lot"] = round(pnl_per_lot, 2)
            result.at[idx, "sim_pnl"]         = round(pnl_per_lot * lot, 2)
            result.at[idx, "sim_bars"]        = bars_taken
        except Exception:
            continue

    return result


def load_win_loss_events() -> tuple[list[dict], list[dict]]:
    """Load win/loss JSONL from walk-forward analysis."""
    wins: list[dict] = []
    losses: list[dict] = []
    for path, store in [
        (OUTPUTS / "win_analysis_walkforward.jsonl", wins),
        (OUTPUTS / "loss_analysis_walkforward.jsonl", losses),
    ]:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    store.append(json.loads(line))
                except Exception:
                    pass
    return wins, losses


def load_feature_importance() -> pd.DataFrame | None:
    # Try ICT+Wyckoff model first, fall back to legacy model
    mp = OUTPUTS / "model_ict_wyckoff.pkl"
    if not mp.exists():
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
    page_title="XAUUSD AI Bot — Bảng điều khiển ICT+Wyckoff",
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

st.title("📈 XAUUSD AI Bot — Bảng điều khiển ICT+Wyckoff v3.0")

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
 tab_backtest, tab_walkforward, tab_risk, tab_data) = st.tabs([
    "🟢 Theo dõi Live",
    "🔍 Phân tích Chi tiết",
    "📈 P&L & Vốn",
    "🧠 Học Liên Tục",
    "📊 Kiểm nghiệm",
    "🔄 Walk-Forward",
    "⚙️ Rủi ro & Cài đặt",
    "🗄️ Dữ liệu MT5",
])

# --- Shared data -------------------------------------------------------
model_meta      = load_json(OUTPUTS / "model_meta_ict_wyckoff.json")
if not model_meta:
    model_meta  = load_json(OUTPUTS / "model_meta.json")  # fallback
model_path      = OUTPUTS / "model_ict_wyckoff.pkl"
if not model_path.exists():
    model_path  = OUTPUTS / "model.pkl"  # fallback
live_signals    = load_signals()
live_signals_acc2 = load_signals("paper_trade_signals_acc2.csv")
learn_events    = load_learning_events()
learn_events_acc2 = load_learning_events("live_learning_log_acc2.jsonl")
wf_win_events, wf_loss_events = load_win_loss_events()
backtest_report = load_json(OUTPUTS / "backtest_report_acc1.json")  # ACC1 retrain output
if not backtest_report:
    backtest_report = load_json(OUTPUTS / "backtest_report_ict_wyckoff.json")  # fallback old
if not backtest_report:
    backtest_report = load_json(OUTPUTS / "backtest_report.json")
training_report = load_json(OUTPUTS / "training_report_ict_wyckoff.json")
if not training_report:
    training_report = load_json(OUTPUTS / "training_report.json")  # fallback
trades          = load_csv(OUTPUTS / "backtest_trades_acc1.csv")  # ACC1 retrain output
if trades.empty:
    trades      = load_csv(OUTPUTS / "backtest_trades_ict_wyckoff.csv")  # fallback old
if trades.empty:
    trades      = load_csv(OUTPUTS / "backtest_trades.csv")
if not trades.empty and "time" in trades.columns:
    trades["time"] = pd.to_datetime(trades["time"], utc=True, errors="coerce")
    if "is_win" not in trades.columns and "pnl" in trades.columns:
        trades["is_win"] = trades["pnl"] > 0
# ACC2 (Model2) backtest data
backtest_report_acc2 = load_json(OUTPUTS / "backtest_report_acc2.json")
trades_acc2          = load_csv(OUTPUTS / "backtest_trades_acc2.csv")
if not trades_acc2.empty and "time" in trades_acc2.columns:
    trades_acc2["time"] = pd.to_datetime(trades_acc2["time"], utc=True, errors="coerce")
    if "is_win" not in trades_acc2.columns and "pnl" in trades_acc2.columns:
        trades_acc2["is_win"] = trades_acc2["pnl"] > 0
live_trades     = load_live_closed_trades()
live_trades_acc2 = load_live_closed_trades("live_closed_trades_acc2.csv")

threshold_val = float(
    model_meta.get("decision_threshold")
    or model_meta.get("selected_threshold")
    or 0.5
)
model_meta_acc2  = load_json(OUTPUTS / "model_meta2_weekly500.json")
threshold_val_acc2 = float(
    model_meta_acc2.get("decision_threshold")
    or model_meta_acc2.get("selected_threshold")
    or 0.55
)

# =============================================================================
# TAB 1 — LIVE MONITOR
# =============================================================================
with tab_live:
    st.header("🟢 Giám sát Bot Live")

    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        if model_path.exists():
            mtime = dt.datetime.fromtimestamp(model_path.stat().st_mtime)
            st.success(f"✅ Model ICT+Wyckoff đang hoạt động — đã train: **{mtime.strftime('%Y-%m-%d %H:%M:%S')}** | HistGBC 28 features")
        else:
            st.error("❌ Model chưa được train — chạy `python -m xauusd_ai.main train --config configs/train_ict_wyckoff_2022_2026.yaml`")
    with hdr_r:
        if st.button("🔄 Làm mới"):
            st.rerun()

    # ── 2-account quick overview ─────────────────────────────────────────
    def _acc_summary_card(label: str, signals: pd.DataFrame, acc_id: str) -> None:
        if signals.empty:
            st.markdown(
                f'<div style="background:#1e1e2e;border-left:4px solid #555;padding:12px 16px;border-radius:8px">'
                f'<div style="color:#aaa;font-size:0.8rem">{label} ({acc_id})</div>'
                f'<div style="color:#555;font-size:1.1rem">Chưa có dữ liệu</div>'
                f'</div>', unsafe_allow_html=True)
            return
        latest = signals.iloc[0]
        bal = float(latest.get("account_balance", 0) or 0)
        opn = int(float(latest.get("open_positions", 0) or 0))
        mx  = int(float(latest.get("max_positions", 1) or 1))
        conf = float(latest.get("confidence", 0) or 0)
        side = str(latest.get("side", "flat"))
        traded = bool(latest.get("should_trade", False))
        last_time = str(latest.get("time", ""))[:16]
        bal_c = GREEN if bal >= 200 else (AMBER if bal >= 100 else RED)
        side_emoji = {"buy": "📈", "sell": "📉"}.get(side, "➖")
        dec_c = GREEN if traded else "#888"
        st.markdown(
            f'<div style="background:#1e1e2e;border-left:4px solid {bal_c};padding:12px 16px;border-radius:8px">'
            f'<div style="color:#aaa;font-size:0.8rem">{label} <code>{acc_id}</code></div>'
            f'<div style="display:flex;gap:20px;align-items:center;margin-top:6px">'
            f'<span style="font-size:1.3rem;font-weight:700;color:{bal_c}">${bal:,.2f}</span>'
            f'<span style="color:#aaa">{opn}/{mx} lệnh</span>'
            f'<span style="color:{dec_c}">{side_emoji} {side.upper()} {conf:.0%}</span>'
            f'<span style="color:#666;font-size:0.75rem">{last_time}</span>'
            f'</div>'
            f'</div>', unsafe_allow_html=True)

    both_have_data = not live_signals.empty or not live_signals_acc2.empty
    if both_have_data:
        st.subheader("🏦 Tổng quan 2 Tài khoản")
        ov1, ov2 = st.columns(2)
        with ov1:
            _acc_summary_card("Acc 1 — Exness-MT5Trial17", live_signals, "270832477")
        with ov2:
            _acc_summary_card("Acc 2 — Exness-MT5Trial7", live_signals_acc2, "433326057")
        st.divider()

    # ── Account selector ─────────────────────────────────────────────────
    _acc_options = ["Acc 1 — 270832477 (Exness-MT5Trial17)", "Acc 2 — 433326057 (Exness-MT5Trial7)"]
    _sel_acc = st.radio("Xem chi tiết tài khoản:", _acc_options, horizontal=True, key="acc_selector")
    _selected_signals = live_signals if "Acc 1" in _sel_acc else live_signals_acc2
    threshold_val = threshold_val_acc2 if "Acc 2" in _sel_acc else float(
        model_meta.get("decision_threshold") or model_meta.get("selected_threshold") or 0.5
)

    if not _selected_signals.empty:
        latest  = _selected_signals.iloc[0]
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

        st.subheader("📡 Tín hiệu Mới nhất")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(_card("Tín hiệu", {"buy": "BUY", "sell": "SELL"}.get(side, "FLAT"),
                          str(latest.get("time", ""))[:16], side_c), unsafe_allow_html=True)
        c2.markdown(_card("ML Confidence", f"{conf:.1%}", f"ngưỡng {threshold_val:.0%}", conf_c), unsafe_allow_html=True)
        c3.markdown(_card("Strategy Score", f"{score:+.4f}", ">= 0.30 để vào lệnh", score_c), unsafe_allow_html=True)
        c4.markdown(_card("Chế độ", r_label, "thị trường", reg_c), unsafe_allow_html=True)
        c5.markdown(_card("Quyết định", "VÀO LỆNH" if traded else "KHÔNG VÀO",
                          "" if traded else reason[:40], dec_c), unsafe_allow_html=True)

    if not _selected_signals.empty and "account_balance" in _selected_signals.columns:
        latest_s = _selected_signals.iloc[0]
        # ── Đọc live_status.json để lấy open_positions realtime (không bị kẹt bởi bar timestamp) ──
        _status_file = "live_status_acc2.json" if "Acc 2" in _sel_acc else "live_status_acc1.json"
        _live_status = load_json(OUTPUTS / _status_file)
        bal = float(_live_status.get("account_balance") or latest_s.get("account_balance", 0) or 0)
        opn = int(float(_live_status.get("open_positions", latest_s.get("open_positions", 0)) or 0))
        mx  = int(float(_live_status.get("max_positions", latest_s.get("max_positions", 1)) or 1))
        vol = float(latest_s.get("volume", 0) or 0) if "volume" in _selected_signals.columns else 0.0

        st.subheader("💰 Trạng thái Tài khoản")
        a1, a2, a3, a4 = st.columns(4)
        bal_c = GREEN if bal >= 200 else (AMBER if bal >= 100 else RED)
        a1.markdown(_card("Số dư", f"${bal:,.2f}", "số dư MT5 live", bal_c), unsafe_allow_html=True)
        pos_c = RED if opn >= mx else GREEN
        a2.markdown(_card("Lệnh Đang Mở", f"{opn} / {mx}", "đang mở / tối đa", pos_c), unsafe_allow_html=True)
        a3.markdown(_card("Lot Size", f"{vol:.3f}", "kích thước lệnh hiện tại", BLUE), unsafe_allow_html=True)
        bar_pct = int((opn / max(mx, 1)) * 100)
        bar_c = GREEN if bar_pct < 70 else (AMBER if bar_pct < 100 else RED)
        a4.markdown(
            f'<div style="padding:12px 16px;background:#1e1e2e;border-radius:8px">'
            f'<div style="font-size:0.72rem;color:#aaa">Position Utilization</div>'
            f'<div style="margin-top:8px;background:#333;border-radius:6px;height:14px">'
            f'<div style="width:{bar_pct}%;background:{bar_c};height:14px;border-radius:6px"></div>'
            f'</div>'
            f'<div style="font-size:0.9rem;margin-top:4px;color:{bar_c}">{bar_pct}% sử dụng</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if opn >= mx:
            st.warning(f"Đã đầy lệnh: {opn}/{mx} — Bot không mở thêm lệnh mới")

    if model_meta:
        st.divider()
        st.subheader("🧠 Hiệu suất Model — ICT+Wyckoff HistGBC")
        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("Threshold", f"{threshold_val:.2f}")
        mm2.metric("Precision", _pct(model_meta.get("precision")))
        mm3.metric("Recall",    _pct(model_meta.get("recall")))
        mm4.metric("F1 Score",  _pct(model_meta.get("f1")))
        # Walk-Forward aggregate stats
        _wf_agg = load_json(OUTPUTS / "walkforward_report_ict_wyckoff.json").get("aggregate", {})
        if _wf_agg:
            st.caption("📊 Walk-Forward (19 folds, 2022–2026)")
            wm1, wm2, wm3, wm4 = st.columns(4)
            wm1.metric("WF Avg AUC",      f"{_wf_agg.get('avg_roc_auc', 0):.4f}",
                       delta=f"std={_wf_agg.get('std_roc_auc', 0):.4f}")
            wm2.metric("WF Avg Precision", _pct(_wf_agg.get("avg_precision")),
                       delta=f"range {_wf_agg.get('min_precision', 0):.2f}–{_wf_agg.get('max_precision', 0):.2f}")
            wm3.metric("WF Signal Win Rate", _pct(_wf_agg.get("signal_win_rate")),
                       delta=f"{_wf_agg.get('correct_signals', 0)}/{_wf_agg.get('total_signals', 0)} signals")
            wm4.metric("WF Avg F1",        _pct(_wf_agg.get("avg_f1")))

    if not _selected_signals.empty and bool(_selected_signals.iloc[0].get("should_trade", False)):
        latest = _selected_signals.iloc[0]
        entry = float(latest.get("entry_price", 0) or 0)
        sl    = float(latest.get("stop_loss", 0) or 0)
        tp    = float(latest.get("take_profit", 0) or 0)
        rr    = abs((tp - entry) / (entry - sl)) if (sl and tp and entry and sl != entry) else 0
        st.divider()
        st.subheader("📍 Kế hoạch vào lệnh")
        ep1, ep2, ep3, ep4 = st.columns(4)
        ep1.metric("Entry",       f"{entry:.3f}")
        ep2.metric("Stop Loss",   f"{sl:.3f}", delta=f"{sl - entry:+.2f}")
        ep3.metric("Take Profit", f"{tp:.3f}", delta=f"{tp - entry:+.2f}")
        ep4.metric("Risk/Reward", f"{rr:.2f}R")

    st.divider()
    st.subheader("📋 Tất cả tín hiệu — Entry / SL / TP / Kết quả (100 mục gần nhất)")
    if not _selected_signals.empty:
        _sig_disp = _selected_signals.copy().head(100)

        # ── Mô phỏng kết quả nếu vào lệnh (kể cả lệnh bị lọc) ──────────
        with st.spinner("Đang tính kết quả mô phỏng..."):
            _sig_sim = simulate_signal_outcomes(_sig_disp)

        # Format cột hiển thị
        _sig_sim["Thời gian"]    = _sig_sim["time"].astype(str).str[:16]
        _sig_sim["Tín hiệu"]     = _sig_sim["side"].str.upper()
        _sig_sim["Vào lệnh?"]    = _sig_sim["should_trade"].map(lambda x: "✅ CÓ" if x else "❌ KHÔNG")
        _sig_sim["Confidence"]   = _sig_sim["confidence"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        _sig_sim["Strategy Sc."] = _sig_sim["strategy_score"].map(lambda x: f"{x:+.4f}" if pd.notna(x) else "")
        for _c in ["entry_price", "stop_loss", "take_profit"]:
            if _c in _sig_sim.columns:
                _sig_sim[_c] = pd.to_numeric(_sig_sim[_c], errors="coerce")
        if "entry_price" in _sig_sim.columns and "stop_loss" in _sig_sim.columns and "take_profit" in _sig_sim.columns:
            _sig_sim["Entry"] = _sig_sim["entry_price"].map(lambda x: f"{x:.3f}" if pd.notna(x) and x > 0 else "—")
            _sig_sim["SL"]    = _sig_sim["stop_loss"].map(lambda x: f"{x:.3f}" if pd.notna(x) and x > 0 else "—")
            _sig_sim["TP"]    = _sig_sim["take_profit"].map(lambda x: f"{x:.3f}" if pd.notna(x) and x > 0 else "—")
            _sig_sim["R/R"]   = _sig_sim.apply(lambda r: (
                f"{abs((r['take_profit'] - r['entry_price']) / (r['entry_price'] - r['stop_loss'])):.2f}R"
                if (pd.notna(r['entry_price']) and pd.notna(r['stop_loss']) and pd.notna(r['take_profit'])
                    and r['entry_price'] > 0 and r['stop_loss'] > 0 and r['entry_price'] != r['stop_loss'])
                else "—"
            ), axis=1)
        _sig_sim["Lý do"]    = _sig_sim["reason"].astype(str).str[:55] if "reason" in _sig_sim.columns else ""
        if "volatility_regime" in _sig_sim.columns:
            _sig_sim["Regime"] = _sig_sim["volatility_regime"].map(
                lambda x: {0: "Sideways", 1: "Normal", 2: "Volatile"}.get(int(float(x)) if pd.notna(x) else 1, "?")
            )
        if "open_positions" in _sig_sim.columns and "max_positions" in _sig_sim.columns:
            _sig_sim["Pos"] = _sig_sim.apply(
                lambda r: f"{int(float(r['open_positions']))}/{int(float(r['max_positions']))}", axis=1
            )

        # ── Cột mô phỏng ──────────────────────────────────────────────
        def _fmt_sim_outcome(v):
            if v == "win":    return "✅ WIN"
            if v == "loss":   return "❌ LOSS"
            if v == "open":   return "⏳ Chưa KQ"
            return "—"

        _sig_sim["KQ mô phỏng"] = _sig_sim["sim_outcome"].map(_fmt_sim_outcome)
        _sig_sim["P&L sim ($)"] = _sig_sim["sim_pnl"].map(
            lambda x: f"+{x:.2f}" if pd.notna(x) and x > 0 else (f"{x:.2f}" if pd.notna(x) else "—")
        )
        _sig_sim["Bars đến KQ"] = _sig_sim["sim_bars"].map(
            lambda x: str(int(x)) if pd.notna(x) else "—"
        )

        # ── Thống kê mô phỏng ─────────────────────────────────────────
        _has_outcome = _sig_sim[_sig_sim["sim_outcome"].isin(["win", "loss"])]
        if not _has_outcome.empty:
            _sw = (_has_outcome["sim_outcome"] == "win").sum()
            _sl_c = (_has_outcome["sim_outcome"] == "loss").sum()
            _swr = _sw / len(_has_outcome)
            _spnl = _has_outcome["sim_pnl"].sum()
            _st1, _st2, _st3, _st4 = st.columns(4)
            _st1.metric("Tổng tín hiệu sim", len(_has_outcome),
                        delta=f"/ {len(_sig_sim)} tổng")
            _st2.metric("Win Rate (mô phỏng)", f"{_swr:.1%}",
                        delta=f"{_sw}W / {_sl_c}L")
            _st3.metric("P&L tổng (mô phỏng)", f"${_spnl:+.2f}")
            # Lệnh bị lọc nhưng sẽ win
            _filtered_win = _has_outcome[
                (_has_outcome["should_trade"] == False) & (_has_outcome["sim_outcome"] == "win")
            ]
            _st4.metric("Lọc bỏ nhưng WIN", len(_filtered_win),
                        delta="bỏ lỡ" if len(_filtered_win) > 0 else "ok")

        _show_cols = [c for c in [
            "Thời gian", "Tín hiệu", "Vào lệnh?", "Confidence", "Strategy Sc.",
            "Entry", "SL", "TP", "R/R", "KQ mô phỏng", "P&L sim ($)", "Bars đến KQ",
            "Regime", "Pos", "Lý do"
        ] if c in _sig_sim.columns]

        def _highlight_signal(row):
            kq = row.get("KQ mô phỏng", "")
            entered = row.get("Vào lệnh?", "") == "✅ CÓ"
            if kq == "✅ WIN" and entered:
                return ["background-color: #1b3a1b"] * len(row)   # xanh đậm = vào & win
            if kq == "✅ WIN" and not entered:
                return ["background-color: #0a2a1a"] * len(row)   # xanh nhạt = bỏ lỡ win
            if kq == "❌ LOSS" and entered:
                return ["background-color: #2a1a1a"] * len(row)   # đỏ đậm = vào & loss
            if kq == "❌ LOSS" and not entered:
                return ["background-color: #1a0f0f"] * len(row)   # đỏ nhạt = lọc đúng (tránh loss)
            return [""] * len(row)

        st.dataframe(_sig_sim[_show_cols].style.apply(_highlight_signal, axis=1),
                     use_container_width=True)
        st.caption(
            "🟢 Xanh đậm = vào lệnh & WIN  |  🟢 Xanh nhạt = bị lọc nhưng SẼ WIN (bỏ lỡ)  |"
            "  🔴 Đỏ đậm = vào lệnh & LOSS  |  🔴 Đỏ nhạt = bị lọc & tránh được LOSS  |"
            "  ⏳ = chưa có kết quả (giá chưa chạm TP/SL)"
        )

        # ── Closed trades với kết quả win/loss ──────────────────────────
        _closed = live_trades if "Acc 1" in _sel_acc else live_trades_acc2
        _acc_label_closed = "ACC1 (270832477)" if "Acc 1" in _sel_acc else "ACC2 (433326057)"
        if not _closed.empty:
            st.divider()
            st.subheader(f"💰 Lệnh đã đóng thật — {_acc_label_closed} ({len(_closed)} lệnh)")
            _closed_disp = _closed.copy()
            _closed_disp["Kết quả"]   = _closed_disp["is_win"].map(lambda x: "✅ THẮNG" if x else "❌ THUA")
            _closed_disp["P&L ($)"]   = _closed_disp["pnl"].map(lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}")
            _closed_disp["Entry"]     = _closed_disp["open_price"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            _closed_disp["Exit"]      = _closed_disp["close_price"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            _closed_disp["Lot"]       = _closed_disp["volume"].map(lambda x: f"{x:.2f}")
            _closed_disp["Thời gian"] = _closed_disp["time"].astype(str).str[:16]
            _closed_disp["Side"]      = _closed_disp["side"].str.upper() if "side" in _closed_disp.columns else ""
            _w = int(_closed_disp["is_win"].sum())
            _l = len(_closed_disp) - _w
            _total_pnl = _closed_disp["pnl"].sum()
            _wr = _w / len(_closed_disp) if len(_closed_disp) > 0 else 0
            _c1, _c2, _c3, _c4 = st.columns(4)
            _c1.metric("Tổng lệnh", len(_closed_disp))
            _c2.metric("Win Rate", f"{_wr:.1%}", delta=f"{_w}W / {_l}L")
            _c3.metric("Tổng P&L", f"${_total_pnl:+.2f}")
            _c4.metric("Avg P&L/lệnh", f"${_total_pnl/len(_closed_disp):+.2f}" if len(_closed_disp) > 0 else "$0")

            def _highlight_trade(row):
                if row.get("Kết quả", "") == "✅ THẮNG":
                    return ["background-color: #1b3a1b"] * len(row)
                return ["background-color: #2a1a1a"] * len(row)

            _cl_cols = [c for c in ["Thời gian", "Side", "Lot", "Entry", "Exit", "P&L ($)", "Kết quả"] if c in _closed_disp.columns]
            st.dataframe(
                _closed_disp[_cl_cols].sort_values("Thời gian", ascending=False).style.apply(_highlight_trade, axis=1),
                use_container_width=True,
            )
    else:
        st.info("Chưa có tín hiệu. Bot đang chạy...")


# =============================================================================
# TAB 2 — SIGNAL ANALYSIS
# =============================================================================
with tab_analysis:
    st.header("🔍 Phân tích Chi tiết Tín hiệu")
    st.caption("Mỗi bước bot cần PASS để vào lệnh — xem chi tiết từng điều kiện")

    _ana_options = ["Acc 1 — 270832477", "Acc 2 — 433326057"]
    _ana_sel = st.radio("Tài khoản:", _ana_options, horizontal=True, key="ana_acc_selector")
    _analysis_signals = live_signals if "Acc 1" in _ana_sel else live_signals_acc2
    _ana_threshold = threshold_val_acc2 if "Acc 2" in _ana_sel else float(
        model_meta.get("decision_threshold") or model_meta.get("selected_threshold") or 0.5
    )

    if _analysis_signals.empty:
        st.warning("Chưa có tín hiệu. Đợi bot chạy ít nhất 1 chu kỳ.")
    else:
        latest  = _analysis_signals.iloc[0]
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

        st.markdown(f"**Phân tích tại:** `{str(latest.get('time',''))[:19]}`")
        st.divider()

        # 6-step flow
        st.subheader("Luồng quyết định — 6 bước (ICT→Wyckoff→Execution)")
        fl, fr = st.columns([1, 1])
        with fl:
            ml_pass = conf >= _ana_threshold
            _step_ok(1, "Tin cậy ML Model",
                     ml_pass,
                     f"confidence = {conf:.1%} {'>=  ' if ml_pass else '< '} ngưỡng {_ana_threshold:.0%}")
            time_pass = not is_blocked_now
            _step_ok(2, "Bộ lọc thời gian",
                     time_pass,
                     f"UTC {cur_hour:02d}:xx — {'Giờ được phép' if time_pass else f'Giờ bị chặn {blocked_hours}'}")
            _step_ok(3, "Chế độ thị trường",
                     None,
                     f"{r_label} — multiplier={reg_mult}x | ngưỡng score={reg_thr}")
            score_pass = abs(score) >= reg_thr
            _step_ok(4, "Strategy Score (ICT+Wyckoff+Momentum)",
                     score_pass,
                     f"score={score:+.4f} abs={abs(score):.4f} {'>=  ' if score_pass else '< '}{reg_thr}")
            trend_ok = "trend" not in reason.lower()
            _step_ok(5, "Xu hướng D1 == H1",
                     trend_ok if not traded else True,
                     "OK" if trend_ok else reason)
            if "account_balance" in _analysis_signals.columns:
                opn_n = int(float(latest.get("open_positions", 0) or 0))
                mx_n  = int(float(latest.get("max_positions", 1) or 1))
                pos_ok = opn_n < mx_n
            else:
                pos_ok = "position" not in reason.lower()
            _step_ok(6, "Giới hạn số lệnh",
                     pos_ok if not traded else True,
                     f"{opn_n}/{mx_n}" if "account_balance" in _analysis_signals.columns else "")

        with fr:
            dec_c = GREEN if traded else RED
            st.markdown(
                f'<div style="background:{dec_c}22;border:2px solid {dec_c};border-radius:12px;'
                f'padding:20px;text-align:center;margin-bottom:20px">'
                f'<div style="font-size:2.2rem;font-weight:900;color:{dec_c}">{"VÀO LỆNH" if traded else "KHÔNG VÀO"}</div>'
                f'<div style="font-size:0.9rem;color:#ccc;margin-top:8px">'
                f'{"Tất cả điều kiện đã thỏa mãn" if traded else reason}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            _threshold_bar("ML Confidence", conf, _ana_threshold)
            _threshold_bar("Strategy Score (abs)", abs(score), reg_thr)
            prec = float(model_meta.get("precision", 0))
            rec  = float(model_meta.get("recall", 0))
            f1v  = float(model_meta.get("f1", 0))
            _threshold_bar("Model Precision", prec, 0.45)
            _threshold_bar("Model Recall",    rec,  0.50)
            _threshold_bar("Model F1",        f1v,  0.48)

        st.divider()
        st.subheader("Thành phần Chiến lược")
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
                f'<div style="font-size:0.9rem;font-weight:700;color:{sc}">{"Đủ mạnh" if abs(score) >= reg_thr else "Quá yếu"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with sc_r:
            st.code(
                f"score = (ICT x 0.40) + (Wyckoff x 0.30) + (Momentum x 0.30)\n"
                f"      x hệ số chế độ\n\n"
                f"hệ số chế độ = {reg_mult}  ({r_label})\n"
                f"ngưỡng hiện tại   = {reg_thr}\n\n"
                f"score hiện tại = {score:.4f}\n"
                f"|score| x mult = {abs(score) * reg_mult:.4f}",
                language=None,
            )

        for cname, cweight, cdesc, ccond in [
            ("H4 ICT Structure (BOS/ChoCH/FVG/OB/Confluence)", "60% trọng số",
             "H4: Break-of-Structure, Change-of-Character, Fair Value Gap, Order Block, "
             "Displacement, Equal-High/Low, Market Structure Bias, ICT Confluence, Premium/Discount",
             "h4_bos | h4_choch | h4_fvg | h4_order_block | h4_ict_confluence | h4_premium_discount"),
            ("H1 Wyckoff Phase Analysis", "20% trọng số",
             "Hourly bias xác nhận, VSA (Volume Spread Analysis), Wyckoff Spring/Upthrust detection. "
             "Spring(1)=Bullish, Upthrust(-1)=Bearish, 0=Trung lập",
             "wyckoff_spring_signal != 0 | vsa_signal != 0 | hourly_bias aligned"),
            ("M15 Tín hiệu Thực thi (RSI/MACD/ATR/Momentum)", "20% trọng số",
             "M15 thời điểm vào lệnh: RSI, MACD histogram, ATR ratio, range efficiency, liquidity sweep, "
             "order flow proxy, chế độ biến động, session return, tick volume zscore, spread, "
             "kill zone flag (London/NY open), Judas swing",
             "rsi>55+macd>0=Bull | rsi<45+macd<0=Bear | kill_zone_flag=1"),
        ]:
            with st.expander(f"{cname} — {cweight}"):
                st.markdown(f"**Mô tả:** {cdesc}")
                st.markdown(f"**Điều kiện:** `{ccond}`")

        st.divider()
        st.subheader("Chế độ biến động")
        vr_l, vr_r = st.columns([1, 2])
        with vr_l:
            ri = {0: ("SIDEWAYS", AMBER, "Đi ngang — ngưỡng=0.05, hệ số=0.5"),
                  1: ("NORMAL",   BLUE,  "Bình thường — ngưỡng=0.30"),
                  2: ("STRONG",   GREEN, "Biến động mạnh — ngưỡng=0.30, hệ số=1.2")}
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
            if "volatility_regime" in _analysis_signals.columns:
                rc_counts = _analysis_signals["volatility_regime"].map(
                    {0: "Sideways", 1: "Normal", 2: "Strong"}
                ).value_counts()
                if not rc_counts.empty:
                    st.bar_chart(rc_counts, height=180)
            st.markdown(f"=> Hệ số hiện tại: **{r_label}** | hệ số nhân = {reg_mult}x")

        st.divider()
        st.subheader("Bộ lọc thời gian (UTC)")
        tf_l, tf_r = st.columns([1, 2])
        with tf_l:
            tf_c = RED if is_blocked_now else GREEN
            st.markdown(
                f'<div style="background:{tf_c}22;border:2px solid {tf_c};border-radius:10px;'
                f'padding:20px;text-align:center">'
                f'<div style="font-size:1.5rem;font-weight:800;color:{tf_c}">'
                f'{"Giờ bị chặn" if is_blocked_now else "Giờ giao dịch"}</div>'
                f'<div style="font-size:1rem;color:#fff;margin-top:8px">UTC {cur_hour:02d}:xx</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with tf_r:
            hours_df = pd.DataFrame([{
                "Giờ UTC": f"{h:02d}:00",
                "Trạng thái": "Bị chặn" if h in blocked_hours else "OK",
            } for h in range(24)])
            st.dataframe(hours_df, use_container_width=True, height=240)

        st.divider()
        st.subheader("Luồng quyết định đầy đủ")
        st.code(
                f"Dữ liệu: D1=100, H4=200, H1=500, M15=300 nhịp\n"
                f"=> 28 Features: D1(1) + H4(9:ICT) + H1(3:Wyckoff) + M15(15:execution)\n"
            f"   [{('PASS' if conf >= _ana_threshold else 'FAIL')}] >= {_ana_threshold:.0%}?\n"
            f"=> Bộ lọc thời gian UTC {cur_hour:02d}\n"
            f"   [{('PASS' if not is_blocked_now else 'FAIL')}] not in {blocked_hours}?\n"
            f"=> Strategy score = {score:.4f}\n"
            f"   [{('PASS' if abs(score) >= reg_thr else 'FAIL')}] |score| >= {reg_thr}?\n"
            f"=> Kết quả: {'VÀO LỆNH' if traded else 'KHÔNG VÀO — ' + reason}",
            language=None,
        )

        st.divider()
        st.subheader("So sánh tín hiệu vào vs không vào lệnh")
        if len(_analysis_signals) >= 5:
            trd_s  = _analysis_signals[_analysis_signals["should_trade"] == True]
            ntrd_s = _analysis_signals[_analysis_signals["should_trade"] == False]
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Tổng tín hiệu", len(_analysis_signals))
            cm2.metric("Đã vào lệnh",   len(trd_s), delta=f"{len(trd_s)/len(_analysis_signals):.1%}")
            cm3.metric("Không vào",     len(ntrd_s))
            cm4.metric("Conf TB không vào", _pct(ntrd_s["confidence"].mean()) if not ntrd_s.empty else "n/a")
            if not trd_s.empty and not ntrd_s.empty:
                st.dataframe(pd.DataFrame({
                    "Metric": ["Avg Confidence", "Avg |Score|"],
                    "Vào lệnh": [
                        f"{trd_s['confidence'].mean():.1%}",
                        f"{trd_s['strategy_score'].abs().mean():.4f}" if "strategy_score" in trd_s.columns else "n/a",
                    ],
                    "Không vào": [
                        f"{ntrd_s['confidence'].mean():.1%}",
                        f"{ntrd_s['strategy_score'].abs().mean():.4f}" if "strategy_score" in ntrd_s.columns else "n/a",
                    ],
                }), use_container_width=True)
            if "reason" in _analysis_signals.columns and not ntrd_s.empty:
                st.bar_chart(ntrd_s["reason"].value_counts(), height=200)

        # Feature importance
        st.divider()
        st.subheader("28 Features — ICT+Wyckoff Model Map")
        feat_map_l, feat_map_r = st.columns([1, 1])
        with feat_map_l:
            st.markdown("""
**D1 (1 feature — Daily Bias)**
- `daily_bias` — D1 trend direction

**H4 (9 features — ICT Structure)**
- `h4_bos` — Break of Structure
- `h4_choch` — Change of Character
- `h4_fvg` — Fair Value Gap
- `h4_order_block` — Order Block presence
- `h4_displacement` — Displacement candle
- `h4_ehl` — Equal High/Low detection
- `h4_market_structure_bias` — Bias score
- `h4_ict_confluence` — ICT multi-factor confluence
- `h4_premium_discount` — Premium/Discount zone

**H1 (3 features — Wyckoff)**
- `hourly_bias` — H1 directional bias
- `vsa_signal` — Volume Spread Analysis
- `wyckoff_spring_signal` — Spring/Upthrust detection
""")
        with feat_map_r:
            st.markdown("""
**M15 (15 features — Execution)**
- `trend_alignment` — M15 trend vs H1
- `rsi` — RSI(14)
- `macd_hist` — MACD histogram
- `atr_ratio` — ATR normalized
- `range_efficiency` — Bar efficiency
- `liquidity_sweep` — Liquidity level swept
- `order_flow_proxy` — Order flow direction
- `wyckoff_phase` — Wyckoff phase (accumulation/dist.)
- `volatility_regime` — Sideways/Normal/Strong
- `session_return` — Session price return
- `tick_volume_zscore` — Volume Z-score
- `spread_points` — Current spread
- `strategy_score` — Combined ICT+Wyckoff+Mom score
- `kill_zone_flag` — London/NY open active
- `judas_swing_signal` — Fake move detection
""")

        st.divider()
        st.subheader("Feature Importance (Top 25 — HistGBC)")
        fi_df = load_feature_importance()
        if fi_df is not None and not fi_df.empty:
            st.bar_chart(fi_df.set_index("feature")["importance"].sort_values(), height=400)
            st.dataframe(fi_df[["feature", "importance", "abs_importance"]].round(6),
                         use_container_width=True)
        else:
            st.info("Chưa load được model_ict_wyckoff.pkl — cần có model trained. Chạy: python -m xauusd_ai.main train --config configs/train_ict_wyckoff_2022_2026.yaml")


# =============================================================================
# TAB 3 — P&L & VON
# =============================================================================
with tab_pnl:
    st.header("📈 P&L & Vốn — Lợi nhuận và Thua lỗ")

    # ── Live Account P&L sub-tabs (ACC1 / ACC2 / So Sanh) ────────────────────
    _pnl_acc1, _pnl_acc2, _pnl_both = st.tabs([
        "🏦 Live ACC1 — 270832477",
        "🏦 Live ACC2 — 433326057",
        "⚖️ So sánh 2 Tài khoản",
    ])

    def _render_live_pnl(lt: pd.DataFrame, ls: pd.DataFrame, acc_label: str) -> None:
        """Render live P&L section for one account."""
        if not lt.empty:
            lts = summarize_trades(lt)
            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("Net P&L Live",   f"${lts['net_profit']:.2f}", delta=f"{lts['net_profit']:+.2f}")
            lc2.metric("Win Rate",       f"{lts['win_rate']:.1%}")
            lc3.metric("Tổng lệnh",      lts["trades"])
            _pf = lts["gross_profit"] / max(abs(lts["gross_loss"]), 1e-9)
            lc4.metric("Profit Factor",  f"{_pf:.2f}")

            # Per-trade PnL bar
            if "pnl" in lt.columns:
                st.markdown("**P&L từng lệnh live:**")
                _pnl_idx = lt.set_index("time")["pnl"] if "time" in lt.columns else lt["pnl"]
                st.bar_chart(_pnl_idx, height=200)

            # Per-timeframe breakdown using volatility_regime
            if "pnl" in lt.columns:
                st.subheader("📊 P&L theo Khung Thời Gian (Chế độ)")
                _reg_map = {0: "Sideways", 1: "Normal", 2: "Strong Vol"}
                if "volatility_regime" in lt.columns:
                    _tf_grp = lt.groupby("volatility_regime").agg(
                        trades=("pnl", "size"),
                        net_pnl=("pnl", "sum"),
                        avg_pnl=("pnl", "mean"),
                        win_rate=("is_win", "mean") if "is_win" in lt.columns else ("pnl", lambda s: (s>0).mean()),
                    ).reset_index()
                    _tf_grp["regime_label"] = _tf_grp["volatility_regime"].map(_reg_map).fillna("Unknown")
                    st.dataframe(_tf_grp[["regime_label", "trades", "net_pnl", "avg_pnl", "win_rate"]].round(4),
                                 use_container_width=True, hide_index=True)

                # Hour-of-day P&L
                if "time" in lt.columns:
                    st.markdown("**P&L theo giờ (UTC):**")
                    _h_grp = lt.assign(hour=lt["time"].dt.hour).groupby("hour").agg(
                        net_pnl=("pnl", "sum"),
                        win_rate=("is_win", "mean") if "is_win" in lt.columns else ("pnl", lambda s: (s>0).mean()),
                        trades=("pnl", "size"),
                    )
                    if not _h_grp.empty:
                        _ha, _hb = st.columns(2)
                        with _ha:
                            st.bar_chart(_h_grp["net_pnl"], height=200)
                        with _hb:
                            st.bar_chart(_h_grp["win_rate"], height=200)

            # Cumulative equity from live closed trades
            if "pnl" in lt.columns and "time" in lt.columns:
                st.markdown("**Đường vốn lũy kế (lệnh live đã đóng):**")
                _lt_eq = lt.sort_values("time").copy()
                if "profit" in _lt_eq.columns:
                    _lt_eq["cum_pnl"] = _lt_eq["profit"].cumsum()
                else:
                    _lt_eq["cum_pnl"] = _lt_eq["pnl"].cumsum()
                st.line_chart(_lt_eq.set_index("time")["cum_pnl"], height=220)

            # Detailed trade table
            st.markdown("**Danh sách tất cả lệnh live đã đóng:**")
            _lcols = [c for c in ["time", "ticket", "side", "volume", "open_price", "close_price",
                                   "profit", "swap", "commission", "pnl", "is_win"]
                      if c in lt.columns]
            st.dataframe(lt[_lcols], use_container_width=True)
        else:
            st.info(f"Chưa có lệnh live nào được đóng cho {acc_label}.")

        # Balance history from signal log
        if not ls.empty and "account_balance" in ls.columns:
            st.subheader("Lịch sử Số dư Live")
            _bh = ls[["time", "account_balance"]].dropna().sort_values("time").set_index("time")
            if not _bh.empty:
                st.area_chart(_bh["account_balance"], height=200)

    with _pnl_acc1:
        _render_live_pnl(live_trades, live_signals, "ACC1 (270832477)")

    with _pnl_acc2:
        _render_live_pnl(live_trades_acc2, live_signals_acc2, "ACC2 (433326057)")

    with _pnl_both:
        st.subheader("⚖️ So sánh P&L 2 tài khoản")
        _cmp_data = []
        for _lbl, _lt in [("ACC1", live_trades), ("ACC2", live_trades_acc2)]:
            if not _lt.empty and "pnl" in _lt.columns:
                _s = summarize_trades(_lt)
                _pf2 = _s["gross_profit"] / max(abs(_s["gross_loss"]), 1e-9)
                _cmp_data.append({
                    "Tài khoản": _lbl,
                    "Tổng lệnh": _s["trades"],
                    "Thắng": _s["wins"],
                    "Thua": _s["losses"],
                    "Win Rate": f"{_s['win_rate']:.1%}",
                    "Net P&L": f"${_s['net_profit']:.2f}",
                    "Gross Profit": f"${_s['gross_profit']:.2f}",
                    "Gross Loss": f"${_s['gross_loss']:.2f}",
                    "Profit Factor": f"{_pf2:.2f}",
                })
        if _cmp_data:
            st.dataframe(pd.DataFrame(_cmp_data), use_container_width=True, hide_index=True)
            # Side-by-side cumulative equity curves
            _eq_combined = {}
            for _lbl, _lt in [("ACC1", live_trades), ("ACC2", live_trades_acc2)]:
                if not _lt.empty and "pnl" in _lt.columns and "time" in _lt.columns:
                    _lt2 = _lt.sort_values("time").copy()
                    _col = "profit" if "profit" in _lt2.columns else "pnl"
                    _eq_combined[_lbl] = _lt2.set_index("time")[_col].cumsum()
            if _eq_combined:
                st.markdown("**Đường vốn lũy kế 2 tài khoản:**")
                _eq_df = pd.DataFrame(_eq_combined)
                st.line_chart(_eq_df, height=280)
        else:
            st.info("Chưa có lệnh live nào được đóng. Dữ liệu sẽ hiển thị khi bot đóng lệnh.")

    st.divider()
    st.subheader("📊 Backtest P&L (lịch sử)")

    if not trades.empty:
        ts = summarize_trades(trades)
        ds, df = summarize_daily(trades)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Lợi nhuận ròng",    f"${ts['net_profit']:.2f}", delta=f"{ts['net_profit']:+.2f}")
        k2.metric("Tỷ lệ thắng",      f"{ts['win_rate']:.1%}")
        k3.metric("Tổng lệnh",  ts["trades"])
        k4.metric("Profit Factor", backtest_report.get("profit_factor", "n/a"))

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Gross Profit",  f"${ts['gross_profit']:.2f}")
        k6.metric("Gross Loss",    f"${ts['gross_loss']:.2f}")
        k7.metric("Max Drawdown",  str(backtest_report.get("max_drawdown_pct", "n/a")) + "%")
        k8.metric("Sharpe Ratio",  str(backtest_report.get("sharpe_ratio", "n/a")))

        st.divider()
        st.subheader("Đường vốn & Drawdown")
        if "balance_after" in trades.columns:
            eq_df = trades.dropna(subset=["time", "balance_after"]).set_index("time").sort_index()
            if not eq_df.empty:
                eq_s = eq_df["balance_after"]
                dd_s = compute_drawdown(eq_s)
                eq_c, dd_c = st.columns([2, 1])
                with eq_c:
                    st.markdown("**Đường vốn**")
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
        st.subheader("Chuỗi thắng/thua")
        if "is_win" in trades.columns:
            stk = compute_streaks(trades.sort_values("time")["is_win"])
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Tổng Thắng",      stk["wins"])
            s2.metric("Tổng Thua",    stk["losses"])
            cur = stk["current_streak"]
            cur_c = GREEN if cur > 0 else RED
            s3.markdown(_card("Chuỗi hiện tại", f"{abs(cur)}", "thắng" if cur > 0 else "thua", cur_c),
                        unsafe_allow_html=True)
            s4.metric("Chuỗi thắng cao nhất",  stk["max_win_streak"])
            s5.metric("Chuỗi thua cao nhất", stk["max_loss_streak"])

            recent = trades.sort_values("time").tail(60)["is_win"].tolist()
            html = '<div style="display:flex;flex-wrap:wrap;gap:3px;padding:8px">'
            for w in recent:
                c = GREEN if w else RED
                html += f'<div style="width:18px;height:18px;background:{c};border-radius:3px" title="{"Thắng" if w else "Thua"}"></div>'
            html += "</div>"
            st.markdown("**Chuỗi thắng/thua 60 lệnh gần nhất:**")
            st.markdown(html, unsafe_allow_html=True)

        st.divider()
        st.subheader("Phân phối P&L")
        if "pnl" in trades.columns:
            pd1, pd2 = st.columns(2)
            with pd1:
                st.markdown("**Biểu đồ P&L:**")
                st.bar_chart(trades["pnl"].value_counts(bins=20).sort_index(), height=220)
            with pd2:
                st.markdown("**P&L theo ngày:**")
                if not df.empty:
                    st.bar_chart(df.set_index("date")["net_pnl"], height=220)

        st.divider()
        st.subheader("Hiệu suất theo giờ vào lệnh (UTC)")
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
                    st.markdown("**P&L theo giờ:**")
                    st.bar_chart(hourly["total_pnl"], height=220)
                with h2:
                    st.markdown("**Tỷ lệ thắng theo giờ:**")
                    st.bar_chart(hourly["win_rate"], height=220)
                st.dataframe(hourly.round(4), use_container_width=True)

        st.divider()
        st.subheader("Danh sách lệnh (200 gần nhất)")
        dcols = [c for c in ["time", "side", "entry_price", "exit_price",
                               "pnl", "is_win", "balance_after", "drawdown", "realized_rr"]
                  if c in trades.columns]
        st.dataframe(trades[dcols].tail(200), use_container_width=True)
    else:
        st.info("Chưa có dữ liệu backtest. Chạy: python scripts/backtest_ict_wyckoff.py")


# =============================================================================
# TAB 4 — HOC LIEN TUC
# =============================================================================
with tab_learning:
    st.header("🧠 Học Liên Tục — Giám sát Tự học")
    st.caption("Bot tự retrain HistGBC khi có đủ dữ liệu mới, so sánh model mới vs cũ, chỉ giữ nếu tốt hơn — 28 features ICT+Wyckoff")

    # Per-account learning sub-tabs
    _learn_acc1_tab, _learn_acc2_tab, _learn_both_tab = st.tabs([
        "🤖 Học — ACC1 (270832477)",
        "🤖 Học — ACC2 (433326057)",
        "📊 So sánh & Tổng hợp",
    ])

    def _render_learning_tab(
        sl_events: list,
        acc_label: str,
        log_file: str = "live_bot_log.txt",
        interval_min: int = 30,
    ) -> None:
        """Comprehensive real-time learning status panel for one account."""
        import time as _time

        # ── Bot alive detection (log modified < 15 min ago) ──────────────────
        log_path = OUTPUTS / log_file
        bot_alive = False
        log_age_sec: float | None = None
        if log_path.exists():
            log_age_sec = _time.time() - log_path.stat().st_mtime
            bot_alive = log_age_sec < 900

        # ── Detect if currently training (scan last 120 lines of bot log) ────
        currently_training = False
        if bot_alive and log_path.exists():
            try:
                tail = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:]
                n_start = sum(1 for l in tail if "LearnerThread: regular retrain started" in l
                              or "LearnerThread: loss-retrain triggered" in l)
                n_done  = sum(1 for l in tail if "retrain done" in l)
                currently_training = n_start > n_done
            except Exception:
                pass

        # ── Parse events ─────────────────────────────────────────────────────
        sl_e   = [e for e in sl_events if e.get("event") in ("self_learn", "live_retrain", "loss_retrain")]
        loss_e = [e for e in sl_events if e.get("event") == "loss_retrain"]
        last_event: dict = sl_e[-1] if sl_e else {}
        last_ts: str = last_event.get("timestamp", "")

        # ── Compute countdown to next learning session ────────────────────────
        now_utc = dt.datetime.now(dt.timezone.utc)
        minutes_elapsed:   float | None = None
        minutes_remaining: float | None = None
        pct_progress = 0.0
        if last_ts:
            try:
                last_dt = dt.datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=dt.timezone.utc)
                elapsed_sec       = (now_utc - last_dt).total_seconds()
                minutes_elapsed   = elapsed_sec / 60
                minutes_remaining = max(0.0, interval_min - minutes_elapsed)
                pct_progress      = min(1.0, minutes_elapsed / interval_min)
            except Exception:
                pass

        # ── Determine state ──────────────────────────────────────────────────
        if not bot_alive:
            state, state_emoji = "OFFLINE",    "🔴"
            state_detail       = "Bot không chạy — hãy khởi động lại"
            state_bg, state_border = "#2a1010", "#ef5350"
        elif currently_training:
            state, state_emoji = "TRAINING",   "⚙️"
            state_detail       = "Đang retrain model trong background (giao dịch không bị gián đoạn)"
            state_bg, state_border = "#0d1f0d", "#66bb6a"
        elif not sl_e:
            state, state_emoji = "WAIT_FIRST", "🔄"
            state_detail       = "Chờ lần học đầu tiên — bot đã start, sẽ học sau ~30 phút"
            state_bg, state_border = "#12102a", "#ab47bc"
        elif minutes_remaining is not None and minutes_remaining < 2:
            state, state_emoji = "IMMINENT",   "⚙️"
            state_detail       = "Sắp đến giờ học hoặc đang chuẩn bị retrain..."
            state_bg, state_border = "#0d1f0d", "#66bb6a"
        else:
            state, state_emoji = "WAITING",    "⏱️"
            rem_str = f"{minutes_remaining:.0f}" if minutes_remaining is not None else "?"
            state_detail = f"Chờ {rem_str} phút nữa đến chu kỳ tự học tiếp theo"
            state_bg, state_border = "#0a1525", "#42a5f5"

        # ── Format helper strings ────────────────────────────────────────────
        bot_dot    = "🟢" if bot_alive else "🔴"
        bot_lbl    = "ĐANG CHẠY" if bot_alive else "KHÔNG CHẠY"
        if log_age_sec is None:
            age_str = "Không có log"
        elif log_age_sec < 60:
            age_str = f"{log_age_sec:.0f} giây trước"
        elif log_age_sec < 3600:
            age_str = f"{log_age_sec/60:.0f} phút trước"
        else:
            age_str = f"{log_age_sec/3600:.1f} giờ trước"
        last_learn_str = last_ts[:16].replace("T", " ") if last_ts else "Chưa có"
        retrain_count  = len(sl_e)

        # ── Status banner ────────────────────────────────────────────────────
        st.markdown(
            f'<div style="background:{state_bg};border:2px solid {state_border};'
            f'border-radius:12px;padding:18px 22px;margin-bottom:14px">'
            f'<div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap">'
            f'<div style="font-size:2.2rem;line-height:1">{state_emoji}</div>'
            f'<div style="flex:1;min-width:220px">'
            f'<div style="font-size:1.08rem;font-weight:700;color:#fff;margin-bottom:4px">'
            f'{state} — {state_detail}</div>'
            f'<div style="color:#bbb;font-size:0.82rem">'
            f'{bot_dot} Bot: <b>{bot_lbl}</b>&ensp;|&ensp;'
            f'Log cập nhật: <b>{age_str}</b>&ensp;|&ensp;'
            f'Đã retrain: <b>{retrain_count} lần</b>'
            f'</div></div>'
            f'<div style="text-align:right;min-width:130px">'
            f'<div style="color:#aaa;font-size:0.75rem;margin-bottom:2px">Học lần cuối</div>'
            f'<div style="color:#fff;font-weight:700;font-size:1rem">{last_learn_str}</div>'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

        # ── Progress bar + state message ─────────────────────────────────────
        if state == "WAITING" and minutes_remaining is not None:
            mins_done = interval_min - minutes_remaining
            st.markdown(
                f'<div style="color:#aaa;font-size:0.82rem;margin-bottom:4px">'
                f'⏰ Tiến độ chu kỳ: <b style="color:#fff">{mins_done:.0f}</b> / {interval_min} phút'
                f' &ensp;—&ensp; còn <b style="color:#42a5f5">{minutes_remaining:.0f} phút</b></div>',
                unsafe_allow_html=True,
            )
            st.progress(pct_progress)
            if minutes_remaining <= 5:
                st.success("🔔 Sắp đến giờ học! Bot sẽ retrain trong vài phút tới.")
        elif state in ("TRAINING", "IMMINENT"):
            st.progress(1.0)
            st.success("⚙️ **Bot đang retrain model** trong background — giao dịch vẫn tiếp tục bình thường.")
        elif state == "WAIT_FIRST":
            st.progress(0.0)
            st.warning("🔄 Bot mới start. Lần học đầu tiên sẽ diễn ra sau **~30 phút** kể từ khi khởi động.")
        elif state == "OFFLINE":
            st.error(
                "🔴 **Bot không chạy.** Khởi động bằng lệnh:\n\n"
                "```\n.venv\\Scripts\\python.exe scripts/live_runner.py live "
                f"--config configs/live_ict_wyckoff.yaml\n```"
            )

        # ── Checklist (khi chưa có event nào) ───────────────────────────────
        if not sl_e and bot_alive:
            st.markdown(
                '<div style="background:#1a1a2e;border:1px solid #444;border-radius:8px;'
                'padding:14px 18px;margin:10px 0">'
                '<b style="color:#fff">📋 Checklist kích hoạt tự học:</b><br/>'
                '<span style="color:#81c784">✅</span> <code>live_learning_enabled: true</code> trong config<br/>'
                '<span style="color:#81c784">✅</span> CSV lịch sử đã preload (3000 rows/timeframe)<br/>'
                '<span style="color:#ffa726">⏳</span> Chờ đủ 30 phút kể từ khi bot start<br/>'
                '<span style="color:#ffa726">⏳</span> Dataset > 500 rows (đã OK với preload)'
                '</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Metrics & charts (nếu đã có ít nhất 1 sự kiện học) ─────────────
        if sl_e:
            improved  = [e for e in sl_e if "improved" in e.get("status", "")]
            best_roc  = max((e.get("roc_auc", 0) for e in sl_e), default=0)
            avg_rows  = int(sum(e.get("dataset_rows", 0) for e in sl_e) / max(len(sl_e), 1))
            last_rows = last_event.get("dataset_rows", 0)

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Tổng retrain",        len(sl_e))
            mc2.metric("Model cải thiện",     len(improved),
                       delta=f"{len(improved)/max(len(sl_e),1):.0%} tỷ lệ")
            mc3.metric("Best ROC-AUC",        f"{best_roc:.4f}")
            mc4.metric("Loss retrain",        len(loss_e))
            mc5.metric("Avg dataset rows",    f"{avg_rows:,}")

            # Dataset size indicator
            row_ok_color = "#26a69a" if last_rows >= 500 else "#ef5350"
            row_ok_label = "✅ Đủ" if last_rows >= 500 else "❌ Thiếu"
            st.markdown(
                f'<div style="display:inline-block;background:#111;border:1px solid {row_ok_color};'
                f'border-radius:6px;padding:4px 12px;font-size:0.82rem;color:{row_ok_color};margin-bottom:10px">'
                f'Dataset lần cuối: <b>{last_rows:,} rows</b> {row_ok_label} (min: 500)</div>',
                unsafe_allow_html=True,
            )

            # Trend charts
            learn_df = pd.DataFrame({
                "ROC-AUC":   [e.get("roc_auc", 0)   for e in sl_e],
                "Precision": [e.get("precision", 0)  for e in sl_e],
                "Recall":    [e.get("recall", 0)     for e in sl_e],
                "F1":        [e.get("f1", 0)         for e in sl_e],
            })
            st.markdown(f"**📈 Tiến trình học {acc_label}: ROC-AUC / Precision / Recall**")
            st.line_chart(learn_df, height=240)

            c_l, c_r = st.columns(2)
            with c_l:
                row_data = [e.get("dataset_rows", 0) for e in sl_e]
                if any(r > 0 for r in row_data):
                    st.markdown("**Dataset size qua từng lần retrain:**")
                    st.bar_chart(pd.DataFrame({"dataset_rows": row_data}), height=160)
            with c_r:
                status_counts = pd.Series([e.get("status", "unknown") for e in sl_e]).value_counts()
                st.markdown("**Kết quả mỗi lần retrain:**")
                st.bar_chart(status_counts, height=160)

            # Detail table
            with st.expander(f"📋 Chi tiết {min(30, len(sl_e))} lần tự học gần nhất", expanded=len(sl_e) <= 5):
                ev_df = pd.DataFrame(sl_e[-30:])
                dcols = [c for c in [
                    "timestamp", "event", "status", "dataset_rows",
                    "roc_auc", "best_roc_auc", "precision", "recall", "f1",
                    "retrain_count", "triggered_by_losses",
                ] if c in ev_df.columns]
                st.dataframe(ev_df[dcols] if dcols else ev_df, use_container_width=True)

            # Loss retrain history
            if loss_e:
                st.divider()
                st.subheader("⚠️ Lịch sử Loss Retrain (retrain sau lệnh thua liên tiếp)")
                l_df  = pd.DataFrame(loss_e)
                l_cols = [c for c in [
                    "timestamp", "triggered_by_losses", "loss_patterns_used",
                    "dataset_rows", "roc_auc", "status",
                ] if c in l_df.columns]
                st.dataframe(l_df[l_cols] if l_cols else l_df, use_container_width=True)
        else:
            st.info(
                f"**Hệ thống tự học chưa có sự kiện nào cho {acc_label}.**\n\n"
                f"👉 Trạng thái: {'Bot đang chạy ✅ — chờ thêm ~30 phút' if bot_alive else 'Bot KHÔNG chạy ❌'}"
            )

    with _learn_acc1_tab:
        _render_learning_tab(learn_events, "ACC1", log_file="live_bot_log.txt")

    with _learn_acc2_tab:
        _render_learning_tab(learn_events_acc2, "ACC2", log_file="live_bot_log_acc2.txt")

    with _learn_both_tab:
        st.subheader("📊 So sánh tốc độ học 2 tài khoản")
        _lc_data = []
        for _acc_lbl, _evs in [("ACC1 (270832477)", learn_events), ("ACC2 (433326057)", learn_events_acc2)]:
            _sl = [e for e in _evs if e.get("event") in ("self_learn", "live_retrain", "loss_retrain")]
            if _sl:
                _imp  = sum(1 for e in _sl if "improved" in e.get("status", ""))
                _best = max((e.get("roc_auc", 0) for e in _sl), default=0)
                _lc_data.append({
                    "Tài khoản":      _acc_lbl,
                    "Tổng retrain":   len(_sl),
                    "Model cải thiện": _imp,
                    "Tỷ lệ cải thiện": f"{_imp/max(len(_sl),1):.1%}",
                    "Best AUC":       f"{_best:.4f}",
                    "Lần cuối":       _sl[-1].get("timestamp", "")[:16],
                })
        if _lc_data:
            st.dataframe(pd.DataFrame(_lc_data), use_container_width=True, hide_index=True)
            _auc_dfs = {}
            for _acc_lbl2, _evs2 in [("ACC1", learn_events), ("ACC2", learn_events_acc2)]:
                _sl2 = [e for e in _evs2 if e.get("event") in ("self_learn", "live_retrain")]
                if _sl2:
                    _auc_dfs[_acc_lbl2] = [e.get("roc_auc", 0) for e in _sl2]
            if _auc_dfs:
                _max_len = max(len(v) for v in _auc_dfs.values())
                _auc_df2 = pd.DataFrame({k: v + [None]*(_max_len - len(v)) for k, v in _auc_dfs.items()})
                st.markdown("**So sánh AUC qua từng lần retrain:**")
                st.line_chart(_auc_df2, height=240)
        else:
            st.info("Chưa có dữ liệu tự học (cả ACC1 và ACC2). Bot cần chạy ít nhất 30 phút.")

        # Learning cycle diagram
        st.divider()
        st.subheader("Vòng lặp học liên tục (Live Learning Cycle)")
        st.markdown(
            '<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:20px;font-family:monospace">'
            '<div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap">'
            '<div style="background:#1565c022;border:1px solid #1565c0;border-radius:8px;padding:10px 14px;text-align:center;min-width:110px">'
            '<div style="font-size:1.2rem">📥</div><div style="color:#90caf9;font-weight:700;font-size:0.78rem">BƯỚC 1</div>'
            '<div style="color:#fff;font-size:0.82rem">Fetch Data</div><div style="color:#888;font-size:0.7rem">MT5/CSV preload</div></div>'
            '<div style="color:#555;font-size:1.4rem;padding:0 4px">→</div>'
            '<div style="background:#1b5e2022;border:1px solid #2e7d32;border-radius:8px;padding:10px 14px;text-align:center;min-width:110px">'
            '<div style="font-size:1.2rem">🔧</div><div style="color:#81c784;font-weight:700;font-size:0.78rem">BƯỚC 2</div>'
            '<div style="color:#fff;font-size:0.82rem">Feature Eng.</div><div style="color:#888;font-size:0.7rem">RSI/MACD/ATR/ICT</div></div>'
            '<div style="color:#555;font-size:1.4rem;padding:0 4px">→</div>'
            '<div style="background:#4a148c22;border:1px solid #7b1fa2;border-radius:8px;padding:10px 14px;text-align:center;min-width:110px">'
            '<div style="font-size:1.2rem">🤖</div><div style="color:#ce93d8;font-weight:700;font-size:0.78rem">BƯỚC 3</div>'
            '<div style="color:#fff;font-size:0.82rem">Retrain Model</div><div style="color:#888;font-size:0.7rem">HistGBC 28feat</div></div>'
            '<div style="color:#555;font-size:1.4rem;padding:0 4px">→</div>'
            '<div style="background:#e6510022;border:1px solid #e65100;border-radius:8px;padding:10px 14px;text-align:center;min-width:110px">'
            '<div style="font-size:1.2rem">📊</div><div style="color:#ffba79;font-weight:700;font-size:0.78rem">BƯỚC 4</div>'
            '<div style="color:#fff;font-size:0.82rem">Evaluate</div><div style="color:#888;font-size:0.7rem">ROC-AUC / F1</div></div>'
            '<div style="color:#555;font-size:1.4rem;padding:0 4px">→</div>'
            '<div style="background:#26a69a22;border:1px solid #26a69a;border-radius:8px;padding:10px 14px;text-align:center;min-width:110px">'
            '<div style="font-size:1.2rem">💾</div><div style="color:#80cbc4;font-weight:700;font-size:0.78rem">BƯỚC 5</div>'
            '<div style="color:#fff;font-size:0.82rem">Deploy/Skip</div><div style="color:#888;font-size:0.7rem">Nếu AUC tốt hơn</div></div>'
            '</div>'
            '<div style="color:#666;font-size:0.8rem;margin-top:12px;text-align:center">'
            'Lặp lại sau mỗi <b>30 phút</b> — Data bổ sung từ MT5 history + CSV preload (3000 bars/timeframe) + yfinance.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    if training_report:
        st.subheader("Kết quả Training gần nhất")
        tr1, tr2, tr3, tr4 = st.columns(4)
        tr1.metric("ROC-AUC",   _round(training_report.get("roc_auc"), 4))
        tr2.metric("Precision", _pct(training_report.get("precision")))
        tr3.metric("Recall",    _pct(training_report.get("recall")))
        tr4.metric("F1",        _pct(training_report.get("f1")))
        tr5, tr6 = st.columns(2)
        tr5.metric("Train Rows", training_report.get("train_rows", "n/a"))
        tr6.metric("Test Rows",  training_report.get("test_rows",  "n/a"))

    st.divider()
    st.subheader("📊 Phân tích thắng/thua tín hiệu Walk-Forward")
    st.caption("Dữ liệu từ 19 folds walk-forward (2022–2026) — log thắng/thua theo từng tín hiệu")
    if wf_win_events or wf_loss_events:
        n_wins   = len(wf_win_events)
        n_losses = len(wf_loss_events)
        n_total  = n_wins + n_losses
        wl1, wl2, wl3, wl4 = st.columns(4)
        wl_c = GREEN if n_wins / max(n_total, 1) >= 0.50 else AMBER
        wl1.markdown(_card("Tổng tín hiệu", f"{n_total:,}", "Walk-Forward 19 folds", BLUE), unsafe_allow_html=True)
        wl2.markdown(_card("Đúng (Thắng)", f"{n_wins:,}", f"{n_wins/max(n_total,1):.1%}", GREEN), unsafe_allow_html=True)
        wl3.markdown(_card("Sai (Thua)", f"{n_losses:,}", f"{n_losses/max(n_total,1):.1%}", RED), unsafe_allow_html=True)
        wl4.markdown(_card("Tỷ lệ thắng WF", f"{n_wins/max(n_total,1):.1%}", "mục tiêu >= 50%", wl_c), unsafe_allow_html=True)

        if wf_win_events:
            win_df  = pd.DataFrame(wf_win_events)
            loss_df = pd.DataFrame(wf_loss_events) if wf_loss_events else pd.DataFrame()
            st.markdown("**Phan phoi tin hieu thang/thua theo fold:**")
            if "fold" in win_df.columns:
                fold_wins   = win_df.groupby("fold").size().rename("Wins")
                fold_losses = loss_df.groupby("fold").size().rename("Losses") if not loss_df.empty and "fold" in loss_df.columns else pd.Series(dtype=int, name="Losses")
                fold_chart  = pd.concat([fold_wins, fold_losses], axis=1).fillna(0)
                st.bar_chart(fold_chart, height=220)
            if "confidence" in win_df.columns:
                conf_cmp = pd.DataFrame({
                    "Confidence — Win":  win_df["confidence"].dropna(),
                    "Confidence — Loss": loss_df["confidence"].dropna() if not loss_df.empty and "confidence" in loss_df.columns else pd.Series(dtype=float),
                })
                w_l, w_r = st.columns(2)
                with w_l:
                    st.markdown("**Avg Confidence: Win vs Loss**")
                    st.dataframe(pd.DataFrame({
                        "Nhóm": ["Thắng (đúng)", "Thua (sai)"],
                        "Count": [n_wins, n_losses],
                        "Avg Confidence": [
                            f"{win_df['confidence'].mean():.3f}" if "confidence" in win_df.columns else "n/a",
                            f"{loss_df['confidence'].mean():.3f}" if not loss_df.empty and "confidence" in loss_df.columns else "n/a",
                        ],
                    }), use_container_width=True, hide_index=True)
                with w_r:
                    if "side" in win_df.columns:
                        side_wr = pd.DataFrame({
                            "Chiều": win_df["side"].dropna().value_counts().index.tolist(),
                            "Thắng": win_df["side"].dropna().value_counts().values.tolist(),
                        })
                        st.markdown("**Thắng theo chiều:**")
                        st.dataframe(side_wr, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có win/loss log từ walk-forward. Chạy `scripts/walkforward_ict_wyckoff.py` trước.")

    st.divider()
    st.subheader("Feature Importance — Điều model đang học")
    fi_df = load_feature_importance()
    if fi_df is not None and not fi_df.empty:
        # HistGBC uses feature_importances_, no negative coef → show top/bottom by abs
        top_n = fi_df.head(15)
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown(f"**Top 15 features quan trọng nhất:**")
            if not top_n.empty:
                st.bar_chart(top_n.set_index("feature")["abs_importance"], height=320)
        with fc2:
            has_coef = fi_df["importance"].lt(0).any()
            if has_coef:
                neg = fi_df[fi_df["importance"] < 0].sort_values("importance").head(12)
                st.markdown(f"**Top SELL signals (coef < 0) — {len(neg)} features:**")
                if not neg.empty:
                    st.bar_chart(neg.set_index("feature")["importance"], height=320)
            else:
                # HistGBC: show bottom 10 as least important
                bot = fi_df.tail(10)
                st.markdown("**10 features ít quan trọng nhất:**")
                if not bot.empty:
                    st.bar_chart(bot.set_index("feature")["abs_importance"], height=320)
        with st.expander("Bảng đầy đủ feature importance"):
            st.dataframe(fi_df[["feature", "importance", "abs_importance"]].round(6),
                         use_container_width=True)
    else:
        st.info("Chưa có model. Chạy: python -m xauusd_ai.main train --config configs/train_ict_wyckoff_2022_2026.yaml")

    st.divider()
    st.subheader("Cấu hình Self-Learning hiện tại")
    st.code("""
# configs/train_ict_wyckoff_2022_2026.yaml
training:
  live_learning_enabled: true          # Bat/tat tu hoc
  live_learning_min_new_bars: 12       # Toi thieu 12 nen M15 moi (3 gio)
  live_learning_min_rows: 1000         # Dataset toi thieu 1000 hang
  live_learning_interval_hours: 1      # Tan suat fetch yfinance

# Model: HistGradientBoostingClassifier (max_iter=500, lr=0.05, depth=6, balanced)
# 28 features: D1(1) + H4(9:ICT) + H1(3:Wyckoff) + M15(15:execution)
# Walk-Forward: 19 folds, AUC avg=0.6415 (std=0.0123), Precision avg=54.7%
# Signal threshold: 0.55 (saved in model_meta_ict_wyckoff.json)
""", language="yaml")


# =============================================================================
# TAB 5 — BACKTEST
# =============================================================================
def _render_backtest_panel(rpt: dict, trd: pd.DataFrame, training_rpt: dict | None = None, label: str = "") -> None:
    """Render backtest metrics + charts for one account."""
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Lợi nhuận %",    rpt.get("return_pct", "n/a"))
    b2.metric("Profit Factor",  rpt.get("profit_factor", "n/a"))
    b3.metric("Max Drawdown %", rpt.get("max_drawdown_pct", "n/a"))
    b4.metric("Sharpe Ratio",   rpt.get("sharpe_ratio", "n/a"))

    b5, b6, b7, b8 = st.columns(4)
    b5.metric("Bắt đầu",     rpt.get("start",       rpt.get("test_start", "n/a")))
    b6.metric("Kết thúc",    rpt.get("end",         rpt.get("trade_end",  "n/a")))
    b7.metric("Vốn đầu",    f"${rpt.get('starting_balance', rpt.get('initial_balance', 200)):.0f}")
    b8.metric("Vốn cuối",   f"${rpt.get('ending_balance',   rpt.get('final_balance', 0)):.2f}")

    if not trd.empty:
        ts2 = summarize_trades(trd)
        ds2, df2 = summarize_daily(trd)

        bt1, bt2, bt3, bt4 = st.columns(4)
        bt1.metric("Tổng lệnh",     ts2["trades"])
        bt2.metric("Thắng / Thua",  f"{ts2['wins']} / {ts2['losses']}")
        bt3.metric("Tỷ lệ thắng",   f"{ts2['win_rate']:.1%}")
        bt4.metric("Lợi nhuận ròng", f"${ts2['net_profit']:.2f}")

        st.divider()
        if "balance_after" in trd.columns:
            st.subheader("📈 Đường vốn Backtest")
            eq2 = trd.dropna(subset=["time", "balance_after"]).set_index("time").sort_index()
            st.line_chart(eq2["balance_after"], height=280)

        if not df2.empty:
            st.subheader("📅 Hiệu suất hàng ngày")
            dd1, dd2 = st.columns(2)
            with dd1:
                st.bar_chart(df2.set_index("date")["return_pct"], height=200)
            with dd2:
                st.line_chart(df2.set_index("date")["net_pnl"], height=200)
            kd1, kd2, kd3, kd4 = st.columns(4)
            kd1.metric("TB ngày %",      f"{ds2['mean_daily_return']:.3f}%")
            kd2.metric("Ngày tốt nhất %", f"{ds2['best_day_return']:.3f}%")
            kd3.metric("Ngày tệ nhất %",  f"{ds2['worst_day_return']:.3f}%")
            kd4.metric("Ngày >= 2%",      f"{ds2['share_ge_2'] * 100:.1f}%")

        if "pnl" in trd.columns:
            st.subheader("📊 Phân phối P&L")
            st.bar_chart(trd["pnl"].value_counts(bins=30).sort_index(), height=200)

        with st.expander(f"Backtest Report JSON {label}"):
            st.json(rpt)
        if training_rpt:
            with st.expander(f"Training Report JSON {label}"):
                st.json(training_rpt)

        st.subheader("Danh sách lệnh (200 cuối)")
        dcols2 = [c for c in ["time", "side", "entry_price", "exit_price",
                               "pnl", "is_win", "balance_after", "drawdown", "realized_rr"]
                  if c in trd.columns]
        st.dataframe(trd[dcols2].tail(200), use_container_width=True)
    else:
        st.info(f"Chưa có dữ liệu backtest ({label}). Chạy: python scripts/backtest_ict_wyckoff.py")


with tab_backtest:
    st.header("📊 Kết quả Backtest")
    _bt_acc1_tab, _bt_acc2_tab = st.tabs([
        "🏦 ACC1 — Model1 (ICT+Wyckoff, WR 76.8%)",
        "🏦 ACC2 — Model2 (Weekly $500, WR 66.1%)",
    ])
    with _bt_acc1_tab:
        st.subheader("ACC1 — Model1: threshold=0.61 | 832 trades | $200→$12,531")
        _render_backtest_panel(backtest_report, trades, training_report, "ACC1")
    with _bt_acc2_tab:
        st.subheader("ACC2 — Model2: threshold=0.55 | 1,285 trades | $200→$13,373 | Dynamic risk 3-5%")
        # Weekly breakdown table
        _wk2 = backtest_report_acc2.get("weekly", [])
        _active_wk2 = [w for w in _wk2 if w.get("trades_n", 0) > 0]
        if _active_wk2:
            st.subheader("📅 Weekly Breakdown")
            _wk_df = pd.DataFrame(_active_wk2)[["week", "trades_n", "pnl_sum", "wins_n", "win_rate"]]
            _wk_df.columns = ["Tuần", "Lệnh", "PnL ($)", "Thắng", "Win Rate"]
            _wk_df["PnL ($)"] = _wk_df["PnL ($)"].round(2)
            _wk_df["Win Rate"] = (_wk_df["Win Rate"] * 100).round(1).astype(str) + "%"
            st.dataframe(_wk_df, use_container_width=True)
            _wk_pnl = pd.DataFrame(_active_wk2).set_index("week")["pnl_sum"]
            st.subheader("📈 PnL theo tuần")
            st.bar_chart(_wk_pnl, height=250)
            st.divider()
        _render_backtest_panel(backtest_report_acc2, trades_acc2, None, "ACC2")


# =============================================================================
# TAB 6 — WALK-FORWARD
# =============================================================================
with tab_walkforward:
    _wf_hdr_r = load_json(OUTPUTS / "walkforward_report_ict_wyckoff.json")
    _wf_n = _wf_hdr_r.get("walk_forward", {}).get("n_folds", "?") if _wf_hdr_r else "?"
    st.header(f"🔄 Walk-Forward Analysis — ICT+Wyckoff {_wf_n} Folds")
    st.caption("Train=20,000 bars (~7 tháng) | Test=4,000 bars (~1.5 tháng) | Step=4,000 bars | 2022-10 → 2026-01")

    # ── Live progress tracking ──────────────────────────────────────────
    _pdata = load_json(OUTPUTS / "walkforward_progress.json")
    if _pdata:
        _pstatus  = _pdata.get("status", "")
        _pdone    = _pdata.get("completed_combinations", 0)
        _ptotal   = _pdata.get("total_combinations", 0)
        _ppct     = _pdata.get("pct_done", 0.0)
        _pelapsed = _pdata.get("elapsed_seconds", 0)
        _pmins, _psecs = int(_pelapsed // 60), int(_pelapsed % 60)
        _plast    = _pdata.get("last_updated", "")
        if _pstatus == "running":
            if _HAS_AUTOREFRESH:
                _st_autorefresh(interval=15_000, key="wf_progress_autorefresh")
            st.warning(
                f"⏳ Walk-Forward đang chạy: **{_pdone}/{_ptotal}** combinations "
                f"({_ppct:.1f}%) — {_pmins}m {_psecs}s elapsed"
            )
            st.progress(min(_ppct / 100.0, 1.0))
            _pb = _pdata.get("best_so_far")
            if _pb:
                _bpc1, _bpc2, _bpc3, _bpc4 = st.columns(4)
                _bpc1.metric("Best Return",    f"{_pb.get('avg_return_pct', 0):.2f}%")
                _bpc2.metric("Best PF",        f"{_pb.get('avg_profit_factor', 0):.3f}")
                _bpc3.metric("Best Drawdown",  f"{_pb.get('avg_max_drawdown_pct', 0):.2f}%")
                _bpc4.metric("Best Precision", f"{_pb.get('avg_precision', 0):.2%}")
                with st.expander("📋 Best params so far"):
                    st.json(_pb.get("params", {}))
            st.caption(f"Cập nhật lần cuối: {_plast}")
            st.divider()
        elif _pstatus == "done":
            st.success(f"✅ Walk-Forward hoàn thành! {_ptotal} combinations | {_pmins}m {_psecs}s")
            st.divider()

    wf_r = load_json(OUTPUTS / "walkforward_report_ict_wyckoff.json")
    if not wf_r:
        wf_r = load_json(OUTPUTS / "walkforward_report.json")  # fallback

    if wf_r:
        agg = wf_r.get("aggregate", {})
        wf_info = wf_r.get("walk_forward", {})

        # ── Top-level aggregate metrics ──────────────────────────────────────
        st.subheader("Tổng kết Walk-Forward")
        wf1, wf2, wf3, wf4 = st.columns(4)
        auc_avg = agg.get("avg_roc_auc", 0)
        auc_std = agg.get("std_roc_auc", 0)
        auc_c = GREEN if auc_std < 0.04 else AMBER
        wf1.markdown(_card("Avg ROC-AUC", f"{auc_avg:.4f}",
                           f"std={auc_std:.4f} {'✅ ổn định' if auc_std < 0.04 else '⚠️ cao'}",
                           auc_c), unsafe_allow_html=True)

        prec_avg = agg.get("avg_precision", 0)
        prec_c = GREEN if prec_avg >= 0.52 else AMBER
        wf2.markdown(_card("Avg Precision", f"{prec_avg:.1%}",
                           f"range {agg.get('min_precision', 0):.2f}–{agg.get('max_precision', 0):.2f}",
                           prec_c), unsafe_allow_html=True)

        sig_wr = agg.get("signal_win_rate", 0)
        sig_c = GREEN if sig_wr >= 0.50 else AMBER
        wf3.markdown(_card("Tỷ lệ thắng tín hiệu", f"{sig_wr:.1%}",
                           f"{agg.get('correct_signals', 0):,} / {agg.get('total_signals', 0):,}",
                           sig_c), unsafe_allow_html=True)

        wf4.markdown(_card("Folds", f"{wf_info.get('n_folds', 'n/a')}",
                           f"model: HistGBC 28 feat", BLUE), unsafe_allow_html=True)

        wf5, wf6, wf7, wf8 = st.columns(4)
        wf5.metric("Avg Precision",    _pct(agg.get("avg_precision")))
        wf6.metric("Avg Recall",       _pct(agg.get("avg_recall")))
        wf7.metric("Avg F1",           _pct(agg.get("avg_f1")))
        wf8.metric("Avg Accuracy",     _pct(agg.get("avg_accuracy")))

        wf9, wf10, wf11, wf12 = st.columns(4)
        wf9.metric("AUC Min",          f"{agg.get('min_roc_auc', 0):.4f}")
        wf10.metric("AUC Max",         f"{agg.get('max_roc_auc', 0):.4f}")
        wf11.metric("Avg Signal Rate", _pct(agg.get("avg_signal_rate")))
        n_prec50 = sum(1 for f in wf_r.get("folds", []) if f.get("precision", 0) >= 0.50)
        wf12.metric("Folds prec ≥ 50%", f"{n_prec50} / {wf_info.get('n_folds', 0)}")

        # ── Per-fold charts ──────────────────────────────────────────────────
        folds = wf_r.get("folds", [])
        if folds:
            st.divider()
            st.subheader("Kết quả từng fold")
            fd = pd.DataFrame(folds)

            # AUC + Precision line chart
            chart_l, chart_r = st.columns(2)
            with chart_l:
                st.markdown("**ROC-AUC theo fold:**")
                auc_df = fd.set_index("fold")[["roc_auc"]].copy()
                auc_df["threshold_0.62"] = 0.62
                st.line_chart(auc_df, height=220)
            with chart_r:
                st.markdown("**Precision theo fold:**")
                prec_df = fd.set_index("fold")[["precision"]].copy()
                prec_df["target_0.50"] = 0.50
                prec_df["target_0.52"] = 0.52
                st.line_chart(prec_df, height=220)

            # Signals per fold
            chart_l2, chart_r2 = st.columns(2)
            with chart_l2:
                st.markdown("**Số tín hiệu / fold:**")
                if "n_signals" in fd.columns:
                    st.bar_chart(fd.set_index("fold")["n_signals"], height=200)
            with chart_r2:
                st.markdown("**F1 Score theo fold:**")
                if "f1" in fd.columns:
                    st.bar_chart(fd.set_index("fold")["f1"], height=200)

            # Fold table
            st.divider()
            st.subheader("Bảng chi tiết 19 folds")
            display_cols = [c for c in [
                "fold", "test_start", "test_end", "threshold",
                "roc_auc", "precision", "recall", "f1", "accuracy",
                "n_signals", "signal_rate", "elapsed_s"
            ] if c in fd.columns]

            def _highlight_fold(row):
                prec_ok = row.get("precision", 0) >= 0.50
                auc_ok  = row.get("roc_auc", 0) >= 0.62
                if prec_ok and auc_ok:
                    return ["background-color: #1b3a1b"] * len(row)
                elif prec_ok:
                    return ["background-color: #1a2a0a"] * len(row)
                else:
                    return ["background-color: #2a1a1a"] * len(row)

            styled = fd[display_cols].style.apply(_highlight_fold, axis=1).format({
                "roc_auc":     "{:.4f}",
                "precision":   "{:.4f}",
                "recall":      "{:.4f}",
                "f1":          "{:.4f}",
                "accuracy":    "{:.4f}",
                "signal_rate": "{:.3f}",
                "threshold":   "{:.2f}",
                "elapsed_s":   "{:.1f}",
            }, na_rep="n/a")
            st.dataframe(styled, use_container_width=True)
            st.caption("Xanh đậm: precision ≥ 50% & AUC ≥ 0.62 | Xanh nhạt: precision ≥ 50% | Đỏ: precision < 50%")

        # ── Dynamic Concurrent Position Simulation ───────────────────────────
        _agg_csim = agg.get("concurrent_sim", {})
        if _agg_csim or any("concurrent_sim" in f for f in folds):
            st.divider()
            st.subheader("📈 Mô phỏng Lệnh Song Song — Dynamic Position Scaling")
            st.caption("Mỗi fold bắt đầu với $200. Khi balance tăng → mở thêm lệnh cùng lúc (2→3→5→8→15 slots)")

            # Aggregate concurrent sim metrics
            if _agg_csim:
                cs1, cs2, cs3, cs4 = st.columns(4)
                _cs_wr = _agg_csim.get("avg_win_rate", 0)
                _cs_pf = _agg_csim.get("avg_profit_factor", 0)
                _cs_ret = _agg_csim.get("avg_return_pct", 0)
                _cs_dd = _agg_csim.get("avg_max_drawdown_pct", 0)
                cs1.markdown(_card("Win Rate TB", f"{_cs_wr:.1%}",
                                   "trung bình các fold",
                                   GREEN if _cs_wr >= 0.70 else AMBER), unsafe_allow_html=True)
                cs2.markdown(_card("Profit Factor TB", f"{_cs_pf:.3f}",
                                   "lợi nhuận / thua lỗ",
                                   GREEN if _cs_pf >= 1.5 else AMBER), unsafe_allow_html=True)
                cs3.markdown(_card("Return TB/Fold", f"{_cs_ret:+.2f}%",
                                   f"từ $200 mỗi fold",
                                   GREEN if _cs_ret >= 0 else "#8b0000"), unsafe_allow_html=True)
                cs4.markdown(_card("Max DD TB", f"{_cs_dd:.2f}%",
                                   "drawdown tối đa trung bình",
                                   GREEN if abs(_cs_dd) < 15 else AMBER), unsafe_allow_html=True)

                cs5, cs6 = st.columns(2)
                _cs_avgpos = _agg_csim.get("avg_concurrent_positions", 0)
                _cs_maxpos = _agg_csim.get("max_concurrent_positions", 0)
                cs5.metric("Avg Concurrent Positions", f"{_cs_avgpos:.2f}")
                cs6.metric("Max Concurrent Positions", str(_cs_maxpos))

            # Per-fold concurrent sim chart
            _fold_csim_rows = [
                {
                    "fold": f.get("fold"),
                    "return_pct": f.get("concurrent_sim", {}).get("return_pct", 0),
                    "win_rate": f.get("concurrent_sim", {}).get("win_rate", 0),
                    "profit_factor": f.get("concurrent_sim", {}).get("profit_factor", 0),
                    "max_drawdown_pct": f.get("concurrent_sim", {}).get("max_drawdown_pct", 0),
                    "trades": f.get("concurrent_sim", {}).get("trades", 0),
                    "avg_concurrent": f.get("concurrent_sim", {}).get("avg_concurrent_positions", 0),
                    "ending_balance": f.get("concurrent_sim", {}).get("ending_balance", 200),
                }
                for f in folds if "concurrent_sim" in f
            ]
            if _fold_csim_rows:
                _csdf = pd.DataFrame(_fold_csim_rows).set_index("fold")

                _cc1, _cc2 = st.columns(2)
                with _cc1:
                    st.markdown("**Return % mỗi fold (từ $200):**")
                    st.bar_chart(_csdf[["return_pct"]], height=200)
                with _cc2:
                    st.markdown("**Win Rate mỗi fold:**")
                    _wr_chart = _csdf[["win_rate"]].copy()
                    _wr_chart["target_80pct"] = 0.80
                    st.line_chart(_wr_chart, height=200)

                _cc3, _cc4 = st.columns(2)
                with _cc3:
                    st.markdown("**Balance cuối mỗi fold (từ $200):**")
                    st.line_chart(_csdf[["ending_balance"]], height=200)
                with _cc4:
                    st.markdown("**Avg Concurrent Positions / fold:**")
                    st.bar_chart(_csdf[["avg_concurrent"]], height=200)

                st.divider()
                st.markdown("**Bảng chi tiết Concurrent Simulation:**")
                _csdf_display = _csdf.reset_index()
                st.dataframe(
                    _csdf_display.style.format({
                        "return_pct": "{:+.2f}%",
                        "win_rate": "{:.1%}",
                        "profit_factor": "{:.3f}",
                        "max_drawdown_pct": "{:.2f}%",
                        "avg_concurrent": "{:.2f}",
                        "ending_balance": "${:.2f}",
                    }, na_rep="n/a"),
                    use_container_width=True,
                )
                st.caption(
                    "Balance-tier scaling: < $500 → 2 lệnh | $500-$2k → 3 lệnh | "
                    "$2k-$10k → 5 lệnh | $10k-$50k → 8 lệnh | > $50k → 15 lệnh"
                )

        # ── Model info ───────────────────────────────────────────────────────
        st.divider()
        st.subheader("Thông tin Model Walk-Forward")
        mi_l, mi_r = st.columns(2)
        with mi_l:
            st.json({
                "model":       wf_info.get("model", "HistGBC"),
                "train_bars":  wf_info.get("train_bars", 20000),
                "test_bars":   wf_info.get("test_bars", 4000),
                "step_bars":   wf_info.get("step_bars", 4000),
                "n_folds":     wf_info.get("n_folds", 19),
                "n_features":  len(wf_info.get("features", [])),
            })
        with mi_r:
            feats = wf_info.get("features", [])
            if feats:
                st.markdown(f"**{len(feats)} Features:**")
                grps = {
                    "D1 (1)":  [f for f in feats if f.startswith("daily_")],
                    "H4 (9)":  [f for f in feats if f.startswith("h4_")],
                    "H1 (3)":  [f for f in feats if f in ("hourly_bias", "vsa_signal", "wyckoff_spring_signal")],
                    "M15 (15)": [f for f in feats if f not in [x for g in [
                        [f for f in feats if f.startswith("daily_")],
                        [f for f in feats if f.startswith("h4_")],
                        [f for f in feats if f in ("hourly_bias", "vsa_signal", "wyckoff_spring_signal")],
                    ] for x in g]],
                }
                for grp, names in grps.items():
                    if names:
                        st.markdown(f"**{grp}:** `{'`, `'.join(names)}`")

        # ── Walk-Forward signals CSV ─────────────────────────────────────────
        wf_signals = load_wf_signals()
        if not wf_signals.empty:
            st.divider()
            st.subheader(f"Tín hiệu Walk-Forward ({len(wf_signals):,} dòng)")
            disp_wf = [c for c in ["time", "fold", "side", "confidence", "predicted",
                                    "actual", "correct", "strategy_score"] if c in wf_signals.columns]
            st.dataframe(wf_signals[disp_wf].tail(200) if disp_wf else wf_signals.tail(200),
                         use_container_width=True)

    else:
        st.info(
            "Chưa có dữ liệu Walk-Forward. Chạy:\n"
            "```\npython scripts/walkforward_ict_wyckoff.py\n```"
        )


# =============================================================================
# TAB 7 — RISK & CAI DAT
# =============================================================================
with tab_risk:
    st.header("⚙️ Quản lý Rủi ro & Cài đặt")

    st.subheader("Quản lý Vị thế Động")
    st.markdown("""
| Số dư            | Lệnh tối đa | Rủi ro/lệnh | Ghi chú          |
|--------------------|----------|-----------|------------------|
| < $200             | **1**    | 0.65%     | Tài khoản nhỏ    |
| $200 – $500        | **2**    | 0.65%     | Tài khoản nhỏ    |
| $500 – $2,000      | **3**    | 0.65%     | Vừa-thấp        |
| $2,000 – $10,000   | **5**    | 0.65%     | Trung bình       |
| $10,000 – $50,000  | **8**    | 0.65%     | Lớn               |
| > $50,000          | **15**   | 0.65%     | Rất lớn          |

**Điều chỉnh theo Chế độ:**
- Sideways (regime=0): số lệnh / 2
- Strong Volatile (regime=2): số lệnh x 0.7
- Normal (regime=1): giữ nguyên
""")

    st.divider()
    st.subheader("Công cụ tính Lot")
    rc_l, rc_r = st.columns([1, 1])
    with rc_l:
        cb = st.number_input("Số dư (USD)", min_value=10.0, max_value=1_000_000.0, value=200.0, step=50.0)
        cs = st.number_input("Khoảng cách SL (USD)", min_value=0.1, max_value=200.0, value=3.6, step=0.5)
        cp = st.slider("Rủi ro mỗi lệnh (%)", min_value=0.1, max_value=3.0, value=0.65, step=0.05)
    with rc_r:
        ra    = cb * (cp / 100)
        cl    = max(0.01, round((ra / (100.0 * cs)) / 0.01) * 0.01)
        ml    = cl * 100 * cs
        lc    = GREEN if cl <= 0.1 else AMBER
        st.markdown(_card("Lot Size", f"{cl:.2f}", "XAUUSD", lc), unsafe_allow_html=True)
        st.metric("Số tiền rủi ro",    f"${ra:.2f}")
        st.metric("Lỗ tối đa (SL)",  f"${ml:.2f}")
        st.metric("Rủi ro/Số dư",   f"{(ml/cb)*100:.2f}%")

    st.divider()
    if not _selected_signals.empty and "account_balance" in _selected_signals.columns:
        st.subheader("Live Balance History")
        bh2 = _selected_signals[["time", "account_balance"]].dropna().sort_values("time").set_index("time")
        if not bh2.empty:
            st.area_chart(bh2["account_balance"], height=200)

    st.divider()
    sl_ev = [e for e in learn_events if e.get("event") in ("self_learn", "live_retrain")]
    st.subheader("Tóm tắt sự kiện Tự học")
    if sl_ev:
        sk1, sk2, sk3 = st.columns(3)
        sk1.metric("Tổng lần retrain",  len(sl_ev))
        imp2 = sum(1 for e in sl_ev if e.get("status") == "improved")
        sk2.metric("Model cải thiện",  imp2)
        brv  = max((e.get("roc_auc", 0) for e in sl_ev), default=0)
        sk3.metric("Best ROC-AUC",    f"{brv:.4f}")
        if len(sl_ev) >= 2:
            st.line_chart(pd.DataFrame({"ROC-AUC": [e.get("roc_auc", 0) for e in sl_ev]}), height=180)
        ev2 = pd.DataFrame(sl_ev[-20:])
        dc2 = [c for c in ["timestamp", "status", "dataset_rows", "roc_auc",
                             "precision", "recall", "f1"] if c in ev2.columns]
        st.dataframe(ev2[dc2] if dc2 else ev2, use_container_width=True)
    else:
        st.info("Chưa có sự kiện tự học. Bật live_learning_enabled=true và chạy bot live.")


# =============================================================================
# TAB 8 — DU LIEU MT5
# =============================================================================
with tab_data:
    st.header("🗄️ Quản lý Dữ liệu MT5")
    st.caption("Xem trạng thái dữ liệu OHLCV đã lưu, kiểm tra khoảng thiếu, và hướng dẫn cập nhật từ MT5")

    # ── Data coverage per timeframe ──────────────────────────────────────────
    st.subheader("📂 Trạng thái Dữ liệu theo Khung Thời Gian")

    _TF_FILES = {
        "M1":  "XAUUSDm_M1.csv",
        "M5":  "XAUUSDm_M5.csv",
        "M15": "XAUUSDm_M15.csv",
        "M30": "XAUUSDm_M30.csv",
        "H1":  "XAUUSDm_H1.csv",
        "H4":  "XAUUSDm_H4.csv",
        "D1":  "XAUUSDm_D1.csv",
    }
    _DATA_ROOT = ROOT / "src" / "xauusd_ai" / "real_data"

    _tf_status = []
    for _tf, _fname in _TF_FILES.items():
        _fpath = _DATA_ROOT / _fname
        if _fpath.exists():
            try:
                _df_tf = pd.read_csv(_fpath, usecols=["time"], on_bad_lines="skip")
                _df_tf["time"] = pd.to_datetime(_df_tf["time"], errors="coerce")
                _df_tf = _df_tf.dropna()
                _rows = len(_df_tf)
                _first = str(_df_tf["time"].min())[:10] if _rows > 0 else "n/a"
                _last  = str(_df_tf["time"].max())[:10] if _rows > 0 else "n/a"
                _size_kb = round(_fpath.stat().st_size / 1024, 1)
                _tf_status.append({
                    "Timeframe": _tf,
                    "File": _fname,
                    "Rows": f"{_rows:,}",
                    "Tu ngay": _first,
                    "Den ngay": _last,
                    "Size (KB)": _size_kb,
                    "Trang thai": "✅ OK",
                })
            except Exception as _e:
                _tf_status.append({
                    "Timeframe": _tf, "File": _fname,
                    "Rows": "loi", "Tu ngay": "n/a", "Den ngay": "n/a",
                    "Size (KB)": 0, "Trang thai": f"❌ {_e}",
                })
        else:
            _tf_status.append({
                "Timeframe": _tf, "File": _fname,
                "Rows": "0", "Tu ngay": "n/a", "Den ngay": "n/a",
                "Size (KB)": 0, "Trang thai": "⚠️ Chua co file",
            })

    st.dataframe(pd.DataFrame(_tf_status), use_container_width=True, hide_index=True)

    # ── Append instructions ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🔄 Câu Lệnh Cập Nhật Dữ Liệu Từ MT5")
    st.markdown("""
Để tải dữ liệu mới từ MT5 và nối vào file CSV hiện có (không trùng lặp), dùng script sau
trong Python hoặc terminal:
""")
    st.code("""
# Chạy trong Python (hoặc copy vào scripts/fetch_mt5_data.py):
import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path
import datetime

DATA_DIR = Path("src/xauusd_ai/real_data")
SYMBOL = "XAUUSDm"

TIMEFRAMES = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

mt5.initialize()

for tf_name, tf_code in TIMEFRAMES.items():
    csv_path = DATA_DIR / f"{SYMBOL}_{tf_name}.csv"
    # Find last timestamp already saved
    last_ts = None
    if csv_path.exists():
        existing = pd.read_csv(csv_path, usecols=["time"], on_bad_lines="skip")
        existing["time"] = pd.to_datetime(existing["time"], errors="coerce")
        last_ts = existing["time"].max()

    # Fetch from MT5 from last_ts onwards (or last 5000 bars)
    if last_ts is not None and not pd.isna(last_ts):
        from_dt = last_ts.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
        rates = mt5.copy_rates_from(SYMBOL, tf_code, from_dt, 5000)
    else:
        rates = mt5.copy_rates_from_pos(SYMBOL, tf_code, 0, 50000)

    if rates is None or len(rates) == 0:
        print(f"  {tf_name}: khong lay duoc du lieu")
        continue

    new_df = pd.DataFrame(rates)
    new_df["time"] = pd.to_datetime(new_df["time"], unit="s", utc=True)
    new_df = new_df[["time", "open", "high", "low", "close", "tick_volume"]]
    new_df.columns = ["time", "open", "high", "low", "close", "volume"]

    if csv_path.exists():
        old_df = pd.read_csv(csv_path, on_bad_lines="skip")
        old_df["time"] = pd.to_datetime(old_df["time"], errors="coerce")
        combined = pd.concat([old_df, new_df]).drop_duplicates("time").sort_values("time")
    else:
        combined = new_df

    combined.to_csv(csv_path, index=False)
    print(f"  {tf_name}: {len(combined):,} rows -> {csv_path.name}")

mt5.shutdown()
print("Xong!")
""", language="python")

    # ── Output files coverage ────────────────────────────────────────────────
    st.divider()
    st.subheader("📁 Trạng thái File Đầu ra (outputs/)")

    _OUT_FILES = [
        ("paper_trade_signals.csv",             "Tín hiệu live ACC1"),
        ("paper_trade_signals_acc2.csv",        "Tín hiệu live ACC2"),
        ("live_closed_trades.csv",              "Lệnh đã đóng ACC1"),
        ("live_closed_trades_acc2.csv",         "Lệnh đã đóng ACC2"),
        ("backtest_report_ict_wyckoff.json",    "Báo cáo backtest"),
        ("backtest_trades_ict_wyckoff.csv",     "Lệnh backtest"),
        ("walkforward_report_ict_wyckoff.json", "Báo cáo walk-forward"),
        ("live_learning_log.jsonl",             "Log học liên tục ACC1"),
        ("live_learning_log_acc2.jsonl",        "Log học liên tục ACC2"),
        ("model_ict_wyckoff.pkl",               "Model đang dùng"),
        ("model_meta_ict_wyckoff.json",         "Metadata model"),
    ]

    _out_status = []
    for _fname2, _desc in _OUT_FILES:
        _fp2 = ROOT / "outputs" / _fname2
        if _fp2.exists():
            _mtime = dt.datetime.fromtimestamp(_fp2.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            _sz = round(_fp2.stat().st_size / 1024, 1)
            _out_status.append({"File": _fname2, "Mô tả": _desc, "Size (KB)": _sz,
                                 "Cap nhat cuoi": _mtime, "Trang thai": "✅"})
        else:
            _out_status.append({"File": _fname2, "Mô tả": _desc, "Size (KB)": 0,
                                 "Cap nhat cuoi": "—", "Trang thai": "⚠️ Chua co"})

    st.dataframe(pd.DataFrame(_out_status), use_container_width=True, hide_index=True)

    # ── Bot restart commands ─────────────────────────────────────────────────
    st.divider()
    st.subheader("⚡ Lệnh PowerShell — Khởi động lại Bot & Dashboard")
    st.code("""
# Khởi động lại cả 2 bot (chạy trong PowerShell tại thư mục dự án):
Get-Process python -EA SilentlyContinue | Stop-Process -Force; Start-Sleep 2

$p1 = Start-Process python `
    -ArgumentList "scripts/live_runner.py","live","--config","configs/live_ict_wyckoff.yaml" `
    -RedirectStandardError "outputs\\live_err_acc1.txt" `
    -WorkingDirectory "$PWD" -WindowStyle Hidden -PassThru

$p2 = Start-Process python `
    -ArgumentList "scripts/live_runner.py","live","--config","configs/live_acc2.yaml" `
    -RedirectStandardError "outputs\\live_err_acc2.txt" `
    -WorkingDirectory "$PWD" -WindowStyle Hidden -PassThru

Write-Host "ACC1 PID=$($p1.Id)  ACC2 PID=$($p2.Id)"
""", language="powershell")

    st.code("""
# Khởi động lại Dashboard + Cloudflare tunnel:
$env:PYTHONPATH = "src"
Start-Process ".\.venv\Scripts\streamlit.exe" `
    -ArgumentList "run","src/xauusd_ai/dashboard/app.py","--server.port","8501",
                  "--server.headless","true","--browser.gatherUsageStats","false" `
    -WindowStyle Hidden

Start-Process ".\\cloudflared.exe" `
    -ArgumentList "tunnel","--url","http://localhost:8501","--protocol","http2" `
    -RedirectStandardError "outputs\\tunnel_err.txt" -WindowStyle Hidden

Start-Sleep 12
Get-Content outputs\\tunnel_err.txt | Select-String "trycloudflare.com"
""", language="powershell")
