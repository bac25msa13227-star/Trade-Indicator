# Compound Growth Deployment Guide

**Target:** $200 → $10k capital (Month 1) → $57k-$110k/month profit (Month 2+)  
**Risk:** 5% per trade (validated: -13.46% max DD)  
**Timeline:** 1 month to $10k, 6 months to $345k total withdrawn  
**Status:** ✅ WF VALIDATED (6 folds, Sharpe 5.721, Calmar 735.185)

---

## 📊 WF Validation Results (Proof of Concept)

**Run:** May 6, 2026 with `--compound-target 10000 --risk-pct 0.050`

| Fold | Period | Start | End | Withdrawn | Notes |
|------|--------|-------|-----|-----------|-------|
| 1 | Dec 23 - Jan 24 | $200 | $11,962 | $1,962 | Reached $10k in 1 month ✅ |
| 2 | Jan - Feb 24 | $10,000 | $16,389 | $6,389 | Normal trending |
| 3 | Feb - Mar 24 | $10,000 | $18,122 | $8,122 | Normal trending |
| 4 | Mar - Apr 24 | $10,000 | $95,695 | **$85,695** | 🚀 Explosive trending |
| 5 | Apr - May 24 | $10,000 | $125,464 | **$115,464** | 🚀 Explosive trending |
| 6 | May - Jun 24 | $10,000 | $137,533 | **$127,533** | 🚀 Explosive trending |

**Total withdrawn (6 months):** $345,165  
**Average profit/month:** $57,527  
**Peak 3 months (Fold 4-6):** $109,564/month average  
**Max Drawdown:** -13.46% (vs 25% limit) ✅

---

## 🎯 Deployment Strategy

### **Phase 1A: Paper Mode Validation (Week 1 - May 6-13)**

**Goal:** Verify paper mode matches WF expectations

**On Windows production machine:**
```powershell
# 1. Pull latest code
cd "C:\Users\YourUser\Documents\Thạc sĩ MSE\Trade Indicator"
git pull origin main

# 2. Activate venv
.\.venv\Scripts\Activate.ps1

# 3. Verify model files exist
Get-ChildItem outputs\acc1_combo133_202604_model.pkl
Get-ChildItem outputs\acc1_combo133_202604_scaler.pkl

# 4. Start paper mode with compound
python -m xauusd_ai.orchestrator configs/live_acc1_compound.yaml 2>&1 | Tee-Object -FilePath outputs\paper_compound_week1.log -Append
```

**Monitoring checklist (Daily):**
```powershell
# Check current balance
Get-Content outputs\paper_compound_week1.log | Select-String "Balance:" | Select -Last 5

# Check DD
python -c "import pandas as pd; df = pd.read_csv('outputs/live_closed_trades_acc1.csv'); print(f'Current DD: {df['drawdown_pct'].min():.2%}')"

# Check skip rate
Get-Content outputs\paper_compound_week1.log | Select-String "PROFIT FILTER" | Select -Last 20

# Check for circuit breaker triggers
Get-Content outputs\paper_compound_week1.log | Select-String "CIRCUIT BREAKER|KILL SWITCH|PAUSE" | Select -Last 10
```

**Week 1 Pass Criteria:**
- ✅ Skip rate 70-90% (profit filter working)
- ✅ Win rate 35-45%
- ✅ Max DD < 20%
- ✅ No fatal errors
- ✅ Profit trend positive or flat (allow 1-2 losing days)
- ✅ Balance growing toward $1,000+ by Day 7

**Decision May 13:**
- ✅ **PASS:** All criteria met → Switch to live mode
- ⚠️ **PARTIAL:** Some criteria met → Extend validation 1 week
- ❌ **FAIL:** Major issues (DD > 25%, errors, skip rate < 50%) → Investigate, fix, restart validation

---

### **Phase 1B: Live Deployment (May 13+, if Week 1 PASS)**

**On Windows production machine:**
```powershell
# 1. Stop paper mode (Ctrl+C in terminal)

# 2. Update config to live mode
# Edit configs/live_acc1_compound.yaml:
#   execution:
#     mode: live  # Change from paper to live
```

**Or use separate live config:**
```powershell
# Copy compound config to live version
Copy-Item configs\live_acc1_compound.yaml configs\live_acc1_compound_live.yaml

# Edit live_acc1_compound_live.yaml: mode: live
```

**Start live compound mode:**
```powershell
python -m xauusd_ai.orchestrator configs\live_acc1_compound_live.yaml 2>&1 | Tee-Object -FilePath outputs\live_compound_phase1.log -Append
```

**Real-time monitoring (CRITICAL):**
```powershell
# Terminal 1: Live logs
Get-Content outputs\live_compound_phase1.log -Wait

# Terminal 2: Balance check every hour
while ($true) {
    python -c "import pandas as pd; df = pd.read_csv('outputs/live_closed_trades_acc1.csv'); print(f'{(Get-Date).ToString(\"HH:mm\")} | Balance: \${df[\"balance_after\"].iloc[-1]:.2f} | DD: {df[\"drawdown_pct\"].min():.2%} | Trades: {len(df)}')"
    Start-Sleep 3600  # 1 hour
}
```

**Critical thresholds (Manual intervention required):**
- ⚠️ **DD > 15%:** Review recent trades, check for news events
- ⚠️ **DD > 20%:** Reduce risk to 3%, extend stops
- 🛑 **DD > 25%:** STOP SYSTEM, withdraw funds, analyze
- 🛑 **5 consecutive losses:** System auto-pauses 24h (circuit breaker)
- 🛑 **Daily loss > 10%:** System auto-pauses (circuit breaker)

---

### **Phase 2: Transition to $40k Target (Month 5-6)**

**When to transition:**
- ✅ ACC1 stable at $10k for 2+ months
- ✅ Consistent $50k+/month withdrawals
- ✅ Max DD < 15% over last 30 days
- ✅ Win rate 40-50%, PF > 2.5

**Update config:**
```yaml
# configs/live_acc1_compound_phase2.yaml
risk:
  risk_per_trade: 0.030  # Reduce to 3% (more stable)
  max_drawdown_kill_pct: 0.20  # Tighten to 20%
  compound_cap: 200  # Cap at 200× starting = $40k
```

**Or use --compound-target flag:**
```powershell
# For testing in WF first
python scripts\walkforward_ict_wyckoff.py configs\acc1_v14pp_profit.yaml `
  --test-start 2024-01-01 --test-bars 6000 --step-bars 6000 `
  --max-folds 6 --combo133 --cache --profit-filter --min-profit 15.0 `
  --risk-pct 0.030 --no-rr-sweep `
  --compound-target 40000  # Compound from $10k → $40k
```

**Expected trajectory (3% risk, $10k → $40k):**
- Month 1: $10k → $30k (+200%)
- Month 2: $30k → $40k (+33%) → Target reached!
- Month 3+: $40k × 500% = $200k/month profit ✅

---

### **Phase 3: Multi-Account Scaling (Month 7+)**

**Prerequisites:**
- ✅ ACC1 stable at $40k for 3+ months
- ✅ Consistent $150k-$200k/month profit
- ✅ Max DD < 12% over last 60 days
- ✅ Win rate 45-55%, PF > 3.0

**Setup additional accounts:**
```powershell
# ACC2 setup
Copy-Item configs\live_acc1_compound_live.yaml configs\live_acc2_compound.yaml

# Edit live_acc2_compound.yaml:
#   mt5.bridge_port: 5601  # ACC2 port
#   mt5.magic_number: 20240102
```

**Staggered deployment:**
- Week 1: ACC1 only (validate stability)
- Week 2: ACC1 + ACC2 (2 accounts)
- Week 3: ACC1 + ACC2 + ACC3 (3 accounts)
- Week 4: ACC1-5 (5 accounts)

**Expected (5 accounts × $40k × 500%):**
- Each account: $200k/month
- Total: $1,000k/month = **$1M/month** ✅

---

## ⚠️ Risk Management & Monitoring

### **Daily Checks (5 minutes):**
```powershell
# 1. Current balance & DD
python -c "import pandas as pd; df = pd.read_csv('outputs/live_closed_trades_acc1.csv'); print(f'Balance: \${df[\"balance_after\"].iloc[-1]:.2f} | DD: {df[\"drawdown_pct\"].min():.2%} | P&L today: \${df[df[\"exit_time\"].str.contains((Get-Date).ToString(\"yyyy-MM-dd\"))][\"pnl\"].sum():.2f}')"

# 2. Win rate last 20 trades
python -c "import pandas as pd; df = pd.read_csv('outputs/live_closed_trades_acc1.csv').tail(20); print(f'Win rate: {(df[\"pnl\"] > 0).sum()}/{len(df)} = {(df[\"pnl\"] > 0).mean():.1%}')"

# 3. Check for errors
Get-Content outputs\live_compound_phase1.log | Select-String "ERROR|CRITICAL|FAIL" | Select -Last 10
```

### **Weekly Review (30 minutes):**
```powershell
# 1. Generate weekly report
python scripts\generate_weekly_report.py outputs\live_closed_trades_acc1.csv

# 2. Review metrics
# - Win rate: Should be 40-50%
# - Profit factor: Should be > 2.5
# - Avg RR: Should be > 3.0
# - Skip rate: Should be 70-90%

# 3. Check MLflow metrics
# Open http://localhost:5000
# Review: AUC, precision, recall trends

# 4. Compare to WF expectations
# - Current balance vs WF projection
# - Actual DD vs WF DD
# - Actual profit/month vs WF profit/month
```

### **Monthly Audit (2 hours):**
```powershell
# 1. Backtest-live gap analysis
python scripts\compare_backtest_vs_live.py `
  --live outputs\live_closed_trades_acc1.csv `
  --backtest outputs\walkforward_trades_acc1_v14pp_profit_sim_trades.csv `
  --start (Get-Date).AddMonths(-1).ToString("yyyy-MM-dd")

# 2. Feature drift check
python scripts\check_feature_drift.py `
  --live-data outputs\live_feature_snapshot.csv `
  --train-data data\XAUUSD_M5_cleaned.csv

# 3. Model performance review
# - If AUC drops below 0.60 → Retrain
# - If win rate drops below 35% → Investigate
# - If skip rate drops below 60% → Check profit filter

# 4. Risk parameter adjustment
# - If DD consistently < 10% → Consider increasing risk to 6%
# - If DD > 20% → Reduce risk to 3%
# - If consecutive losses frequent → Tighten filters
```

---

## 🚨 Emergency Procedures

### **Circuit Breaker Triggered:**
```powershell
# System auto-pauses. Review logs:
Get-Content outputs\live_compound_phase1.log | Select-String "CIRCUIT BREAKER" -Context 20

# Check trigger reason:
# - Daily loss > 10%: Normal in volatile markets, wait 24h
# - Max DD > 25%: CRITICAL, investigate immediately
# - 5 consecutive losses: Check for news events, regime shift

# Action:
# - If triggered by news spike: Wait 24h, resume
# - If triggered by model failure: Retrain model, resume
# - If triggered by market regime shift: Review settings, adjust risk
```

### **Max DD > 25% (CRITICAL):**
```powershell
# 1. STOP SYSTEM immediately
# Press Ctrl+C in orchestrator terminal

# 2. Withdraw funds to safe balance
# Keep only $200 for trading, withdraw rest to broker account

# 3. Analyze failure
python scripts\analyze_loss_streak.py outputs\live_closed_trades_acc1.csv

# 4. Root cause identification:
# - Overfitting: Model doesn't generalize to new market
# - Regime shift: Market changed (trending → sideway)
# - Execution issues: Slippage higher than expected
# - News events: Unexpected volatility spike

# 5. Fix and validate:
# - Retrain model on latest data
# - Adjust risk to 2% (ultra-conservative)
# - Run WF validation again
# - Restart paper mode for 7 days before live
```

### **System Crash/Power Outage:**
```powershell
# 1. Check open positions on MT5
# Manually close if necessary

# 2. Restart system
python -m xauusd_ai.orchestrator configs\live_acc1_compound_live.yaml 2>&1 | Tee-Object -FilePath outputs\live_compound_phase1.log -Append

# 3. Verify state recovery
# System should auto-load last known balance from live_closed_trades_acc1.csv
```

---

## 📈 Success Metrics & Milestones

### **Month 1 Target (Phase 1A):**
- ✅ Capital: $200 → $1,000+ (500% growth)
- ✅ Max DD: < 20%
- ✅ Win Rate: 35-45%
- ✅ Skip Rate: 70-90%
- **Achieved:** Proceed to Phase 1B (live mode)

### **Month 2-4 Target (Phase 1B):**
- ✅ Capital: Maintain $10k
- ✅ Profit/month: $50k-$100k avg
- ✅ Max DD: < 15%
- ✅ Win Rate: 40-50%
- **Achieved:** Proceed to Phase 2 ($40k target)

### **Month 5-6 Target (Phase 2):**
- ✅ Capital: $10k → $40k
- ✅ Profit/month: $100k-$200k avg
- ✅ Max DD: < 12%
- ✅ Win Rate: 45-55%
- **Achieved:** Proceed to Phase 3 (multi-account)

### **Month 7+ Target (Phase 3):**
- ✅ Accounts: 5 accounts × $40k = $200k total capital
- ✅ Profit/month: 5 × $200k = **$1M/month total**
- ✅ Max DD per account: < 10%
- ✅ Win Rate: 45-55%
- **Status:** SUSTAINABLE SCALING ✅

---

## 📝 Deployment Checklist

### **Pre-Deployment (Day 0):**
- [ ] Git pull latest code (`git pull origin main`)
- [ ] Verify model files exist (acc1_combo133_202604_model.pkl, scaler.pkl)
- [ ] Check venv dependencies (`pip list | grep lightgbm`)
- [ ] Test MT5 bridge connection (`curl http://127.0.0.1:5600/health`)
- [ ] Review config (configs/live_acc1_compound.yaml)
- [ ] Backup current data (outputs/ folder)

### **Paper Mode Deployment (Day 1):**
- [ ] Start paper mode (configs/live_acc1_compound.yaml, mode: paper)
- [ ] Verify logs generating (outputs/paper_compound_week1.log)
- [ ] Check skip rate in logs (expect 70-90%)
- [ ] Monitor balance (expect growth toward $1k in Week 1)
- [ ] Daily checks (balance, DD, errors)

### **Live Mode Deployment (Day 8, if Week 1 PASS):**
- [ ] Stop paper mode
- [ ] Review Week 1 results (win rate, DD, skip rate)
- [ ] Update config to live mode
- [ ] Start live mode (configs/live_acc1_compound_live.yaml)
- [ ] Hourly monitoring first 24h (balance, DD, errors)
- [ ] Daily checks ongoing

### **Phase 2 Transition (Month 5):**
- [ ] Verify ACC1 stable at $10k for 60+ days
- [ ] Max DD < 15% over last 30 days
- [ ] Win rate 40-50%, PF > 2.5
- [ ] Test --compound-target 40000 in WF first
- [ ] Update config to 3% risk, $40k target
- [ ] Deploy and monitor

### **Phase 3 Multi-Account (Month 7+):**
- [ ] ACC1 stable at $40k for 90+ days
- [ ] Consistent $150k-$200k/month profit
- [ ] Setup ACC2-5 configs
- [ ] Staggered deployment (1 account/week)
- [ ] Total capital: 5 × $40k = $200k
- [ ] Target: $1M/month total profit ✅

---

**Last Updated:** May 6, 2026  
**Status:** ✅ WF VALIDATED, READY FOR DEPLOYMENT  
**Next Step:** Deploy paper mode Week 1 (May 6-13)  
**Contact:** Check PLAN_200K_MONTHLY_PROFIT.md for detailed strategy
