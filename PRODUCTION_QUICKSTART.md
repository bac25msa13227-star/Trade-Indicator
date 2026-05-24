# 🚀 Production Deployment — Quick Start

## ⚡ ONE-COMMAND DEPLOYMENT

```bash
# Copy-paste this command on production machine:
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator" && \
git fetch origin && \
git checkout main && \
git pull origin main && \
chmod +x pull_deploy_checklist.sh && \
./pull_deploy_checklist.sh
```

**What it does:**
1. ✅ Pulls latest code from GitHub (main branch)
2. ✅ Verifies all files (configs, models, scripts)
3. ✅ Checks configuration (ACC1=paper, ACC2=live, filter=$15)
4. ✅ Validates model files (34MB combo133)
5. ✅ Verifies Docker running
6. ✅ Stops old containers
7. ✅ Deploys ACC1 (paper) + ACC2 (demo)
8. ✅ Runs monitoring & checklist

**Time:** ~3 minutes (including Docker startup)

---

## 📋 Pre-Deployment Checklist

**Before running command above:**
- [ ] Production machine has internet access
- [ ] Docker Desktop installed
- [ ] Repository cloned at `~/Documents/Thạc sĩ MSE/Trade Indicator`
- [ ] Git credentials configured (can pull from GitHub)
- [ ] MT5 bridge running (ports 5600, 5601)

**If first-time setup:**
```bash
# 1. Clone repository
cd "$HOME/Documents/Thạc sĩ MSE"
git clone https://github.com/bac25msa13227-star/Trade-Indicator.git

# 2. Configure git
cd "Trade Indicator"
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 3. Create Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Start Docker
open -a Docker
sleep 30

# 5. Run deployment
./pull_deploy_checklist.sh
```

---

## 📊 Post-Deployment Monitoring

**Daily check (30 seconds):**
```bash
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"
./monitor_dual_mode.sh
```

**Expected output:**
```
ACC1 (Paper):  Skip 82%, 27 signals kept
ACC2 (Demo):   Skip 81%, P&L $145, Win 58%
Status: ✅ Both accounts healthy
```

**View live logs:**
```bash
docker logs -f live-acc1  # ACC1 paper
docker logs -f live       # ACC2 demo
```

---

## 🔧 Troubleshooting

### Problem: Git pull fails
```bash
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"
git status
git stash  # Save local changes
git pull origin main
```

### Problem: Docker not running
```bash
open -a Docker
sleep 30
./pull_deploy_checklist.sh
```

### Problem: Containers not starting
```bash
docker compose down  # Stop all
docker compose up -d postgres mlflow  # Start infrastructure
sleep 10
docker compose up -d live-acc1 live  # Start trading
```

### Problem: Skip rate too low (<60%)
```bash
vim configs/live_acc1.yaml  # min_expected_profit: 18.0
vim configs/live_acc2.yaml  # min_expected_profit: 18.0
docker compose restart live-acc1 live
```

---

## � IMPROVEMENT PHASE — Extended Training + M1 Scalp

After initial validation, improve model generalization:

### Phase 1: Extended Training (150k bars = 500+ days)

**Problem:** Current model trained on only 30k bars (~104 days). Too short → overfits to recent market regime.

**Solution:** Retrain with 150k bars (2003-2026 full history) = 2 years per fold → better generalization.

**Step 1: Build extended features (2003-2026)**
```bash
# Requires: MT5 terminal running with full history loaded
# Time: ~30-60 minutes first run (downloads 20+ years data)
cd F:\Trading_BOT_AUTO\Trade-Indicator
python scripts/build_historical_features_extended.py --from 2003-01-01 --to 2026-05-10

# Output: outputs/historical_features_2003_2026.csv (~1M rows, 4GB)
```

**Step 2: Retrain WF with longer history**
```bash
# Uses extended features, TRAIN_BARS=150,000 (~520 days per fold)
# Time: ~2-3 hours (150k bars takes longer to train)
python scripts/walkforward_extended_150k.py --config configs/xauusd_combo133_best.yaml --max-folds 5

# Output: outputs/walkforward_extended_150k_xauusd_combo133_best_results.json
# Expected: Accuracy should stabilize ~55-58% (longer training = more stable)
```

**Step 3: Generate new signals from extended model**
```bash
# Create signals_extended.csv using retrained ensemble
# (Requires engineering work to integrate extended model)
# Alternative: Use walkforward fold checkpoints to generate live signals
```

**Why this helps x5/month:**
- ✅ Model sees 2 years of history instead of 3.5 months → better regime detection
- ✅ Reduces overfitting to 2022-2026 market regime
- ✅ Should reduce bad-month frequency (2025-07, 2026-01 losing streaks)
- ⚠️  Does NOT fix structural WR issues in certain months (e.g., 2024-12 WR=0%, Dec 2025 WR=0%)

---

### Phase 2: M1 Scalp Signals (parallel to M5 swing)

**Problem:** M5 only gives 50 signals/month, 7.9 signals/day. At MaxPos=3, execution capacity limits returns.

**Solution:** Add M1 scalp model running in parallel:
- TP/SL: 2:1 ratio (faster exits, lower risk per trade)
- Lookahead: 10 minutes (vs 2.5 hours for M5)
- Min confidence: 0.68 (vs 0.74 for M5, lower threshold = more signals)
- Expected frequency: 200-400 M1 signals/month (much higher)

**Step 1: Build M1 features**
```bash
# M1 bars (1-minute candles) for same 2003-2026 period
# Time: ~45 minutes
python scripts/build_historical_features_extended_m1.py --from 2003-01-01 --to 2026-05-10

# Output: outputs/historical_features_2003_2026_M1.csv
```

**Step 2: Train M1 scalp WF**
```bash
# TRAIN_BARS=50,000 (~35 days M1, less than M5 due to 1440 bars/day)
# Time: ~1-2 hours
python scripts/walkforward_m1_scalp.py --config configs/xauusd_combo133_best.yaml

# Output: outputs/walkforward_m1_scalp_results.json
# Expected: Acc=52-55%, WR=45-50% (lower quality due to faster timeframe)
```

**Step 3: Integrate M1 signals into EA**
```
Modified EA reads:
  1. signals_for_mt5.csv (M5 swing, 50 signals/month)
  2. signals_m1_scalp.csv (M1 scalp, 200-400 signals/month)

EA allocates MaxPositions:
  - Swing trades: up to 3 positions (M5 signals)
  - Scalp trades: up to 2 positions (M1 signals, lower TP = less capital tied)
  - Total: up to 5 concurrent positions
```

**Expected impact on x5/month:**
- More signals = more chances to execute when slots are free
- Scalp WR=45% (lower) but 2:1 RR offsets it (1% avg trade × 400/month = +40% monthly)
- Combined M5+M1 at MaxPos=5: ~20-30% monthly geometric mean vs current 14.7%
- Still won't guarantee x5 EVERY month (structural bad months remain), but increases x5 months from 1/19 to ~3-4/19

---

## �📈 Week 1 Validation (May 6-13)

**Daily monitoring:**
```bash
./monitor_dual_mode.sh
```

**Success criteria:**
- ✅ Skip rate: 60-82% (both accounts)
- ✅ Win rate: 40%+ (ACC2 demo)
- ✅ P&L: Positive trend (ACC2 demo)
- ✅ Consistency: <5pp difference ACC1/ACC2 skip rates

**Day 7 decision matrix:**

| Condition | Action |
|-----------|--------|
| Skip 60-82% + P&L positive | ✅ Deploy to LIVE account |
| Skip 40-60% | ⚠️  Adjust threshold $18-20 |
| Skip <40% OR P&L negative | ❌ Disable filter, investigate |

---

## 📞 Support

**Full documentation:**
- `PRODUCTION_DEPLOYMENT.md` (430 lines, detailed guide)
- `pull_deploy_checklist.sh` (deployment script with 7-step verification)
- `monitor_dual_mode.sh` (monitoring with skip rates, P&L, status)

**Quick commands:**
```bash
# Help
cat PRODUCTION_DEPLOYMENT.md | less

# Status
docker ps --filter "name=live"

# Restart
docker compose restart live-acc1 live

# Stop
docker compose down live live-acc1

# Logs
docker logs live-acc1 --tail 50
docker logs live --tail 50
```

---

## 🎯 Expected Results (from WF validation)

**WF validation (10 folds, 2.4 years):**
- Skip rate: 82.3% average (67-95% range)
- Average return: +15,107% per fold
- Win rate: 78% (7/9 profitable folds)
- Sharpe: 4.369 | Sortino: 14.821 | Calmar: 132.899

**Production expectations:**
- Skip rate: 60-82% (with real prices, some variance expected)
- Win rate: 40%+ (expected 41% from WF)
- P&L: Positive trend after 50+ trades
- Consistency: ACC1 (paper) and ACC2 (demo) skip rates within 5pp

---

**Last Updated:** 2026-05-06  
**Version:** 1.0.0  
**Status:** ✅ Ready for production deployment
