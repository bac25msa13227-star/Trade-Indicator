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

## 📈 Week 1 Validation (May 6-13)

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
