# ECC Installation Manifest

**Installed**: 5 May 2026
**Version**: ECC 2.0.0-rc.1 (selective install)
**Method**: Manual copy from everything-claude-code

---

## Installed Components

### Agents (5)
Located in `.github/agents/`

1. **planner.md** — Implementation planning for complex features
2. **tdd-guide.md** — Test-driven development workflow
3. **code-reviewer.md** — Code quality and maintainability review
4. **security-reviewer.md** — Vulnerability detection and security best practices
5. **build-error-resolver.md** — Fix build/type errors

### Skills (5)
Located in `.github/agents/skills/`

1. **tdd-workflow/** — TDD methodology with 80%+ coverage requirements
2. **verification-loop/** — Pre-commit verification (lint, type, test, security)
3. **eval-harness/** — Formal evaluation framework for AI sessions
4. **security-review/** — Security checklist and vulnerability patterns
5. **strategic-compact/** — Manual context compaction at logical intervals

### Rules (14)
Located in `.github/rules/`

**Python Rules** (5):
- `python/python-coding-style.md` — PEP 8, naming conventions
- `python/python-patterns.md` — Design patterns, best practices
- `python/python-testing.md` — Pytest, fixtures, mocking
- `python/python-security.md` — Input validation, secrets management
- `python/python-hooks.md` — Hook development patterns

**Common Rules** (9):
- `common/common-coding-style.md` — Cross-language style
- `common/common-patterns.md` — Architecture patterns
- `common/common-testing.md` — Testing strategies
- `common/common-security.md` — Security fundamentals
- `common/common-performance.md` — Performance optimization
- `common/common-git-workflow.md` — Git conventions
- `common/common-hooks.md` — Hook system guidelines
- `common/common-agents.md` — Agent orchestration
- `common/common-development-workflow.md` — Dev cycle

---

## Usage

### Invoke Agents
Use `@agent-name` syntax in Claude Code:

```
@planner Design the slippage model architecture
@tdd-guide Implement regime detection with tests first
@code-reviewer Review the ensemble model implementation
@security-reviewer Check credential handling in configs
@build-error-resolver Fix pytest import errors
```

### Use Skills
Skills activate automatically based on context, or manually:

```
Use tdd-workflow skill to implement paper trading framework
Use security-review skill before committing MT5 integration
Use verification-loop skill to validate build before push
```

### Apply Rules
Rules apply automatically when working in relevant contexts:
- Python files → Python rules active
- Testing → Common testing rules active
- Security-sensitive code → Security rules active

---

## Integration with XAUUSD AI Project

### Custom Agent Integration
ECC agents work alongside custom `xauusd-ai-dev.agent.md`:

**Use XAUUSD AI Dev for**:
- AI/ML model development (ensemble, RL, regime detection)
- Feature engineering and model training
- Walk-forward optimization
- Backtesting and live trading logic

**Use ECC Agents for**:
- Code quality review (`@code-reviewer`)
- Test development (`@tdd-guide`)
- Security audits (`@security-reviewer`)
- Implementation planning (`@planner`)
- Build troubleshooting (`@build-error-resolver`)

### Workflow Example
```bash
# 1. Plan feature with ECC planner
@planner Design slippage model with spread + volatility components

# 2. Implement with custom XAUUSD agent
@XAUUSD AI Dev Implement slippage model in src/xauusd_ai/backtesting/slippage.py

# 3. Write tests with TDD guide
@tdd-guide Create pytest tests for slippage model

# 4. Review code quality
@code-reviewer Review slippage implementation

# 5. Security check
@security-reviewer Check for any security issues

# 6. Verify before commit
./scripts/verify.sh
```

---

## Next Steps

1. ✅ **ECC Install Complete** (5 agents, 5 skills, 14 rules)
2. ⏳ **Setup Test Infrastructure** (pytest, coverage, verification script)
3. ⏳ **Test Coverage to 70%** (AI-assisted test generation)
4. ⏳ **Implement Advanced ML** (slippage, regime, metrics, paper trading)

See `.github/agents/knowledge/ecc-action-plan.md` for full roadmap.

---

## References

- **Action Plan**: `.github/agents/knowledge/ecc-action-plan.md`
- **ECC Integration Strategy**: `.github/agents/knowledge/ecc-integration-plan.md`
- **Project Architecture**: `.github/agents/knowledge/architecture.md`
- **Original ECC Repo**: `$HOME/Documents/Thạc sĩ MSE/everything-claude-code/everything-claude-code`
