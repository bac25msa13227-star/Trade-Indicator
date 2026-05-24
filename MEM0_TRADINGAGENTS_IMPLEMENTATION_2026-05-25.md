# mem0 + TradingAgents Implementation Update - 2026-05-25

User decisions applied:

- mem0: enabled in code path with Qdrant local support and safe JSONL fallback when key/service is unavailable.
- Branch scope: implement Week 1 now, not research-only.
- Decision log: per-account files, e.g. `outputs/decision_log_acc1.md` and `outputs/decision_log_acc2.md`.

Implementation status:

- Added append-only per-account decision log and deterministic reflection in `src/xauusd_ai/learning/decision_log.py`.
- Wired `PaperTradeLogger` entry/skip/exit events into the decision log.
- Replaced legacy `TradeDecision` dataclass with a Pydantic model while preserving positional constructor compatibility.
- Added 5-tier rating helper in `src/xauusd_ai/strategies/rating.py`.
- Added optional `risk.rating_risk_enabled` for rating-based risk scaling; default is off to avoid changing live risk unexpectedly.
- Added `LossMemoryStore` in `src/xauusd_ai/learning/loss_memory.py`; it attempts mem0 when `MEM0_ENABLED=true`, otherwise writes/searches local JSONL fallback.
- Added Qdrant service to `docker-compose.yml` and mem0/Qdrant env placeholders to `.env.example`.

Validation:

- Python smoke test passed for schema validation, decision log writes, and loss memory fallback.
- `docker compose config --quiet` passed.
- Qdrant started and `http://localhost:6333/readyz` returned ready.

Runtime notes:

- `MEM0_ENABLED=false` keeps the system on local JSONL only.
- To use mem0 for real, set `MEM0_ENABLED=true`, keep Qdrant running, and provide an LLM/embedding key such as `OPENAI_API_KEY`.
- `fastembed` was tested but failed in this Windows Python environment because `onnxruntime` could not load its native DLL. The repo default is therefore OpenAI embeddings for mem0, with JSONL fallback when credentials are absent.
