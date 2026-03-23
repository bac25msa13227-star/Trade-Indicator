    """
Diagnostic: Xem probability phân phối của model trên data M15 gần nhất.
Chạy: $env:PYTHONPATH="src"; python scripts/diag_model_confidence.py --config configs/live_acc2.yaml
"""
import argparse, sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xauusd_ai.config import load_settings
from xauusd_ai.orchestrator import _bootstrap
from xauusd_ai.features.dataset import prepare_training_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_acc2.yaml")
    parser.add_argument("--rows", type=int, default=500, help="Số bar cuối để score")
    args = parser.parse_args()

    settings = load_settings(args.config)
    data_service, trainer, strategy, _, _, _ = _bootstrap(settings)
    trainer.load_artifacts()

    print(f"\n{'='*60}")
    print(f"  Model: {settings.app.model_path}")
    print(f"  Decision threshold: {trainer.decision_threshold:.4f}")
    print(f"  Features: {len(trainer.feature_columns)}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading CSV data...")
    frames = data_service.fetch_multi_timeframe_data(source=settings.market.training_data_source, all_bars=True)
    print(f"  M5: {len(frames.get('M5', []))} bars")

    # Build dataset
    print("Building features...")
    dataset = prepare_training_dataset(settings, frames, strategy)
    print(f"  Dataset: {len(dataset)} rows\n")

    # Chỉ score N rows cuối
    recent = dataset.tail(args.rows).copy()
    print(f"Scoring {len(recent)} recent bars: {recent['time'].iloc[0]} → {recent['time'].iloc[-1]}\n")

    # Vectorized scoring (faster)
    scored = trainer.predict_dataset(recent)
    probs = scored["probability"].values
    thr = trainer.decision_threshold

    print(f"Probability distribution trên {len(probs)} bars gần nhất:")
    print(f"  mean  = {probs.mean():.4f}")
    print(f"  std   = {probs.std():.4f}")
    print(f"  min   = {probs.min():.4f}")
    print(f"  max   = {probs.max():.4f}")
    print(f"  p50   = {np.percentile(probs, 50):.4f}")
    print(f"  p75   = {np.percentile(probs, 75):.4f}")
    print(f"  p90   = {np.percentile(probs, 90):.4f}")
    print(f"  p95   = {np.percentile(probs, 95):.4f}")
    print(f"  p99   = {np.percentile(probs, 99):.4f}")
    print()
    print(f"  Threshold hiện tại = {thr:.4f}")
    signals = (probs >= thr).sum()
    print(f"  Signals với thr={thr:.2f}: {signals}/{len(probs)} ({signals/len(probs)*100:.1f}%)")

    print()
    print("  Candidate thresholds:")
    for candidate in [0.75, 0.70, 0.65, 0.62, 0.60, 0.58, 0.55]:
        n = (probs >= candidate).sum()
        pct = n / len(probs) * 100
        label = " ← current" if abs(candidate - thr) < 0.01 else ""
        print(f"    thr={candidate:.2f}: {n:4d} signals ({pct:5.1f}%){label}")

    print()
    if signals == 0:
        print("WARNING Model KHÔNG tạo signal nào với threshold hiện tại!")
        suggested = round(float(np.percentile(probs, 92)), 2)
        print(f"  Đề xuất threshold mới: {suggested:.2f} (p92 của distribution)")
        print(f"  → Sửa outputs/{os.path.basename(settings.app.model_path).replace('.pkl', '')}_meta.json")
        print(f"     hoặc retrain với --config {args.config}")
    elif signals < len(probs) * 0.02:
        print(f"WARNING Signal rate rất thấp (<2%). Cân nhắc hạ threshold.")
    else:
        print(f"OK Signal rate bình thường ({signals/len(probs)*100:.1f}%).")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
