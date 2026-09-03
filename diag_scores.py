"""
Quick diagnostic: print explanation score statistics for GSAT checkpoints.
Usage: python diag_scores.py
"""
import torch, glob, numpy as np

def stats(label, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # The state dict key for GSAT extractor attention is typically 'extractor.*'
    keys = [k for k in ckpt["model_state_dict"].keys() if "extractor" in k or "att" in k.lower()]
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"  ckpt: {ckpt_path.split('checkpoints/')[-1]}")
    print(f"  extractor-related keys: {keys[:6]}")
    # Print weight norms as a proxy for whether the extractor is active
    for k in keys[:6]:
        t = ckpt["model_state_dict"][k]
        print(f"    {k}: shape={list(t.shape)}, mean={t.mean():.4f}, std={t.std():.4f}, "
              f"min={t.min():.4f}, max={t.max():.4f}")

# MNIST GSAT seeds 1-5
for seed in range(1, 6):
    path = glob.glob(
        f"storage/checkpoints/round{seed}/"
        "MNIST_basis_no_shift/GSAT*/0.001lr_0.0wd/GSAT_0.1_False_10_0.7/id_best.ckpt"
    )
    if path:
        stats(f"MNIST GSAT seed{seed}", path[0])

print("\n" + "="*60)
print("MUTAG GSAT seeds 1-5")
for seed in range(1, 6):
    path = glob.glob(
        f"storage/checkpoints/round{seed}/"
        "MUTAG_basis_no_shift/GSAT*/0.001lr_0.0wd/GSAT_1_False_10_0.7/id_best.ckpt"
    )
    if path:
        stats(f"MUTAG GSAT seed{seed}", path[0])
