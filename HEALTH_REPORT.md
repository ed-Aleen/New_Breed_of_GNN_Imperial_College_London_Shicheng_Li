# Stage-6 baseline health report

Generated 2026-08-24 from `health_check.py` (CPU-only, read-only).
Artifacts: `health_check.out` / `health_check_last.out` (first sweep),
`hc_new_idbest.out` / `hc_new_last.out` (+ `.json`) for the `dense` and `topup` groups.

Majority-class baseline on id_val: **MUTAG 0.5958**, RBGV 0.5160, MNIST 0.1201.
A run with `acc_idval` below the majority baseline is dead, not weak.

## Reading trap: GSAT anneals r

`r(epoch) = max(final_r, 0.9 - floor(epoch/10) * 0.1)`.
For MUTAG `final_r <= 0.4` the `id_best` checkpoint is selected at epoch 43-58, i.e.
*before* r has reached `final_r`. Read `last.ckpt` for the r-dial.
MNIST and RBGV `id_best` epochs are past the schedule, so `id_best` is valid there.

## Group `dense` — VERIFIED CLONE of the base configs

`*_dense.yaml` differ from the base YAMLs only in `clean_save: false`.
Epoch snapshots are now kept (epoch 0 present in all runs):
MUTAG 55-58 snaps, RBGV 108-115, MNIST 51-60.

Behavioural equivalence at `last.ckpt` (mean attention score, 3 seeds):

| dataset | base | dense |
|---|---|---|
| MUTAG  | 0.866 | 0.867 / 0.867 / 0.870 |
| RBGV   | 0.189 | 0.202 / 0.194 / 0.186 |
| MNIST  | 0.906 | 0.913 / 0.905 / 0.904 |

=> the dense snapshots may be used as the training history of the base runs (E2 t0, E3).

## Group `topup`

### `mutag_gsat_w1000` — 5/5 DEAD, systematic, exclude from E4c
All five seeds: `last.ckpt` scores collapse to <= 1e-4 everywhere (std = 0), acc 0.404-0.564,
every one below the 0.5958 majority baseline. `id_best` epochs are 0/3/4/17/34, i.e. validation
never improved after the first few epochs.
Training trace (`logs_baselines/mutag_gsat_w1000_s345.log`): train loss *rises* from epoch 0
(0.71 -> 1.02 -> 2.10 -> 1.74), train acc pinned at 0.5549.
This is an optimizer failure at lr=0.001, not a selection phenomenon. The flatness theorem's
precondition (training reaches the data-envelope floor) is violated, so the cell is outside
its scope. Note also that Omega = KL(p||r=0.7) is *maximised*, not minimised, at p=0 --
further evidence that Omega is not what put the model there.

### `mutag_smgnn_sec6` — 3/5 usable
Dead: seed 2 (acc 0.439), seed 4 (acc 0.478, `id_best` at epoch 23). Usable: seeds 1, 3, 5.

### `mutag_gsat_w300` — 5/5 train, but GT-AUC is seed-noise
acc 0.764-0.831 at `id_best`, yet GT-AUC spans 0.377-0.869 at fixed hyperparameters
(seeds 1 and 5 below chance). Width 300 is trainable but the explanation is not reproducible
across seeds; any E4c width claim needs >= 5 seeds and an error bar.

## RBGV replication

`rbgv_dense` (3 fresh seeds) reproduces the base result exactly: acc **1.000** on all seeds,
GT-AUC 0.466 / 0.512 / 0.479 = chance. Combined with the first sweep this is **6/6 seeds at
perfect accuracy with a chance-level explanation** -- the cleanest degenerate cell available.
