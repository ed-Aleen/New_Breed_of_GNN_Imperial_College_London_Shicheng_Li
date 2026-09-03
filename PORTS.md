# PORTS.md — every deviation from upstream `steveazzolin/gnn_deg_expl` @ b7b14be

This repo is a clean clone of the official paper repo, at the SAME commit as the
the research working copy (upstream HEAD = local HEAD = `b7b14be`).
Only the minimal bug-fix tier of the research repo's uncommitted changes was ported.
Nothing here is committed to git — `git diff` against HEAD shows exactly this list.

Environment assumed: a virtualenv with (torch 2.5.1+cu121, **numpy 2.2.6, PyG 2.6.0**).

## Ported (tier: required — upstream crashes or is invalid in this environment)

| File | Change | Reason |
|---|---|---|
| `GOOD/utils/evaluation.py` | `np.float` removed from a type annotation | numpy ≥ 1.24 removed `np.float` → **import-time crash** |
| `GOOD/data/good_datasets/mnist.py` (+ `colored_patch_mnist.py`) | `np.bool` → `np.bool_` | same numpy removal (raw-processing path) |
| `GOOD/data/good_datasets/{mnist,colored_patch_mnist,colored_patch_mnist2}.py` | **train/id_val split-leakage fix**: upstream hardcodes `n_train_data, n_val_data = 20000, 5000` over a pool of exactly 20000 → `id_val` was a 100% subset of `train` (verified 5000/5000 overlap). Now `n_train_data = min(20000, len(pool) − n_val_data)` | any id_val-based statement was invalid |
| `GOOD/networks/models/BaseGNN.py` | `self._fixed_explain` → `getattr(self, '_fixed_explain', False)` (3 message fns) | under PyG ≠ 2.4.0 `set_masks` never sets the attr → **AttributeError on any masked forward** |
| `GOOD/kernel/pipelines/basic_pipeline.py` | SummaryWriter path `/home/azzolin/...` → `<this repo>/outputs/logs/` | upstream path does not exist → crash on `--task train` |
| `GOOD/kernel/pipelines/basic_pipeline.py` | histogram-logging guards: shape-mismatch `continue`, `torch.isfinite` checks; `node_is_spurious` `hasattr` fallback (MUTAG/SST2P have none) | training-loop logging crashed on datasets without per-node GT / with NaN scores |
| `GOOD/kernel/pipelines/basic_pipeline.py` | removed stray `exit()` in the "too few intervened samples" branch of `compute_metric` | one sparse graph killed the whole eval run |
| `GOOD/kernel/pipelines/basic_pipeline.py` | SMGNN instance-wise min-max in `generate_binary_explanations` when scores collapse (`max ≤ 0.5`) | paper Appendix D.5's own rule; upstream has it commented out entirely |
| `GOOD/kernel/main.py` | `.get("train")` guards on final AUROC/F1 printing | KeyError when the split dict lacks "train" |

## Added (not upstream files)

- `GOOD/networks/models/GINvirtualnode.py` — **one-line shim** (`from .GINs import
  FeatExtractor as vFeatExtractor`). Upstream `SMGNN.py`/`GSATGNNs.py`/`DIRGNN.py` import
  this module but upstream never committed the file → the repo does not even import
  as shipped. Same shim the research repo uses.
- `goodtg` — wrapper script. The venv has the OLD repo installed in editable mode, so the
  bare `goodtg` console command always runs the OLD repo. Use `./goodtg ...` here (or the
  copied .sh scripts, which prepend this repo to PATH).
- `storage/datasets` — **symlink** to the research repo's datasets (8.7G, shared read-only;
  processed .pt files incl. SST2P are real files there). `storage/checkpoints/` is FRESH and
  empty: clean runs never mix with the research repo's round1–8 checkpoints.
- Section-6 configs copied from the research repo (pure YAML, no research keys):
  `SMGNN_sec6.yaml` (MNIST/MUTAG/SST2P/BAColorGVIsol), `SMGNN_sec6_bn.yaml`, `GSAT_bn.yaml`,
  `DIR_K1pct.yaml` (MUTAG/BAColorGVIsol), `DIR_K10pct.yaml` (SST2P).
- Helper scripts copied + path-adapted: `get_{mnist,mutag,sst2}_data.sh`, `smoke_test.sh`,
  `train.sh`, `run_table4_eval.sh`, `diag_scores.py`.

## Experiment additions (Stage-6, deliberately added — not upstream, not research-scaffold)

- `GOOD/networks/models/CALGNNs.py` + `GOOD/ood_algorithms/algorithms/CAL.py` — CAL
  (Sui et al., KDD 2022) adapted to this codebase's node-attention skeleton: the
  Omega ≡ 0 family for experiment E1b (deterministic sigmoid attention, NO score
  regulariser; losses = CE(causal) + λ·KL(trivial‖uniform) + λ·CE(intervention)).
  Faithful-adaptation notes in the model file header. Registered in models
  `__init__.py` (`, CALGNNs` added to the explicit import line).
- Experiment configs (all verified to land in DISTINCT checkpoint dirs):
  MUTAG `GSAT_r{2,3,4,5,6,8,9}.yaml` + MNIST `GSAT_r{3,5,9}.yaml` (E1a r-dial;
  `final_r = ood.extra_param[2]`, shipped baselines are the r=0.7 points),
  MUTAG `SMGNN_L1only.yaml` / `SMGNN_EntrOnly.yaml` (E1b pure-Ω arms at Section-6
  magnitudes), MUTAG `CAL.yaml`, MUTAG `GSAT_w{16,300,1000}.yaml` (E4c width sweep —
  dim_hidden not in the ckpt dirname, so launch WITH `--save_tag wNN`).
- `run_baselines.sh` — grouped launcher (smoke gate → mutag_dial / mutag_family /
  mutag_width / rbgv / mnist_dial), logs to `logs_baselines/`.

## Deliberately NOT ported (stays in the research repo only)

- All research scaffolds and their wiring (default-off, but still non-upstream code):
  SWG (`swg_alpha`/`extractor_tap`/`swg_layers`), LSC, IAR, PairNorm, node-local residual,
  `acr_readout_gain`, `init_scale`, `input_noise_std`, `entr_warmup`, `smgnn_stochastic`,
  `GINvirtualnode.py`, the +164-line `GINs.py` instrumentation, `Classifiers.py` iar widening,
  `BaseOOD/GSAT/SMGNN` ood-algorithm research hooks, `config_reader.py` checkpoint-dir tags,
  `args.py`/`base.yaml` research keys, `analysis.py` plot-annotation additions.
- Research datasets/configs beyond the paper: BenzeneAnchored, BenzeneWatermark, BA2Motifs,
  FluorideCarbonyl, MutagenicityGXAI.
- **`GOOD/ood_algorithms/algorithms/DIR.py` alpha schedule** — research repo normalises
  `alpha = λ·(epoch/max_epoch)^1.6`; upstream (= original DIR code) uses `λ·epoch^1.6`.
  This is a SEMANTIC training difference, left at upstream behaviour here. ⚠️ Decide
  explicitly before training DIR in this repo; the research repo's round1–8 DIR
  checkpoints were trained with the normalised schedule.

## Known upstream leftovers (harmless, documented)

- `GOOD/networks/models/DIRGNN.py:462` hardcodes `/home/azzolin/...` in
  `debug_subgraph_plot` — dead code (never called).
- `sparse_sort`/`sparse_topk` in `DIRGNN.py` use `argsort(stable=False)` with a
  batch-index offset in float32 — tie handling is device-dependent (CPU ≠ CUDA) and
  batch-position-quantised during training. NOT fixed on purpose: this is a measured
  artifact of the paper's pipeline (experiment E4d studies it).

## Experiment additions — OCC mitigation (E4, added 2026-09-02)

Retro-documentation first: an earlier session added the **opt-in
`fix_mask_aggregation` repair** (config YAML key, default OFF) to
`GSATGNNs.py` / `SMGNN.py` / `DIRGNN.py` `set_masks`/`clear_masks`
(`.bak-maskfix` files are the pre-fix state) plus
`BAColorGVIsol/.../GSAT_maskfix.yaml`. It was not logged here at the time —
logged now. With the flag on, ACRConv2.message multiplies messages by the
node mask, so a hard {0,1} mask equals true node deletion on sum-pool cells.

OCC (Occurrence-Calibrated Classifier; REAL-X/EVAL-X, Jethani et al. AISTATS
2021, transplanted at colour-class granularity — design doc in the research
repo: `BACKBONE_PREFERENCE/OCC_MITIGATION.md`):

- `GOOD/utils/occ.py` — NEW: hash-based broadcast-enriched colour refinement,
  mu0 Bern(r0) class-mask sampler (non-empty conditioning), induced-subgraph
  rebuilder.
- `masked_forward.py` — NEW (port of the research-repo helper): grad-capable
  masked classifier pass returning raw logits.
- `occ_stage_a.py` — NEW: Stage A calibration (classifier-side params only,
  CE on mu0 displays, holdout calib NLL, saves `<ckpt_dir>/occ_g0.ckpt`).
- `occ_ceiling.py` — NEW: E-OCC0 pre-registered evidence ceiling (zero
  training; discrete cells only, refuses continuous cells).
- `occ_channel_check.py` — NEW: certifies masked-forward(fix on, hard mask)
  == true subgraph re-encoding per cell; quantifies the fix-off leak.
- `GOOD/kernel/pipelines/basic_pipeline.py` — `train()` loads+freezes g0 when
  `occ_g0_ckpt` is set (new `_occ_load_freeze`; frozen modules re-pinned to
  eval() each `train_batch`). Opt-in; default path unchanged.
- `GOOD/ood_algorithms/algorithms/BaseOOD.py` — optimizer built over
  `requires_grad` params only (no-op unless something is frozen).
- `GOOD/networks/models/GSATGNNs.py`, `SMGNN.py` — opt-in `occ_hard_st`:
  straight-through hard display during training (returned tuple stays soft,
  so regularisers and the eval score channel are untouched). REAL-X uses
  REBAR here; ST is our documented simplification. DIR not yet ported.
- Configs: `{MUTAG,MNIST,SST2Planted,BAColorGVIsol}/basis/no_shift/
  {GSAT,SMGNN_sec6}_OCC.yaml` (include base yaml + `fix_mask_aggregation`,
  `occ_hard_st`, `occ_g0_ckpt: auto`). MUST be launched with `--save_tag occ`.
- `run_occ.sh` — phased launcher (prereg / check / baselines / Stage A / B /
  eval); logs under `logs_occ/`.

## 2026-09-03 — analysis.py:print_metric 聚合对缺类键崩溃(修复)

**症状**:`--task eval_metric` 在任一 seed 的模型只预测单一类时崩溃:
`ValueError: setting an array element with a sequence ... inhomogeneous shape ... (5,)`
命中格:`evalo_mutag_smgnn`(seed 1 常数预测器)、`evalo_mnist_smgnn`(5 个 seed 中 4 个塌到 chance)。

**根因**:`GOOD/utils/analysis.py` 的 per-seed 聚合块里
`if f"{c}_{div}" not in metrics_score[split][metric][i].keys(): continue`
中的 `i` 是上一个 seed 循环**泄漏**的变量(= 最后一个 seed 的下标),因此只检查了最后一个
seed 是否有该类。`metrics_score[...]` 是 `defaultdict(list)`,缺键索引返回 `[]`,于是
跨 seed 列表变成 `[[], [0.076], ...]` 参差数组,`np.nanmean` 无法堆叠。

**修复**:新增 `per_seed_values(metric_list, key)`——逐 seed 用 `in` 测存在性(不插键),
缺失者按宽度补 `NaN`,全员缺失返回 `None` 由调用方跳过;三处调用点(per-class、
`all_{div}`、`rejection`)改用它。`print_metric` 加 `np.asarray(dtype=float)` 与
RuntimeWarning 抑制,使整列全 NaN 返回 NaN 而非崩溃。

**语义变化**:某类的均值现在只在**实际预测过该类的 seed** 上取(此前该情形直接崩溃,
无既有行为可破坏)。其余路径逐字不变。备份:`GOOD/utils/analysis.py.bak-*`。
