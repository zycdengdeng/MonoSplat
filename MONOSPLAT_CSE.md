# Running MonoSplat on the CARLA CSE benchmark (reviewer rebuttal)

This adds MonoSplat as a **feed-forward / generalizable 3DGS** competitor on the
same **Cross-Sensor (CSE)** benchmark used for GS-Net, scored by the *identical*
`gsnet/eval_cse.py`. It directly answers the reviewers who asked to compare
GS-Net against other feed-forward Gaussian methods (审稿意见 1 & 3) and to
position the work in the generalizable-3DGS literature (审稿意见 2).

Adapter: [`src/scripts/run_monosplat_cse.py`](src/scripts/run_monosplat_cse.py).

## Results (CARLA CSE, 5 seqs, same `eval_cse.py`)

| Method | PSNR | SSIM | LPIPS | Type |
|---|---|---|---|---|
| MonoSplat zero-shot (6-view) | 16.70 | 0.694 | 0.498 | feed-forward, no opt |
| **MonoSplat fine-tuned (CARLA, best @500)** | **17.91** | 0.690 | 0.456 | feed-forward |
| MVSplat zero-shot | 17.28 | – | – | feed-forward |
| MVSplat fine-tuned | 17.96 | 0.674 | 0.405 | feed-forward |
| 3DGS baseline | 18.00 | – | – | per-scene opt |
| **GS-Net + 3DGS (ours)** | **19.89** | – | – | per-scene opt |

MonoSplat fine-tune per-step CSE (PSNR/SSIM peak at step 500, then decline as
more fine-tuning overfits odd→odd interpolation and hurts odd→even cross-sensor
extrapolation): 500→17.91, 1000→17.45, 1500→17.13, 2000→16.90, 2500→16.66,
3000→16.51.

**Takeaway**: both leading feed-forward generalizable Gaussian methods (MVSplat,
MonoSplat), even after CARLA fine-tuning, plateau at ~17.9–18.0 PSNR — below
GS-Net+3DGS (19.89, +1.9). Cross-sensor extrapolation is a structural limit of
the feed-forward paradigm, which is exactly the gap GS-Net+3DGS fills.



---

## What it does

For each CSE scene `runs/cse_scenes/<id>/`:

1. Reads the COLMAP model (`cameras.txt`, `images.txt`, `test.txt`, `points3D.*`).
2. Splits **60 source** (not in `test.txt`) vs **60 target** (in `test.txt`).
3. For each target view, feeds the **N nearest source views** as MonoSplat
   context, predicts per-pixel Gaussians in one forward pass, and renders the
   **target pose at the GT resolution**.
4. Writes each render with the **exact target filename** so `eval_cse.py`
   matches it against `images/<name>`.

MonoSplat is purely feed-forward — there is **no per-scene optimization**, unlike
GS-Net + 3DGS. That is the point of the comparison.

---

## Prerequisites (on your GPU server)

- The MonoSplat conda env from `README.md` (torch 2.1.2 + `requirements.txt`,
  including `diff_gaussian_rasterization`).
- A MonoSplat checkpoint (the RealEstate10K-trained one used for the DTU
  cross-dataset table is the right analogue — it's a *generalizable* model, no
  fine-tuning on CARLA).
- **DINOv2 backbone**: the encoder calls `torch.hub.load("facebookresearch/dinov2", ...)`
  at construction. On an offline server, pre-populate the torch hub cache
  (`~/.cache/torch/hub/`) once from a machine with internet, or set
  `TORCH_HOME` to a directory that already has it. The frozen depth backbone
  weights themselves are restored from the checkpoint.

---

## Run

Render all 5 sequences in one process (model is loaded once):

```bash
cd MonoSplat
python -m src.scripts.run_monosplat_cse \
  --checkpoint /path/to/monosplat_re10k.ckpt \
  --multi \
    runs/cse_scenes/110:monosplat/110/renders \
    runs/cse_scenes/210:monosplat/210/renders \
    runs/cse_scenes/310:monosplat/310/renders \
    runs/cse_scenes/410:monosplat/410/renders \
    runs/cse_scenes/510:monosplat/510/renders
```

Or one scene at a time:

```bash
python -m src.scripts.run_monosplat_cse \
  --checkpoint /path/to/monosplat_re10k.ckpt \
  --scene runs/cse_scenes/110 --out monosplat/110/renders
```

## Score (identical evaluator, apples-to-apples)

```bash
python gsnet/eval_cse.py --multi \
  monosplat/110/renders:runs/cse_scenes/110 \
  monosplat/210/renders:runs/cse_scenes/210 \
  monosplat/310/renders:runs/cse_scenes/310 \
  monosplat/410/renders:runs/cse_scenes/410 \
  monosplat/510/renders:runs/cse_scenes/510 \
  --out monosplat/cse_scores.json
```

---

## Design choices (so the comparison is defensible)

- **Pose convention**: COLMAP `(q,t)` are world→camera. The adapter forms the
  OpenCV cam→world `[R^T | -R^T t]` that MonoSplat expects (`w2c.inverse()` in
  the original `dataset_re10k.py`).
- **Normalized intrinsics**: MonoSplat uses resolution-normalized `K`
  (`fx/W, fy/H, cx/W, cy/H`); the CUDA rasterizer derives FOV from these and
  assumes a centered principal point (true for the CARLA rig).
- **Targets rendered at native GT resolution** → render shapes match
  `images/<name>` exactly, and metrics are computed at the same resolution as
  GS-Net's. No cropping of GT.
- **Context at native aspect ratio** (resized so the long side ≈ `--net_long_side`,
  rounded to a multiple of 16), preserving the full field of view so renders
  don't get black borders from a square crop.
- **near/far** are estimated **in COLMAP units** from the sparse `points3D`
  projected into the source cameras (robust 1–99 percentiles + 20% margin).
  This is essential for cross-dataset depth: SfM scale is arbitrary, so DTU/re10k
  constants would be wrong. Override with `--near/--far` if needed.
- **Context count** `--n_context` defaults to 2 (matching the released re10k /
  DTU cross-dataset protocol). MonoSplat's cross-view attention is not tied to a
  fixed view count, so you can try 3–4 if you want a stronger MonoSplat number.

## Knobs to tune if numbers look off

| Symptom | Try |
|---|---|
| Washed-out / wrong depth scale | check `points3D` exist; tune `--near/--far`, or widen percentiles |
| Black borders at image edges | increase `--net_long_side` (e.g. 320/384) — keeps more FOV/detail |
| Too blurry | raise `--n_context` to 3–4 nearest source views |
| `points3D` missing | pass `--near/--far` explicitly (e.g. from the rig radius scale) |

---

## Reporting (rebuttal table)

| Seq | MonoSplat PSNR | SSIM | LPIPS |
|---|---|---|---|
| 110 | … | … | … |
| 210 | … | … | … |
| 310 | … | … | … |
| 410 | … | … | … |
| 510 | … | … | … |
| **Avg** | … | … | … |

Compare against our reference numbers (same evaluator):
**GS-Net + 3DGS (densify=2000) Avg PSNR ≈ 19.89** vs **3DGS-baseline 18.00**.
A feed-forward, no-per-scene-optimization MonoSplat number on this same CSE
protocol lets the paper state precisely where GS-Net sits relative to
generalizable Gaussian methods — exactly what the reviewers asked for.

---

## Fine-tuning MonoSplat on CARLA (optional, stronger rebuttal)

Zero-shot already shows MonoSplat < GS-Net. To also show a *CARLA-fine-tuned*
feed-forward method stays below GS-Net, fine-tune with photometric self-
supervision (`src/scripts/finetune_cse.py`).

Key MonoSplat-specific point: the monocular foundation (DINOv2 + Depth-Anything
`depth_head`) is **frozen** in the model code, so only the multi-view fusion +
Gaussian-prediction params train. That preserves the generalizable prior and
avoids overfitting the small CARLA set. The script filters `requires_grad`
params automatically.

```bash
# train (43 CARLA *_dense scenes; test seqs 110/210/310/410/510 auto-excluded)
CUDA_VISIBLE_DEVICES=2 python -m src.scripts.finetune_cse \
    --train_glob '/mnt/zihanw/carla/input_output/*_dense' \
    --checkpoint checkpoints/monosplat_re10k.ckpt \
    --out checkpoints/monosplat_carla_ft \
    --lr 5e-6 --steps 3000 --save_every 500 --warmup 100
# ~30 min on one A100; saves finetune_000500.ckpt ... finetune_003000.ckpt

# render + score each saved ckpt, report the best step (same protocol as MVSplat ft)
CUDA_VISIBLE_DEVICES=2 python -m src.scripts.run_monosplat_cse \
    --checkpoint checkpoints/monosplat_carla_ft/finetune_001000.ckpt \
    --multi runs/cse_scenes/110:monosplat_ft/110/renders ... runs/cse_scenes/510:monosplat_ft/510/renders
# then gsnet/eval_cse.py --multi monosplat_ft/<id>/renders:runs/cse_scenes/<id> ...
```

Recipe (mirrors the proven CARLA MVSplat run): lr 5e-6, warmup 100, Adam,
6 direction-aware context views, loss = MSE + 0.05·LPIPS(vgg), best ≈ step 1000
(early stop — lr higher than 5e-6 catastrophically forgets). Reference: MVSplat
zero-shot 17.28 → fine-tuned 17.96, still below GS-Net 19.89 / 3DGS 18.00.

