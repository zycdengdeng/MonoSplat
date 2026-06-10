#
# Fine-tune MonoSplat on CARLA with photometric self-supervision, mirroring the
# proven CARLA MVSplat recipe but respecting MonoSplat's design: the monocular
# foundation (DINOv2 + Depth-Anything depth_head) is FROZEN in the model code,
# so only the multi-view fusion + Gaussian-prediction params are updated. That
# preserves the generalizable monocular prior (MonoSplat's whole point) and
# avoids overfitting on the small CARLA set.
#
# Paradigm (same novel-view loss MonoSplat was pre-trained with):
#   per step: pick a random source view as target, its 6 direction-aligned
#   nearest views as context; encoder -> Gaussians; decoder renders the target;
#   loss = MSE + 0.05 * LPIPS(vgg) vs the real target image.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 python -m src.scripts.finetune_cse \
#       --train_glob '/mnt/zihanw/carla/input_output/*_dense' \
#       --checkpoint checkpoints/monosplat_re10k.ckpt \
#       --out checkpoints/monosplat_carla_ft \
#       --lr 5e-6 --steps 3000 --save_every 500 --warmup 100
#
# Then render/score each saved ckpt with run_monosplat_cse + eval_cse.py and
# report the best step (same protocol as the MVSplat fine-tune).
#
import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.scripts.run_monosplat_cse import (
    build_model, load_checkpoint, read_colmap, read_points3D, estimate_near_far,
    qvec2rotmat, normalized_K, c2w_from_qt, round_to_multiple, load_image_tensor,
)

TEST_IDS = {"110", "210", "310", "410", "510"}  # never train on the eval seqs


def prepare_scene(scene_dir, args):
    """Parse one CARLA training scene; return everything needed to sample from it."""
    sparse = os.path.join(scene_dir, "sparse", "0")
    images_dir = os.path.join(scene_dir, "images")
    if not (os.path.exists(os.path.join(sparse, "images.bin"))
            or os.path.exists(os.path.join(sparse, "images.txt"))):
        return None
    cams, imgs = read_colmap(sparse)
    names = [n for n in imgs if os.path.exists(os.path.join(images_dir, n))]
    if len(names) < args.n_context + 1:
        return None

    def center(n):
        q, t, _ = imgs[n]
        return -qvec2rotmat(q).T @ t

    def forward(n):
        q, _, _ = imgs[n]
        return qvec2rotmat(q)[2, :]

    centers = {n: center(n) for n in names}
    fwds = {n: forward(n) for n in names}

    pts = read_points3D(sparse)
    if pts is not None and len(pts) >= 10:
        near, far = estimate_near_far(pts, [imgs[n] for n in names])
    else:
        # rig fallback (named '<cam>_<frame>') — training scenes have points, so rare
        from src.scripts.run_monosplat_cse import estimate_near_far_from_rig
        near, far, _ = estimate_near_far_from_rig(imgs, names,
                                                  args.rig_radius, args.near_m, args.far_m)
    return dict(imgs=imgs, cams=cams, names=names, centers=centers, fwds=fwds,
                near=near, far=far, images_dir=images_dir)


def sample_batch(scene, args, device):
    imgs, cams, names = scene["imgs"], scene["cams"], scene["names"]
    centers, fwds = scene["centers"], scene["fwds"]
    near, far = scene["near"], scene["far"]

    anchor = random.choice(names)
    afwd, acen = fwds[anchor], centers[anchor]
    others = [n for n in names if n != anchor]
    aligned = [n for n in others if float(fwds[n] @ afwd) > args.align_thresh]
    pool = aligned if len(aligned) >= args.n_context else sorted(
        others, key=lambda n: -float(fwds[n] @ afwd))
    ctx = sorted(pool, key=lambda n: np.linalg.norm(centers[n] - acen))[: args.n_context]

    W, H, fx, fy, cx, cy = cams[imgs[anchor][2]]
    s = args.net_long_side / max(W, H)
    nw, nh = round_to_multiple(W * s, 16), round_to_multiple(H * s, 16)

    def K_of(n):
        cW, cH, cfx, cfy, ccx, ccy = cams[imgs[n][2]]
        return torch.from_numpy(normalized_K(cfx, cfy, ccx, ccy, cW, cH)).float()

    ctx_imgs = torch.stack([load_image_tensor(os.path.join(scene["images_dir"], c), (nw, nh))
                            for c in ctx])
    ctx_ext = torch.stack([torch.from_numpy(c2w_from_qt(*imgs[c][:2])).float() for c in ctx])
    ctx_K = torch.stack([K_of(c) for c in ctx])
    v = len(ctx)
    context = {
        "image": ctx_imgs[None].to(device),
        "extrinsics": ctx_ext[None].to(device),
        "intrinsics": ctx_K[None].to(device),
        "near": torch.full((1, v), near, device=device),
        "far": torch.full((1, v), far, device=device),
        "index": torch.arange(v, device=device)[None],
    }
    tgt_ext = torch.from_numpy(c2w_from_qt(*imgs[anchor][:2])).float()[None, None].to(device)
    tgt_K = K_of(anchor)[None, None].to(device)
    gt = load_image_tensor(os.path.join(scene["images_dir"], anchor), (nw, nh)).to(device)
    return context, tgt_ext, tgt_K, near, far, (nh, nw), gt


def main():
    ap = argparse.ArgumentParser(description="Fine-tune MonoSplat on CARLA (photometric).")
    ap.add_argument("--train_glob", default="/mnt/zihanw/carla/input_output/*_dense",
                    help="glob for CARLA training scene dirs (each has sparse/0 + images/)")
    ap.add_argument("--checkpoint", required=True, help="MonoSplat init .ckpt")
    ap.add_argument("--out", required=True, help="output dir for fine-tuned ckpts")
    ap.add_argument("--experiment", default="re10k")
    ap.add_argument("--n_context", type=int, default=6)
    ap.add_argument("--align_thresh", type=float, default=0.5)
    ap.add_argument("--net_long_side", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--lpips_weight", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rig_radius", type=float, default=0.75)
    ap.add_argument("--near_m", type=float, default=0.5)
    ap.add_argument("--far_m", type=float, default=100.0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder, decoder = build_model(args.n_context, args.experiment, device)
    load_checkpoint(encoder, decoder, args.checkpoint)
    encoder.train()
    decoder.eval()  # splatting decoder is parameter-free

    # Only the trainable (non-frozen) params: DINOv2 + depth_head stay frozen.
    params = [p for p in encoder.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in encoder.parameters())
    print(f"[finetune] trainable {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params "
          f"(DINOv2 + depth_head frozen)")
    opt = torch.optim.Adam(params, lr=args.lr)

    import lpips as lpips_pkg
    lpips_fn = lpips_pkg.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    # Load all training scenes.
    scene_dirs = sorted(glob.glob(args.train_glob))
    scenes = []
    for d in scene_dirs:
        sid = os.path.basename(d).split("_")[0]
        if sid in TEST_IDS:
            continue
        sc = prepare_scene(d, args)
        if sc is not None:
            scenes.append((os.path.basename(d), sc))
    assert scenes, f"no usable training scenes under {args.train_glob}"
    print(f"[finetune] {len(scenes)} training scenes: "
          f"{', '.join(n for n, _ in scenes[:8])}{' ...' if len(scenes) > 8 else ''}")

    os.makedirs(args.out, exist_ok=True)

    def save(step):
        path = os.path.join(args.out, f"finetune_{step:06d}.ckpt")
        state = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
        torch.save({"state_dict": state, "step": step}, path)
        print(f"  -> saved {path}")

    running = 0.0
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(1, args.warmup))
        for g in opt.param_groups:
            g["lr"] = lr

        _, scene = random.choice(scenes)
        context, tgt_ext, tgt_K, near, far, (nh, nw), gt = sample_batch(scene, args, device)
        gaussians = encoder(context, step, deterministic=False)
        out = decoder.forward(
            gaussians, tgt_ext, tgt_K,
            torch.full((1, 1), near, device=device),
            torch.full((1, 1), far, device=device),
            (nh, nw), depth_mode=None,
        )
        pred = out.color[0, 0]  # no clamp: keep gradients on saturated pixels
        loss_mse = F.mse_loss(pred, gt)
        loss_lpips = lpips_fn(pred[None], gt[None], normalize=True).mean()
        loss = loss_mse + args.lpips_weight * loss_lpips

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running += loss.item()
        if step % 50 == 0:
            print(f"step {step:5d} | lr {lr:.2e} | loss {running/50:.4f} "
                  f"(mse {loss_mse.item():.4f} lpips {loss_lpips.item():.4f})")
            running = 0.0
        if step % args.save_every == 0:
            save(step)
    if args.steps % args.save_every != 0:
        save(args.steps)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
