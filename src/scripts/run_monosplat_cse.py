#
# Run MonoSplat (feed-forward, generalizable 3DGS) on the CARLA CSE benchmark.
#
# This adapter lets MonoSplat be scored on the *exact same* Cross-Sensor (CSE)
# benchmark used for GS-Net, so that the numbers are apples-to-apples with
# `gsnet/eval_cse.py`. It addresses the reviewers' request to compare GS-Net
# against other feed-forward / generalizable Gaussian methods (e.g. MonoSplat).
#
# Protocol (per scene, see CARLA_CSE_DATA_1.md):
#   - 60 source views (odd cameras)  -> used as MonoSplat context.
#   - 60 target views (even cameras, listed in sparse/0/test.txt) -> to render.
# MonoSplat is feed-forward: for each target view we feed the N nearest source
# views as context, predict per-pixel Gaussians in one forward pass, and render
# the target pose. Renders are written with EXACTLY the target filename so that
# `eval_cse.py` matches them against images/<name>.
#
# Usage (single scene):
#   python -m src.scripts.run_monosplat_cse \
#       --scene runs/cse_scenes/110 \
#       --checkpoint /path/to/monosplat_re10k.ckpt \
#       --out monosplat/110/renders
#
# Then score with the shared evaluator (in the gsnet repo):
#   python gsnet/eval_cse.py --renders monosplat/110/renders --scene runs/cse_scenes/110
#
# All 5 scenes + averaging is driven by `--multi` in eval_cse.py after you have
# produced renders for 110/210/310/410/510 with this script.
#
import argparse
import os
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Build the model exactly like src/main.py (so the architecture matches the ckpt).
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder


# --------------------------------------------------------------------------- #
# COLMAP text/binary model parsing
# --------------------------------------------------------------------------- #
def qvec2rotmat(q):
    """COLMAP quaternion (qw, qx, qy, qz) -> world->camera rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_cameras_txt(path):
    """Returns {cam_id: (W, H, fx, fy, cx, cy)} for PINHOLE/SIMPLE_PINHOLE."""
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            t = line.split()
            cam_id = int(t[0])
            model = t[1]
            W, H = int(t[2]), int(t[3])
            params = list(map(float, t[4:]))
            if model in ("PINHOLE", "OPENCV"):
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                raise ValueError(f"Unsupported camera model {model}")
            cams[cam_id] = (W, H, fx, fy, cx, cy)
    return cams


def read_images_txt(path):
    """Returns {name: (qvec(4,), tvec(3,), cam_id)} from a COLMAP images.txt."""
    imgs = {}
    with open(path) as f:
        lines = [l for l in f]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        t = line.split()
        qvec = np.array(list(map(float, t[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, t[5:8])), dtype=np.float64)
        cam_id = int(t[8])
        name = t[9]
        imgs[name] = (qvec, tvec, cam_id)
        i += 2  # skip the 2D-point line that follows each image header
    return imgs


def read_points3D(scene_sparse):
    """Read points3D.txt or points3D.bin -> (N,3) float array (or None)."""
    txt = os.path.join(scene_sparse, "points3D.txt")
    binp = os.path.join(scene_sparse, "points3D.bin")
    if os.path.exists(txt):
        xyz = []
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                t = line.split()
                xyz.append([float(t[1]), float(t[2]), float(t[3])])
        return np.array(xyz, dtype=np.float64) if xyz else None
    if os.path.exists(binp):
        xyz = []
        with open(binp, "rb") as f:
            (num,) = struct.unpack("<Q", f.read(8))
            for _ in range(num):
                f.read(8)  # point3D_id
                x, y, z = struct.unpack("<ddd", f.read(24))
                xyz.append([x, y, z])
                f.read(3)  # rgb
                f.read(8)  # error
                (track_len,) = struct.unpack("<Q", f.read(8))
                f.read(8 * track_len)  # track elements
        return np.array(xyz, dtype=np.float64) if xyz else None
    plyp = os.path.join(scene_sparse, "points3D.ply")
    if os.path.exists(plyp):
        try:
            from plyfile import PlyData
            v = PlyData.read(plyp)["vertex"]
            xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]),
                            np.asarray(v["z"])], axis=1).astype(np.float64)
            return xyz if len(xyz) else None
        except Exception as e:
            print(f"[warn] failed to read {plyp}: {e}")
            return None
    return None


def read_test_txt(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def c2w_from_qt(qvec, tvec):
    """COLMAP (world->cam) -> OpenCV cam->world 4x4 (MonoSplat extrinsics)."""
    R = qvec2rotmat(qvec)  # world->cam
    C = -R.T @ tvec  # camera center in world
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = C
    return c2w


def normalized_K(fx, fy, cx, cy, W, H):
    """MonoSplat uses resolution-normalized intrinsics (fx/W, fy/H, cx/W, cy/H)."""
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx / W
    K[1, 1] = fy / H
    K[0, 2] = cx / W
    K[1, 2] = cy / H
    return K


def estimate_near_far(points, src_qt, percentiles=(1.0, 99.0), margin=0.2):
    """Robust near/far in COLMAP units from sparse points projected into source cams."""
    depths = []
    for (qvec, tvec, _) in src_qt:
        R = qvec2rotmat(qvec)
        cam = (R @ points.T).T + tvec  # world->cam
        z = cam[:, 2]
        depths.append(z[z > 0])
    depths = np.concatenate(depths)
    lo = np.percentile(depths, percentiles[0])
    hi = np.percentile(depths, percentiles[1])
    near = max(lo * (1 - margin), 1e-3)
    far = hi * (1 + margin)
    return float(near), float(far)


def round_to_multiple(x, m=16):
    return max(m, int(round(x / m)) * m)


def estimate_near_far_from_rig(imgs, src_names, rig_radius_m=0.75,
                               near_m=0.5, far_m=100.0):
    """near/far when there are no points3D, using the known CARLA rig geometry.

    The source views are 6 cameras of a radius-`rig_radius_m` ring per frame.
    Grouping source views by frame (the token after the last '_' in the name),
    the mean distance of a frame's camera centers to their centroid ~= the ring
    radius in COLMAP units. That gives a COLMAP-units-per-meter scale, which we
    use to turn a metric depth bracket [near_m, far_m] into COLMAP units.
    """
    from collections import defaultdict

    def center(n):
        q, t, _ = imgs[n]
        return -qvec2rotmat(q).T @ t

    groups = defaultdict(list)
    for n in src_names:
        frame = os.path.splitext(n)[0].split("_")[-1]
        groups[frame].append(center(n))
    radii = [float(np.linalg.norm(np.asarray(cs) - np.asarray(cs).mean(0), axis=1).mean())
             for cs in groups.values() if len(cs) >= 3]
    if not radii:
        raise RuntimeError(
            "no points3D and could not group source cameras into rig frames "
            "(expected names like '<cam>_<frame>.ext'); pass --near/--far explicitly.")
    ring = float(np.median(radii))
    scale = ring / rig_radius_m  # COLMAP units per meter
    near, far = near_m * scale, far_m * scale
    info = (f"ring~{ring:.4f}u (={rig_radius_m}m) -> {scale:.4f} u/m | "
            f"near={near:.4f} far={far:.4f} (={near_m}-{far_m}m, {len(radii)} frames)")
    return near, far, info



# --------------------------------------------------------------------------- #
# Model construction (mirrors src/main.py so the checkpoint loads cleanly)
# --------------------------------------------------------------------------- #
def build_model(n_context, experiment, device):
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = str(repo_root / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg_dict = compose(
            config_name="main",
            overrides=[
                f"+experiment={experiment}",
                "mode=test",
                f"dataset.view_sampler.num_context_views={n_context}",
                "wandb.mode=disabled",
            ],
        )
    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    return encoder, decoder


def load_checkpoint(encoder, decoder, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    enc_sd = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    dec_sd = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
    if not enc_sd:  # checkpoint may already be encoder-only
        enc_sd = state
    me, ue = encoder.load_state_dict(enc_sd, strict=False)
    if dec_sd:
        decoder.load_state_dict(dec_sd, strict=False)
    # The frozen DINOv2 / depth backbone is restored via the ckpt; warn only on
    # unexpected gaps in trained weights.
    trained_missing = [k for k in me if "pretrained" not in k and "depth_head" not in k]
    if trained_missing:
        print(f"[warn] {len(trained_missing)} trained encoder keys missing from ckpt "
              f"(showing 5): {trained_missing[:5]}")
    if ue:
        print(f"[warn] {len(ue)} unexpected keys in ckpt (showing 5): {ue[:5]}")


# --------------------------------------------------------------------------- #
# Per-scene rendering
# --------------------------------------------------------------------------- #
def load_image_tensor(path, size_wh=None):
    img = Image.open(path).convert("RGB")
    if size_wh is not None:
        img = img.resize(size_wh, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # 3,H,W


@torch.no_grad()
def run_scene(encoder, decoder, scene, out_dir, args, device):
    sparse = os.path.join(scene, "sparse", "0")
    images_dir = os.path.join(scene, "images")
    cams = read_cameras_txt(os.path.join(sparse, "cameras.txt"))
    imgs = read_images_txt(os.path.join(sparse, "images.txt"))
    test_names = read_test_txt(os.path.join(sparse, "test.txt"))
    test_set = set(test_names)

    src_names = [n for n in imgs if n not in test_set]
    assert src_names, f"no source views found in {scene}"
    assert test_names, f"no target views in test.txt for {scene}"

    # Camera centers (world) for context selection.
    def center(name):
        q, t, _ = imgs[name]
        return -qvec2rotmat(q).T @ t

    src_centers = {n: center(n) for n in src_names}

    # near / far in COLMAP units.
    if args.near is not None and args.far is not None:
        near, far = args.near, args.far
    else:
        pts = read_points3D(sparse)
        if pts is not None and len(pts) >= 10:
            src_qt = [imgs[n] for n in src_names]
            near, far = estimate_near_far(pts, src_qt)
        else:
            # No sparse points (CARLA CSE scenes): derive scale from the rig.
            near, far, info = estimate_near_far_from_rig(
                imgs, src_names, args.rig_radius, args.near_m, args.far_m)
            print(f"  [no points3D] {info}")
    print(f"[{os.path.basename(scene.rstrip('/'))}] near={near:.4f} far={far:.4f} "
          f"| {len(src_names)} src, {len(test_names)} target")

    os.makedirs(out_dir, exist_ok=True)

    for name in test_names:
        if name not in imgs:
            print(f"[warn] {name} in test.txt but not in images.txt; skipping")
            continue
        q_t, t_t, cam_t = imgs[name]
        W, H, fx, fy, cx, cy = cams[cam_t]
        tgt_center = center(name)

        # Pick N nearest source views as context.
        order = sorted(src_names, key=lambda n: np.linalg.norm(src_centers[n] - tgt_center))
        ctx_names = order[: args.n_context]

        # Network input size for context (preserve aspect, divisible by 16).
        scale = args.net_long_side / max(W, H)
        net_w = round_to_multiple(W * scale, 16)
        net_h = round_to_multiple(H * scale, 16)

        # Build context batch.
        ctx_imgs, ctx_ext, ctx_K = [], [], []
        for cn in ctx_names:
            cq, ct, ccam = imgs[cn]
            cW, cH, cfx, cfy, ccx, ccy = cams[ccam]
            ctx_imgs.append(load_image_tensor(os.path.join(images_dir, cn), (net_w, net_h)))
            ctx_ext.append(torch.from_numpy(c2w_from_qt(cq, ct)).float())
            ctx_K.append(torch.from_numpy(normalized_K(cfx, cfy, ccx, ccy, cW, cH)).float())

        v = len(ctx_names)
        context = {
            "image": torch.stack(ctx_imgs)[None].to(device),       # 1,v,3,h,w
            "extrinsics": torch.stack(ctx_ext)[None].to(device),   # 1,v,4,4
            "intrinsics": torch.stack(ctx_K)[None].to(device),     # 1,v,3,3
            "near": torch.full((1, v), near, device=device),
            "far": torch.full((1, v), far, device=device),
            "index": torch.arange(v, device=device)[None],
        }

        gaussians = encoder(context, 0, deterministic=True)

        # Render the target pose at the GT resolution (so eval_cse.py shapes match).
        tgt_ext = torch.from_numpy(c2w_from_qt(q_t, t_t)).float()[None, None].to(device)
        tgt_K = torch.from_numpy(normalized_K(fx, fy, cx, cy, W, H)).float()[None, None].to(device)
        out = decoder.forward(
            gaussians,
            tgt_ext,
            tgt_K,
            torch.full((1, 1), near, device=device),
            torch.full((1, 1), far, device=device),
            (H, W),
            depth_mode=None,
        )
        color = out.color[0, 0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        Image.fromarray((color * 255).round().astype(np.uint8)).save(
            os.path.join(out_dir, name))

    print(f"  -> wrote {len(test_names)} renders to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Run MonoSplat on the CARLA CSE benchmark.")
    ap.add_argument("--scene", help="single cse_scenes/<id> dir")
    ap.add_argument("--out", help="output renders dir for --scene")
    ap.add_argument("--multi", nargs="+",
                    help="space-separated 'scene_dir:out_dir' pairs for all sequences")
    ap.add_argument("--checkpoint", required=True, help="MonoSplat .ckpt path")
    ap.add_argument("--experiment", default="re10k",
                    help="base experiment config (re10k or dtu); controls model hyperparams")
    ap.add_argument("--n_context", type=int, default=2,
                    help="number of nearest source views fed as context per target")
    ap.add_argument("--net_long_side", type=int, default=256,
                    help="context network input long side (rounded to /16)")
    ap.add_argument("--near", type=float, default=None, help="override near (COLMAP units)")
    ap.add_argument("--far", type=float, default=None, help="override far (COLMAP units)")
    ap.add_argument("--rig_radius", type=float, default=0.75,
                    help="CARLA rig radius in meters (for near/far when no points3D)")
    ap.add_argument("--near_m", type=float, default=0.5,
                    help="metric near (m) used with rig-based scale")
    ap.add_argument("--far_m", type=float, default=100.0,
                    help="metric far (m) used with rig-based scale")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, decoder = build_model(args.n_context, args.experiment, device)
    load_checkpoint(encoder, decoder, args.checkpoint)

    pairs = []
    if args.multi:
        for p in args.multi:
            sc, od = p.split(":")
            pairs.append((sc, od))
    else:
        assert args.scene and args.out, "provide --scene and --out (or --multi)"
        pairs.append((args.scene, args.out))

    for sc, od in pairs:
        run_scene(encoder, decoder, sc, od, args, device)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
