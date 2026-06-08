#!/usr/bin/env bash
#
# One-shot environment setup for MonoSplat on a (possibly China-network) GPU server.
# - Sets gh-proxy.com as the GLOBAL git mirror for github.com / raw.githubusercontent.com
# - Creates the conda env and installs torch matching your CUDA (12.1 or 11.8)
# - Installs requirements (incl. the git+github diff-gaussian-rasterizer via the mirror)
# - Pre-fetches the DINOv2 backbone into the torch.hub cache through the mirror
#
# Usage:
#   bash setup_env.sh 121      # CUDA 12.1  (default)
#   bash setup_env.sh 118      # CUDA 11.8
#
set -euo pipefail

CUDA="${1:-121}"
ENV_NAME="${ENV_NAME:-monosplat}"
PROXY="https://gh-proxy.com/"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$CUDA" != "121" && "$CUDA" != "118" ]]; then
  echo "CUDA must be 121 or 118 (got '$CUDA')"; exit 1
fi
echo ">>> Target CUDA: cu${CUDA}   conda env: ${ENV_NAME}"

# --------------------------------------------------------------------------- #
# 1) Global GitHub mirror (gh-proxy.com). Affects ALL git clones, including
#    `pip install git+https://github.com/...`. urllib downloads (e.g. torch.hub
#    zipballs) are NOT git, so we also pre-fetch DINOv2 by git in step 5.
# --------------------------------------------------------------------------- #
echo ">>> [1/5] Configuring global git mirror -> ${PROXY}"
git config --global url."${PROXY}https://github.com/".insteadOf "https://github.com/"
git config --global url."${PROXY}https://raw.githubusercontent.com/".insteadOf "https://raw.githubusercontent.com/"
# Robustness on flaky networks: HTTP/1.1 avoids the common
# "RPC failed; curl 92 HTTP/2 stream ... INTERNAL_ERROR" / "early EOF" on clones.
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999
echo "    git insteadOf rules:"
git config --global --get-regexp 'url\..*\.insteadof' || true

# --------------------------------------------------------------------------- #
# 2) Conda env
# --------------------------------------------------------------------------- #
echo ">>> [2/5] Creating conda env '${ENV_NAME}' (python 3.10)"
if ! command -v conda >/dev/null 2>&1; then
  echo "    conda not found on PATH. Install Miniconda/Anaconda first."; exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "${ENV_NAME}" python=3.10
conda activate "${ENV_NAME}"

# --------------------------------------------------------------------------- #
# 3) PyTorch 2.1.2 for the chosen CUDA. (PyTorch wheels are on download.pytorch.org,
#    not github, so no proxy needed here.)
# --------------------------------------------------------------------------- #
echo ">>> [3/5] Installing torch 2.1.2 (cu${CUDA})"
pip install --upgrade pip wheel
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url "https://download.pytorch.org/whl/cu${CUDA}"

# --------------------------------------------------------------------------- #
# 4) Requirements. The `git+https://github.com/...` line (diff-gaussian-
#    rasterization-modified) is compiled from source and routed via the mirror.
#    It needs a matching CUDA toolkit; set CUDA_HOME if nvcc isn't auto-found.
# --------------------------------------------------------------------------- #
echo ">>> [4/5] Installing requirements.txt"
if [[ -z "${CUDA_HOME:-}" ]]; then
  for c in "/usr/local/cuda-${CUDA:0:2}.${CUDA:2}" "/usr/local/cuda-12.1" "/usr/local/cuda-11.8" "/usr/local/cuda"; do
    [[ -d "$c" ]] && export CUDA_HOME="$c" && break
  done
fi
echo "    CUDA_HOME=${CUDA_HOME:-<unset>} (needed to compile diff-gaussian-rasterization)"

# The modified rasterizer pulls a git submodule (glm). Letting pip recurse
# submodules is fragile on flaky networks, so pre-clone it recursively (with
# retries) through the mirror and install from the local path instead.
RAST_DIR="${REPO_DIR}/third_party/diff-gaussian-rasterization-modified"
mkdir -p "${REPO_DIR}/third_party"
if [[ ! -d "${RAST_DIR}/.git" ]]; then
  for i in 1 2 3 4 5; do
    git clone --recursive \
      https://github.com/dcharatan/diff-gaussian-rasterization-modified "${RAST_DIR}" && break
    echo "    rasterizer clone retry ${i}..."; sleep 3
  done
fi
for i in 1 2 3 4 5; do
  ( cd "${RAST_DIR}" && git submodule update --init --recursive ) && break
  echo "    submodule retry ${i}..."; sleep 3
done
# Fallback: if the glm submodule still didn't land (flaky git-over-proxy keeps
# throwing 'HTTP/2 stream ... INTERNAL_ERROR'), fetch the *pinned* glm snapshot
# as a zip via the mirror. Archive downloads through gh-proxy are far more
# reliable than recursive git clones.
if [[ ! -e "${RAST_DIR}/third_party/glm/glm/glm.hpp" ]]; then
  echo "    glm submodule missing; fetching pinned snapshot as zip via mirror"
  GLM_SHA=$( cd "${RAST_DIR}" && git ls-tree HEAD third_party/glm 2>/dev/null | awk '{print $3}' )
  GLM_SHA="${GLM_SHA:-master}"
  echo "    glm pinned at ${GLM_SHA}"
  rm -rf /tmp/glm_unzip && mkdir -p /tmp/glm_unzip
  for i in 1 2 3 4 5; do
    curl -fL "${PROXY}https://github.com/g-truc/glm/archive/${GLM_SHA}.zip" -o /tmp/glm_snapshot.zip \
      && unzip -q -o /tmp/glm_snapshot.zip -d /tmp/glm_unzip && break
    echo "    glm zip retry ${i}..."; sleep 3
  done
  rm -rf "${RAST_DIR}/third_party/glm"
  mv /tmp/glm_unzip/glm-* "${RAST_DIR}/third_party/glm"
fi
if [[ ! -e "${RAST_DIR}/third_party/glm/glm/glm.hpp" ]]; then
  echo "    ERROR: glm headers still missing under ${RAST_DIR}/third_party/glm -- check network."; exit 1
fi

# Install everything else first (skip the git+ line, handled below). Use a
# constraints file so deps like timm/pytorch_lightning (which only say
# torch>=2.1.0) can't UPGRADE torch/numpy/setuptools out from under us:
#   - torch must stay 2.1.2+cu121 (the cu version we installed)
#   - torch 2.1.2 needs NumPy 1.x (crashes under NumPy 2.x)
#   - torch's cpp_extension imports pkg_resources, dropped by setuptools >= 81
#   - e3nn >= 0.6 requires torch >= 2.2 (we're on 2.1.2)
#   - moviepy 2.x removed moviepy.editor, which model_wrapper.py imports
#   - torchvision 0.16.2 predates Pillow 11/12
CONSTRAINTS="${REPO_DIR}/third_party/constraints.txt"
cat > "${CONSTRAINTS}" <<'EOF'
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
numpy<2
setuptools<81
e3nn<0.6
moviepy<2
pillow<11
EOF
grep -v '^[[:space:]]*git+' "${REPO_DIR}/requirements.txt" \
  | pip install -c "${CONSTRAINTS}" -r /dev/stdin
pip install -c "${CONSTRAINTS}" "numpy<2" "setuptools<81" "e3nn<0.6" "moviepy<2" "pillow<11" ninja

echo "    sanity:"
python -c "import torch, numpy, PIL, moviepy.editor; print('    torch', torch.__version__, '| numpy', numpy.__version__, '| pillow', PIL.__version__)"

# --no-build-isolation: the rasterizer's setup.py imports torch at build time,
# which pip's isolated build env wouldn't have. Install into the torch-equipped
# conda env with isolation disabled.
pip install --no-build-isolation "${RAST_DIR}"

# --------------------------------------------------------------------------- #
# 5) Pre-fetch the DINOv2 backbone code into the torch.hub cache via the mirror.
#    torch.hub.load() downloads a zipball over urllib (not git), which would
#    bypass the git mirror -- so we git-clone it into the cache dir torch expects.
# --------------------------------------------------------------------------- #
echo ">>> [5/5] Pre-fetching DINOv2 into torch.hub cache"
HUB_DIR="${TORCH_HOME:-$HOME/.cache/torch}/hub"
mkdir -p "${HUB_DIR}"
DINO_DIR="${HUB_DIR}/facebookresearch_dinov2_main"
if [[ ! -d "${DINO_DIR}" ]]; then
  git clone --depth 1 "${PROXY}https://github.com/facebookresearch/dinov2" "${DINO_DIR}"
else
  echo "    already present: ${DINO_DIR}"
fi

cat <<EOF

>>> Done.

Activate with:   conda activate ${ENV_NAME}

Remaining downloads (NOT on github, so NOT proxied -- usually reachable directly):
  * DINOv2 pretrained weights (~88MB) auto-download from dl.fbaipublicfiles.com on
    first model build, cached at ${HUB_DIR}/checkpoints/dinov2_vits14_pretrain.pth .
    The full MonoSplat checkpoint also contains these frozen weights, so this is
    only needed to construct the model. If fbaipublicfiles is blocked, copy that
    .pth from another machine into that path.
  * UniMatch depth weights (only needed for TRAINING, not for CSE inference):
      wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints

Next: put your MonoSplat .ckpt somewhere, then run the CSE benchmark per MONOSPLAT_CSE.md.
EOF
