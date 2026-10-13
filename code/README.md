# SOAR-Flow reference code

This directory contains reference research programs for Flow policies, Gaussian task critics, intervention learning, CBF–MPC protection, data collection, training and evaluation. Original module and class names are retained so that the code can be traced to its research versions.

## Core modules

| Component | File |
| --- | --- |
| Flow actor and protected online learning | [`relax/algorithm/guardian_dsac_flow_paper.py`](relax/algorithm/guardian_dsac_flow_paper.py) |
| Gaussian task critics and intervention learning | [`relax/algorithm/guardian_dsac_v2_3.py`](relax/algorithm/guardian_dsac_v2_3.py) |
| Flow policy networks | [`relax/network/flow_policy.py`](relax/network/flow_policy.py) |
| Actor, critic and intervention networks | [`relax/network/safe_sac_jax.py`](relax/network/safe_sac_jax.py) |
| Distributional return utilities | [`relax/utils/distributional_q.py`](relax/utils/distributional_q.py) |
| Flat NMPC and barrier constraints | [`relax/control/flat_nmpc.py`](relax/control/flat_nmpc.py) |
| Driving references and control geometry | [`relax/control/drive_mgf.py`](relax/control/drive_mgf.py) |
| MetaDrive safety wrapper | [`relax/env/drive/lane_change_mpc.py`](relax/env/drive/lane_change_mpc.py) |
| Replay and transition records | [`relax/utils/experience.py`](relax/utils/experience.py), [`relax/buffer/seed_indexed.py`](relax/buffer/seed_indexed.py) |
| Offline data interface | [`relax/offline/guardian_dsac_multiseed_dataset.py`](relax/offline/guardian_dsac_multiseed_dataset.py) |
| Flow training | [`scripts/train_guardian_flow_dsac_paper.py`](scripts/train_guardian_flow_dsac_paper.py) |
| Checkpoint evaluation | [`scripts/evaluate_guardian_flow_dsac_paper.py`](scripts/evaluate_guardian_flow_dsac_paper.py) |
| Fresh-process policy evaluation | [`scripts/evaluate_guardian_dsac_isolated.py`](scripts/evaluate_guardian_dsac_isolated.py) |
| Data collection | [`scripts/collect_guardian_dsac_multiseed.py`](scripts/collect_guardian_dsac_multiseed.py) |

Older algorithm modules are included where the reference programs import reusable helpers or support earlier checkpoint formats.

## Environment

Use **Linux, Python 3.10 or 3.11**. Training uses POSIX file locking (`fcntl`) and launches isolated evaluation processes. The original formal training protocol targets an NVIDIA RTX 4080 SUPER. CPU mode is available for component checks; full training requires the configured GPU runtime and experiment assets.

From the repository root:

```bash
cd code
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,video]'
```

For the original CUDA 12 setup, install the `cuda12` extra in an environment with a compatible NVIDIA driver:

```bash
python -m pip install -e '.[cuda12,dev,video]'
```

## Entry points

Run commands from `code/`. Each script exposes its options through `--help`:

```bash
python scripts/collect_guardian_dsac_multiseed.py --help
python scripts/train_guardian_dsac_multiseed.py --help
python scripts/train_guardian_flow_dsac_paper.py --help
python scripts/evaluate_guardian_flow_dsac_paper.py --help
```

The training chain is data collection → offline Gaussian critic initialization → offline Flow initialization → protected online refinement → checkpoint evaluation. Supply your generated dataset manifest and checkpoints to the following templates.

```bash
# Collect the reference offline scenes.
python scripts/collect_guardian_dsac_multiseed.py \
  --output-root runs/offline_data --seeds 0,1,2,3,4

# Initialize shared offline critics and the intervention model.
python scripts/train_guardian_dsac_multiseed.py \
  --manifest runs/offline_data/manifest.json \
  --phase offline --output-root runs/shared --seed 0

# Initialize the offline Flow policy from the shared checkpoint.
python scripts/train_guardian_flow_dsac_paper.py \
  --method flow_bc --phase offline \
  --shared-offline-checkpoint /path/to/shared/final.pkl \
  --manifest runs/offline_data/manifest.json \
  --output-root runs/flow --seed 0

# Refine the Flow policy online under safety protection.
python scripts/train_guardian_flow_dsac_paper.py \
  --method full_flow --phase online \
  --shared-offline-checkpoint /path/to/shared/final.pkl \
  --flow-offline-checkpoint /path/to/flow_bc/final.pkl \
  --manifest runs/offline_data/manifest.json \
  --output-root runs/flow --seed 0

# Evaluate a checkpoint in isolated processes.
python scripts/evaluate_guardian_flow_dsac_paper.py \
  --checkpoint /path/to/full_flow/final.pkl \
  --output evaluation.json
```

Keep each checkpoint with its adjacent `config.json` and companion run metadata. These templates require user-generated data and weights; they do not download experiment assets. For short component checks, the training scripts expose `--component-smoke --no-require-gpu`.

## Tests

```bash
JAX_PLATFORM_NAME=cpu python -m pytest -q \
  tests/test_guardian_flow_dsac_paper.py \
  tests/test_guardian_dsac_v2_3.py \
  tests/test_guardian_flow_dsac.py \
  tests/test_flat_nmpc.py \
  tests/test_drive_mgf_target.py \
  tests/test_drive_mgf_geometry.py \
  tests/test_metasafe_cbf_collection.py
```

The repository includes a Linux CPU CI workflow for these reference component tests. It does not launch training or vehicle experiments.

## Source record

The programs were extracted from research revision `396a9b0b2aad84a27a8a21aaf97cefdfbf02b0a1`. [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) records the original paths and SHA-256 hashes. Core source files are preserved without algorithm changes. Packaging metadata and this guide were added for this release.

This release contains source and tests. Model weights, replay buffers, datasets, vehicle interfaces and paper source files are not included. No additional software license has been assigned.
