# Environment and execution

## Validated CPU workflow

Use Python 3.13 and run from the repository root. A new virtual environment keeps project
packages separate from other projects:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-replay.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -W error
python src/check_reproducibility.py --output outputs/reproducibility.json
```

On Windows use `py -3.13 -m venv .venv`, then `.venv\Scripts\Activate.ps1`.
Set the three environment variables with PowerShell `$env:NAME="value"` assignments before
running `python -m pytest -q -W error`. Worker tests use `spawn` and select the GNU MKL
threading layer to avoid mixed OpenMP runtimes.

`requirements-replay.txt` pins the direct dependencies validated for CPU checkpoint inference,
loaders and tests. It is not a transitive lock or the exact original GPU training environment.
`requirements.txt` covers the full project, including matplotlib, notebooks and diffusion
generation, with a required Pillow pin. Installing it alone does not reproduce historical
training environments. For the full workflow while retaining the validated replay versions:

```bash
python -m pip install -r requirements-replay.txt -r requirements.txt
```

The offline audit needs only the Python standard library, tracked artifacts and Git history.
No images, checkpoints, CUDA or package installation are required for that audit.
A shallow clone may need `git fetch --unshallow` for the historical baseline source check.

## GPU workflow

Install a CUDA-enabled matching torch/torchvision pair appropriate to the target driver
before the project requirements. Verify the actual environment:

```bash
python -c 'import torch, torchvision, PIL; print(torch.__version__, torchvision.__version__, PIL.__version__); print("CUDA:", torch.cuda.is_available())'
```

Use `--device cuda --amp` for the recorded training protocol. CPU runs omit `--amp`.
Generation uses Stable Diffusion 1.5 in FP16 and was planned for GPUs with at least 8 GB
VRAM; the generation workflow was not rerun for this documentation deliverable. Different
hardware, CUDA, package versions and decoding builds can change numerical results even
with equal seeds. Run configs record the available software/hardware information.

## Required external artifacts

| Task | Requirements beyond tracked repository files |
| --- | --- |
| Offline saved-result audit | Git history for the legacy baseline source |
| All loader tests and preprocessing preview | Processed CUB images; matplotlib for preview |
| New classifier training | Processed images; ImageNet weights (downloaded on first use) |
| Reviewed-synthetic training | Accepted image files matching the archived manifest hashes |
| Checkpoint replay | Matching `best.pt` archive and processed images |
| New generation | Full dependencies, generator weights, CUDA GPU |

Download CUB with `python src/load_data.py`. Images save to
`data/processed/cub_200_2011/`. Synthetic paths resolve relative to the manifest directory and must match the
manifest; extract the team archive preserving `synthetic/sd15_wildlife/<species>/...`.
Copy the archived prepared manifest and leakage report into
`data/synthetic/sd15_wildlife/prepared_accepted/` as shown in the README before synthetic
training. Passing the archived CSV in `reports/` directly resolves its `../<species>/...`
paths to the wrong directory.
Restore standard checkpoints beneath
`outputs/kaggle_standard_aug/runs/standard_aug_10shot_seed{0,1,2}/best.pt` for the existing
replay commands. Data, outputs and `.venv` are ignored by Git. No public shared-storage
URL is recorded in this repository; obtain the archives from their owners.

## Troubleshooting and reproducibility limits

- **Changed validation predictions:** check Pillow is 12.2.0 in the active interpreter.
  The previous 11.1.0 build changed five predictions across three standard runs; the
  validated 12.2.0 environment matched all 3,000. See [replay evidence](reproducibility.md).
- **Seven skipped image checks:** run `python src/load_data.py` to restore processed images.
  Tests do not download them automatically.
- **CUDA unavailable:** check the active interpreter and PyTorch build. CPU-only wheels
  support data checks and classifier inference but not the recorded CUDA/AMP protocol.
- **Existing output directory:** choose a new `--output`; training/evaluation preserve old runs.
- **Audit completeness:** `--require-complete` now passes for all nine primary runs;
  the checker covers 15 validation runs. It preserves T4 seed-0 source verification,
  while final reporting selects Karina’s seed-0 bundle via `evaluation_selection.py`.
- **Manifest hash differs across Windows/Linux:** the audit accepts LF/CRLF equivalents,
  not content edits. Avoid rebuilding splits for replay.

Do not use test scores to adjust preprocessing, select synthetic images or choose
checkpoints. The final test exports are already tracked under `reports/test/`; no new
inference is required to recompute their hard-label metrics. Their original inference
environment records are unavailable; see the evaluation README for provenance limits.
