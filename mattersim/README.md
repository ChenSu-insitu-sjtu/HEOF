# MatterSim fine tuning and molecular dynamics

## Data and model

The reference dataset contains VASP DFT total energies (eV per structure) and atomic forces (eV/Angstrom). The three ASE PKLs are `data/train.pkl`, `data/validation.pkl` and `data/test.pkl`, containing 160, 20 and 20 configurations, respectively. The chronological training/validation/test split is 8:1:1. Stress labels are not used.

`data/dataset_manifest.json` specifies the label type, units and expected split sizes. The training script reads the current PKLs and automatically writes a dataset manifest with their actual hashes and a `frame_manifest.csv` into the chosen output directory. The test configurations are loaded only after validation-based checkpoint selection; the final frame index then covers all three splits. Input PKLs are not modified, and no precomputed frame-index file is required in `data/`.

Fine-tuning starts from MatterSim v1.0.0 5M. Default settings are batch size 1, a maximum of 200 epochs, learning rate 0.0001, Huber delta 0.01, equal energy/force loss weighting, no stress loss, validation-loss early-stopping patience 10, StepLR step size 10 and gamma 0.95, seed 42, and 5/4 Angstrom pair/three-body cutoffs. Pretrained energy normalization is retained. `config/training_config.json` documents these defaults; the actual settings are saved in each run's `run_config.json`.

## System requirements and installation

Tested software environment: Ubuntu 24.04.2 LTS, Python 3.10.20, MatterSim 1.2.2, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, torchmetrics 1.9.0, ASE 3.28.0 and NumPy 2.2.6. The environment check used an NVIDIA RTX 5090 with an approximately 8 GiB allocation; see `config/tested_environment.json`. Use a CUDA 12.8 compatible NVIDIA driver. Training requires CUDA. The MD script also permits CPU execution. Other training environments have not been tested.

From this directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_pretrained.py
```

Allow approximately 10-30 minutes for installation and model download on a desktop with broadband internet. This estimate depends on download speed. The approximately 91 MB pretrained checkpoint is downloaded directly from its publisher and its SHA256 is verified.

## Short demo

```bash
python train_mattersim.py --data data --output demo_training --checkpoint models/mattersim-v1.0.0-5M.pth --batch-size 1 --smoke
```

The demo performs one training pass over four training configurations. Expected output: `demo_training/smoke_batch1.json` with `status: SMOKE_PASSED` and finite loss/error values, plus the dataset and frame indices. It does not create a final trained model. Allow approximately 1-3 minutes on a comparable GPU desktop after dependencies and the checkpoint are available; this is an estimate.

To demonstrate MD trajectory generation:

```bash
python run_mlmd.py --structure structures/CONTCAR3 --model models/mattersim-v1.0.0-5M.pth --output demo_md.traj --steps 10 --repeat 1 1 1 --seed 42
```

Expected outputs are `demo_md.traj` and `demo_md.json`. This short example uses the pretrained checkpoint. A CPU-only software demonstration is also available in `../Code for RSM/`.

## Fine tuning and held-out evaluation

```bash
python train_mattersim.py --data data --output training_output --checkpoint models/mattersim-v1.0.0-5M.pth --batch-size 1 --epochs 200
```

Outputs under `training_output/` include:

- `run_config.json`, `dataset_manifest.json` and `frame_manifest.csv` for the actual inputs and settings.
- `learning_curve.csv`, `checkpoints/best_model.pth` and the compact `export/best_model.pth`.
- `baseline_validation_metrics.json`, `best_validation_metrics.json`, `test_metrics.json` and `training_result.json`.
- `test_predictions.csv` and `test_forces.npz` for parity-plot analysis against the DFT labels.

Validation Huber loss selects the checkpoint. Test labels are used only for the final evaluation. Energy errors are reported per atom; force errors are calculated over Cartesian force components. No new data split is made by the script.

## Molecular dynamics with the fine-tuned model

Place the fine-tuned checkpoint at `models/best_model.pth`, then run:

```bash
python run_mlmd.py --structure structures/CONTCAR3 --model models/best_model.pth --output md_CONTCAR3.traj
python run_mlmd.py --structure structures/CONTCAR6 --model models/best_model.pth --output md_CONTCAR6.traj
```

When using a newly retrained model, pass `training_output/export/best_model.pth` as `--model` instead.

The MD defaults are a 1 x 2 x 2 supercell, periodic boundaries, Maxwell-Boltzmann initial velocities, 1000 K, Nose-Hoover chain NVT, a 2 fs timestep, a 200 fs thermostat damping time and 50,000 steps (100 ps). Trajectories are written every step. Use `--temperature` to set the temperature required for a particular simulation, and optional `--seed` to fix velocity initialization. The analysis window is the central 60 ps after excluding the first and last 20 ps. The MD script generates trajectories; MSD/diffusion postprocessing is not included.

## Run on your data

For fine-tuning, provide three lists of ASE `Atoms` with stored energy and force labels, using the same filenames under a directory passed with `--data`. Retain the supplied data specification, adjusting its `expected_frames` entries and split description when using a different dataset. The script checks finite values, force-array shapes and duplicate geometries. Additional ASE metadata or stress arrays are permitted but not used for training. File hashes and frame indices are generated automatically from the current inputs.

For MD, pass a VASP-format structure with `--structure`, a fine-tuned checkpoint with `--model`, and a new `--output` path. Temperature, timestep, step count, supercell size and velocity seed are command-line arguments.
