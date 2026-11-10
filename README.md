# High entropy layered cathodes code and data

Companion materials for **A chemical descriptor enabled data-driven discovery of high-entropy layered cathodes**.

Repository: https://github.com/ChenSu-insitu-sjtu/HEOF

| Directory | Contents | Run instructions |
|---|---|---|
| `mattersim/` | MatterSim fine-tuning on DFT energies and forces, held-out evaluation, and ASE molecular dynamics | [MatterSim README](mattersim/README.md) |
| `Code for RSM/` | Quadratic least-squares fits of `Cap.` and `Ret.` against PC1 and PC2, with the two Excel datasets | [Fitting README](Code%20for%20RSM/README.md) |

The MatterSim dataset is split into 160 training, 20 validation and 20 test configurations. The response-surface workflow is independent and provides a short CPU-only demonstration. MatterSim training requires a compatible NVIDIA GPU.

Custom scripts are released under the [MIT License](LICENSE). Third-party terms are described in [THIRD_PARTY.md](THIRD_PARTY.md).

`SHA256SUMS.txt` covers the scripts, documentation, configuration files and fixed example inputs/outputs. Replaceable PKL datasets and PTH checkpoints are excluded; the training workflow records their actual hashes in its output files.
