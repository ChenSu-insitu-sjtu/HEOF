# Quadratic response surface fitting

This folder contains the two supplied RSM notebooks and their Excel input data. It is independent of the MatterSim workflow.

## Files and method

| Input workbook | Columns | Notebook |
|---|---|---|
| `RawData for PC1-PC2-Cap.xlsx` | `PC1`, `PC2`, `Cap.` | `scientificProject/PC1-PC2-Cap.ipynb` |
| `RawData for PC1-PC2-RE.xlsx` | `PC1`, `PC2`, `Ret.` | `scientificProject/PC1-PC2-RE.ipynb` |

Each dataset has 12 observations. The Excel values are unchanged. Each response is fitted separately using ordinary least squares:

`z = b0 + b1*PC1 + b2*PC2 + b3*PC1^2 + b4*PC2^2 + b5*PC1*PC2`.

The responses retain the supplied column names and scales. PC1 and PC2 are precomputed inputs; PCA is not performed by these scripts. All observations are used in each fit, so R-squared and RMSE describe the in-sample fit.

## Requirements and installation

Tested on Windows 11 with Python 3.12.14 and the package versions in `requirements.txt`. No GPU or other specialized hardware is required. On systems without a Chinese font, install a suitable font such as Noto Sans CJK SC to display Chinese labels in the notebooks.

From the repository root:

```bash
cd "Code for RSM"
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS instead:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Allow approximately 2-5 minutes for installation on a desktop with broadband internet; download speed affects this estimate.

## Demo

```bash
python run_rsm.py
```

This command runs the actual code cells from both supplied notebooks and saves output under `results/Cap/` and `results/RE/`. Each directory contains a response-surface PNG, a text report with fitting/ANOVA/residual statistics, coefficients, fitted values and a metrics JSON file. A Jupyter installation is not required for this command.

| Response | R-squared | Adjusted R-squared | RMSE |
|---|---:|---:|---:|
| Cap. | 0.725562 | 0.496864 | 22.018726 |
| Ret. | 0.639606 | 0.339278 | 25.212150 |

The demo normally finishes within one minute on a desktop after dependencies are installed. The measured execution/export time and environment are recorded in `results/environment_and_timing.json`.

To use the notebooks interactively, optionally install Jupyter with `python -m pip install notebook`, then run `jupyter notebook` from this folder and open the notebook under `scientificProject/`. Use Run All. Input paths are relative to each notebook's directory.

## Use your own data

Replace the values in the corresponding workbook while preserving its column names, then run the same command. For a different input filename, edit the `file_path` cell in the relevant notebook. Use `python run_rsm.py --output new_results` to keep a separate output directory. These are ordinary least-squares fits without a train/test split or hyperparameter search.

The packaged notebooks retain the supplied fitting formula. Input paths, renamed Excel column references and plot response labels have been aligned with the supplied files. Stale saved outputs and obsolete Python 2 metadata have been removed.
