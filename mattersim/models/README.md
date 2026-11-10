# Model files

- `best_model.pth`: place the fine-tuned checkpoint here for the full MD commands in the parent README.
- `mattersim-v1.0.0-5M.pth`: the pretrained checkpoint used for fine-tuning and the short MD demo; obtain it with `python download_pretrained.py` from the parent directory.

Retraining exports a compact checkpoint to `training_output/export/best_model.pth`. Pass that path directly to `run_mlmd.py --model` when using the new training output.
