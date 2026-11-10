"""Fine-tune MatterSim-5M using ASE PKLs with VASP DFT energy/force labels.

The test split is evaluated only after model selection using validation loss.
File hashes and frame indices are generated from the current inputs at run time.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield import MatterSimCalculator
from mattersim.forcefield.potential import Potential


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def geometry_hash(a):
    h = hashlib.sha256()
    for array, dtype in [(a.numbers, '<i8'), (a.positions, '<f8'), (a.cell.array, '<f8'), (a.pbc, 'u1')]:
        h.update(np.asarray(array, dtype=dtype).tobytes(order='C'))
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def load_split(data, split, manifest, records):
    path = data / (split + '.pkl')
    assert sha256(path) == manifest['files'][path.name]['sha256']
    with path.open('rb') as f:
        atoms = pickle.load(f)
    expected = manifest['files'][path.name]['expected_frames']
    if not isinstance(atoms, (list, tuple)) or len(atoms) != expected:
        raise ValueError(f'{path.name}: expected a list of {expected} ASE Atoms objects')
    rows = []
    for i, a in enumerate(atoms):
        assert len(a) > 0 and np.isfinite(a.positions).all() and np.isfinite(a.cell.array).all()
        assert {'energy', 'forces'} <= set(a.calc.results)
        assert np.isfinite(a.get_potential_energy(apply_constraint=False))
        forces = a.get_forces(apply_constraint=False)
        assert forces.shape == (len(a), 3) and np.isfinite(forces).all()
        rows.append({'frame_id': f'{split}_{i:04d}', 'split': split,
                     'pkl_index_0based': i, 'natoms': len(a), 'geometry_sha256': geometry_hash(a)})
    all_rows = records + rows
    if len({r['geometry_sha256'] for r in all_rows}) != len(all_rows):
        raise ValueError('Duplicate geometries detected within or across the supplied splits')
    records.extend(rows)
    manifest['files'][path.name]['frames'] = len(atoms)
    return atoms, rows


def initialize_manifest(data):
    manifest = json.loads((data / 'dataset_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('label_type') != 'DFT':
        raise ValueError('dataset_manifest.json must describe the DFT reference dataset')
    for split in ('train', 'validation', 'test'):
        name = split + '.pkl'
        expected = manifest['files'][name]['expected_frames']
        if not isinstance(expected, int) or expected < 1:
            raise ValueError(f'Invalid expected_frames for {name}')
        path = data / name
        manifest['files'][name] = {'expected_frames': expected, 'sha256': sha256(path),
                                  'bytes': path.stat().st_size}
    manifest['status'] = 'INPUT_FILES_INDEXED'
    return manifest


def write_dataset_index(output, manifest, records):
    indexed = sorted({r['split'] for r in records})
    manifest['indexed_splits'] = indexed
    manifest['status'] = 'COMPLETE' if len(indexed) == 3 else 'TRAINING_SPLITS_INDEXED'
    dump(output / 'dataset_manifest.json', manifest)
    with (output / 'frame_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_id', 'split', 'pkl_index_0based',
                                               'natoms', 'geometry_sha256'])
        writer.writeheader()
        writer.writerows(records)


class RecordedPotential(Potential):
    """Only add progress output; keep MatterSim's optimizer/loss/training loop."""
    def train_one_epoch(self, *args, **kwargs):
        started = time.monotonic()
        metric = super().train_one_epoch(*args, **kwargs)
        if hasattr(self, 'record_dir'):
            epoch = args[1]
            row = {'epoch_0based': epoch, 'epoch_1based': epoch + 1, 'mode': kwargs.get('mode', 'train'),
                   'loss': float(metric[0]), 'energy_MAE_eV_atom': float(metric[1]),
                   'force_component_MAE_eV_A': float(metric[2]),
                   'learning_rate': float(self.optimizer.param_groups[0]['lr']),
                   'epoch_phase_seconds': time.monotonic() - started,
                   'peak_gpu_memory_MiB': torch.cuda.max_memory_allocated() / 2**20}
            if not all(np.isfinite(row[k]) for k in ['loss', 'energy_MAE_eV_atom', 'force_component_MAE_eV_A']):
                raise ValueError('Nonfinite training/validation metric')
            path = self.record_dir / 'learning_curve.csv'
            new = not path.exists()
            with path.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            dump(self.record_dir / 'progress.json', dict(status='TRAINING', **row))
            print('EPOCH_PROGRESS', json.dumps(row), flush=True)
        return metric


def summarize_errors(energy_errors, force_errors):
    e, f = np.asarray(energy_errors), np.concatenate(force_errors)
    assert np.isfinite(e).all() and np.isfinite(f).all()
    return {'frames': len(e), 'force_components': len(f),
            'energy_MAE_eV_atom': float(np.mean(np.abs(e))),
            'energy_MAE_meV_atom': float(np.mean(np.abs(e)) * 1000),
            'energy_RMSE_eV_atom': float(np.sqrt(np.mean(e * e))),
            'force_component_MAE_eV_A': float(np.mean(np.abs(f))),
            'force_component_RMSE_eV_A': float(np.sqrt(np.mean(f * f)))}


def evaluate(checkpoint, atoms_list, records, output_stem=None):
    calc = MatterSimCalculator(load_path=str(checkpoint), device='cuda', compute_stress=False, load_training_state=False)
    energy_errors, force_errors, predictions = [], [], []
    force_ref, force_pred = [], []
    for original, record in zip(atoms_list, records):
        ref_e, ref_f = original.get_potential_energy(apply_constraint=False), original.get_forces(apply_constraint=False)
        a = original.copy()
        a.calc = calc
        pred_e, pred_f = float(a.get_potential_energy(apply_constraint=False)), np.asarray(a.get_forces(apply_constraint=False))
        ee, fe = (pred_e - ref_e) / len(a), (pred_f - ref_f).ravel()
        energy_errors.append(ee)
        force_errors.append(fe)
        predictions.append({'frame_id': record['frame_id'], 'split': record['split'],
                            'pkl_index_0based': record['pkl_index_0based'],
                            'natoms': len(a), 'reference_energy_eV': float(ref_e), 'predicted_energy_eV': pred_e,
                            'energy_error_eV_atom': ee, 'force_component_MAE_eV_A': float(np.mean(np.abs(fe)))})
        force_ref.append(ref_f)
        force_pred.append(pred_f)
    metrics = summarize_errors(energy_errors, force_errors)
    if output_stem is not None:
        dump(str(output_stem) + '_metrics.json', metrics)
        with Path(str(output_stem) + '_predictions.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        np.savez_compressed(str(output_stem) + '_forces.npz', reference_eV_A=np.concatenate(force_ref),
                            predicted_eV_A=np.concatenate(force_pred),
                            atom_counts=np.array([len(a) for a in atoms_list]),
                            frame_ids=np.array([r['frame_id'] for r in records]))
    del calc
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.55)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.data, args.output, args.checkpoint = args.data.resolve(), args.output.resolve(), args.checkpoint.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.smoke and (args.output / 'run_config.json').exists():
        raise FileExistsError('Existing run: inspect before resuming or replacing')
    assert torch.cuda.is_available(), 'CUDA is required for this training run'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # Record the current files; test labels are loaded only after checkpoint selection.
    manifest = initialize_manifest(args.data)
    records = []
    train, _ = load_split(args.data, 'train', manifest, records)
    valid, valid_records = load_split(args.data, 'validation', manifest, records)
    write_dataset_index(args.output, manifest, records)
    config = dict(device='cuda', cutoff=5.0, threebody_cutoff=4.0, epochs=args.epochs,
                  batch_size=args.batch_size, lr=1e-4, step_size=10, gamma=0.95,
                  include_energy=True, include_forces=True, include_stresses=False,
                  force_loss_ratio=1.0, stress_loss_ratio=0.0, early_stop_patience=10,
                  seed=42, re_normalize=False, save_checkpoint=True, ckpt_interval=10,
                  save_path=str(args.output / 'checkpoints'), metric_name='val_loss')
    if args.smoke:
        train = [train[i] for i in sorted({0, min(1, len(train) - 1), max(0, len(train) - 2), len(train) - 1})]
        valid = valid[:1]
    loaders = {}
    for split, atoms in [('train', train), ('validation', valid)]:
        loaders[split] = build_dataloader(atoms, [a.get_potential_energy(apply_constraint=False) for a in atoms],
                                        [a.get_forces(apply_constraint=False) for a in atoms], None,
                                        shuffle=(split == 'train'), pin_memory=True,
                                        num_workers=0, **config)
        assert len(loaders[split].dataset) == len(atoms), 'Graph conversion dropped a structure'
    potential = RecordedPotential.from_checkpoint(load_path=str(args.checkpoint), load_training_state=False, **config)
    potential.model = potential.model.to('cuda')
    if args.smoke:
        start = time.monotonic()
        metric = potential.train_one_epoch(loaders['train'], 0, torch.nn.HuberLoss(delta=0.01),
                                           True, True, False, 1.0, 0.0, None, mode='train')
        assert np.isfinite(metric).all()
        report = {'status': 'SMOKE_PASSED', 'batch_size': args.batch_size, 'metric': list(map(float, metric)),
                  'seconds': time.monotonic() - start, 'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
                  'peak_reserved_MiB': torch.cuda.max_memory_reserved() / 2**20,
                  'checkpoint_sha256': sha256(args.checkpoint)}
        dump(args.output / f'smoke_batch{args.batch_size}.json', report)
        print('SMOKE_RESULT', json.dumps(report), flush=True)
        return

    potential.record_dir = args.output
    config_report = dict(config, huber_delta=0.01, pretrained_checkpoint=str(args.checkpoint),
                         pretrained_sha256=sha256(args.checkpoint), input_specification_sha256=sha256(args.data / 'dataset_manifest.json'),
                         dataset_files=manifest['files'], label_source=manifest['label_source'],
                         gpu_memory_fraction=args.gpu_memory_fraction,
                         versions={n: importlib.metadata.version(n) for n in ['torch','mattersim','ase','numpy']},
                         test_use='Held out until validation-based model selection is complete')
    dump(args.output / 'run_config.json', config_report)
    # Baseline uses validation only. Release the training potential first to keep memory bounded.
    del potential
    gc.collect()
    torch.cuda.empty_cache()
    baseline = evaluate(args.checkpoint, valid, valid_records, args.output / 'baseline_validation')
    dump(args.output / 'progress.json', {'status': 'BASELINE_COMPLETE', 'baseline_validation': baseline})
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    potential = RecordedPotential.from_checkpoint(load_path=str(args.checkpoint), load_training_state=False, **config)
    potential.record_dir = args.output
    potential.description = 'Fine-tuned on VASP DFT total energies and atomic forces.'
    start = time.monotonic()
    potential.train_model(loaders['train'], loaders['validation'], loss=torch.nn.HuberLoss(delta=0.01), wandb=None, **config)
    epochs_completed = potential.last_epoch + 1
    best_path = args.output / 'checkpoints/best_model.pth'
    assert best_path.is_file()
    del potential, loaders
    gc.collect()
    torch.cuda.empty_cache()
    best = torch.load(best_path, map_location='cpu', weights_only=True)
    selected_epoch = int(best['last_epoch']) + 1
    export = args.output / 'export'
    export.mkdir()
    slim = {k: best[k] for k in ['model_name','model','model_args','last_epoch','validation_metrics','description']}
    export_model = export / 'best_model.pth'
    torch.save(slim, export_model)
    del best, slim
    gc.collect()
    validation_metrics = evaluate(export_model, valid, valid_records, args.output / 'best_validation')
    test, test_records = load_split(args.data, 'test', manifest, records)
    write_dataset_index(args.output, manifest, records)
    test_metrics = evaluate(export_model, test, test_records, args.output / 'test')
    result = {'status': 'TRAINING_AND_TEST_COMPLETE', 'epochs_completed': epochs_completed,
              'selected_epoch_1based': selected_epoch, 'selection_metric': 'validation Huber loss',
              'baseline_validation': baseline, 'best_validation': validation_metrics, 'test': test_metrics,
              'label_source': manifest['label_source'], 'elapsed_training_and_evaluation_seconds': time.monotonic() - start,
              'best_checkpoint_sha256': sha256(best_path), 'export_model_sha256': sha256(export_model),
              'export_model_bytes': export_model.stat().st_size,
              'dataset_manifest_sha256': sha256(args.output / 'dataset_manifest.json'),
              'test_scope': 'Held-out test configurations; energy and force errors relative to DFT labels.'}
    dump(args.output / 'training_result.json', result)
    dump(args.output / 'progress.json', result)
    print('TRAINING_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
