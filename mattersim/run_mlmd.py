"""ASE fixed-cell NVT MD using the supplied fine-tuned MatterSim model."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs
from mattersim.forcefield import MatterSimCalculator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--structure', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--temperature', type=float, default=1000)
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--timestep-fs', type=float, default=2)
    p.add_argument('--repeat', nargs=3, type=int, default=[1,2,2])
    p.add_argument('--seed', type=int, default=None)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Choose a new trajectory output path')
    if not a.model.is_file():
        raise FileNotFoundError(a.model)
    if a.steps<1 or a.temperature<=0 or a.timestep_fs<=0 or min(a.repeat)<1:
        raise ValueError('Steps, temperature, timestep and repeat must be positive')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    atoms = read(a.structure, format='vasp') * tuple(a.repeat)
    atoms.set_pbc(True)
    atoms.calc = MatterSimCalculator(load_path=str(a.model), device=device,
                                    load_training_state=False, compute_stress=False)
    rng = np.random.RandomState(a.seed) if a.seed is not None else np.random
    MaxwellBoltzmannDistribution(atoms, temperature_K=a.temperature, rng=rng)
    timestep = a.timestep_fs*fs
    dyn = NoseHooverChainNVT(atoms=atoms, timestep=timestep,
                            temperature_K=a.temperature, tdamp=100*timestep)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with Trajectory(str(a.output), 'w', atoms) as trajectory:
        dyn.attach(trajectory.write, interval=1)
        dyn.run(steps=a.steps)
    result = {'structure': str(a.structure), 'model': str(a.model), 'natoms': len(atoms),
              'temperature_K': a.temperature, 'steps': a.steps, 'timestep_fs': a.timestep_fs,
              'duration_ps': a.steps*a.timestep_fs/1000, 'repeat': a.repeat,
              'thermostat': 'NoseHooverChainNVT', 'tdamp_fs': 100*a.timestep_fs,
              'seed': a.seed, 'device': device, 'wall_seconds': time.perf_counter()-started}
    a.output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
