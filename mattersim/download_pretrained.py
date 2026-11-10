"""Download the exact public MatterSim-5M checkpoint used for fine-tuning."""
import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = 'https://raw.githubusercontent.com/microsoft/mattersim/main/pretrained_models/mattersim-v1.0.0-5M.pth'
SHA256 = 'e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'models/mattersim-v1.0.0-5M.pth')
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        if hashlib.sha256(a.output.read_bytes()).hexdigest() != SHA256:
            raise ValueError('Existing file differs from the recorded checkpoint; it was not overwritten')
        print('Checkpoint already present and verified')
        return
    part = a.output.with_suffix('.pth.part')
    urllib.request.urlretrieve(URL, part)
    if hashlib.sha256(part.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Downloaded checkpoint hash does not match the training record')
    part.replace(a.output)
    print('Downloaded and verified', a.output)


if __name__ == '__main__':
    main()
