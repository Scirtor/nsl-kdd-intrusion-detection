"""Dataset schema, verified downloads and feature-only input validation."""
from pathlib import Path
import hashlib
import json
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = '''duration protocol_type service flag src_bytes dst_bytes land
wrong_fragment urgent hot num_failed_logins logged_in num_compromised root_shell
su_attempted num_root num_file_creations num_shells num_access_files
num_outbound_cmds is_host_login is_guest_login count srv_count serror_rate
srv_serror_rate rerror_rate srv_rerror_rate same_srv_rate diff_srv_rate
srv_diff_host_rate dst_host_count dst_host_srv_count dst_host_same_srv_rate
dst_host_diff_srv_rate dst_host_same_src_port_rate dst_host_srv_diff_host_rate
dst_host_serror_rate dst_host_srv_serror_rate dst_host_rerror_rate
dst_host_srv_rerror_rate'''.split()
CATEGORICAL = ['protocol_type', 'service', 'flag']
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
COLUMNS = FEATURES + ['attack_type', 'difficulty']


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(data_dir):
    """Download immutable mirror revisions, checking even existing local files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'data-manifest.json').read_text())
    for name, info in manifest.items():
        target = data_dir / name
        if not target.exists():
            temporary = target.with_suffix('.download')
            try:
                urllib.request.urlretrieve(info['url'], temporary)
                if checksum(temporary) != info['sha256']:
                    raise ValueError(f'Checksum mismatch: {name}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if checksum(target) != info['sha256']:
            raise ValueError(f'Checksum mismatch: {name}; restore the original dataset file')


def validate_features(frame):
    """Select exactly 41 predictors; labels and difficulty never enter a model."""
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not supported')
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError('Missing feature columns: ' + ', '.join(missing))
    result = frame.loc[:, FEATURES].copy()
    for col in NUMERIC:
        result[col] = pd.to_numeric(result[col], errors='raise').astype(float)
        if np.isinf(result[col]).any():
            raise ValueError(f'Infinite values in {col}')
        if (result[col] < 0).any():
            raise ValueError(f'Negative values in {col}')
    for col in CATEGORICAL:
        result[col] = result[col].fillna('__missing__').astype(str)
    return result


def load_dataset(path):
    raw = pd.read_csv(path, header=None)
    if len(raw.columns) != len(COLUMNS):
        raise ValueError(f'Expected {len(COLUMNS)} columns, got {len(raw.columns)}')
    raw.columns = COLUMNS
    if raw.empty or raw.attack_type.isna().any():
        raise ValueError('Dataset is empty or contains missing labels')
    raw['attack_type'] = raw.attack_type.astype(str).str.strip().str.rstrip('.')
    return validate_features(raw), (raw.attack_type != 'normal').astype(int), raw.attack_type


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    download(args.data_dir)
    print('Both dataset files are present and their SHA-256 checksums match.')
