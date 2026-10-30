"""Executed inside the profile image; operator-side config contains no hidden data."""
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess

cfg = json.loads(Path(__file__).with_name('oracle.json').read_text())
sub = Path(os.environ['ABENCH_SUBMISSION_DIR'])
shutil.copytree(Path(__file__).with_name('example_submission'), sub, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__', 'checkpoint'))
if cfg['mode'] == 'checkpoint':
    text = (sub / 'train.py').read_text()
    args = ['python3', 'train.py', '--data', os.environ['ABENCH_DATA_DIR'], '--out', 'checkpoint']
    cal = cfg.get('calibration') or {}
    if cal.get('final_flag'):
        args += shlex.split(cal['final_flag'])
    elif '--include-dev' in text:
        args += ['--include-dev']
    elif '--final' in text:
        args += ['--final']
    if '--models' in text and os.environ.get('ABENCH_MODELS_DIR'):
        args += ['--models', os.environ['ABENCH_MODELS_DIR']]
    if '--device' in text:
        args += ['--device', os.environ['ABENCH_DEVICE']]
    args += shlex.split(cal.get('train_args') or '')
    subprocess.run(args, cwd=sub, check=True)
