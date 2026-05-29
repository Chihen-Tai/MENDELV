# Optional MLIP Environment Setup

MENDELV keeps MLIP dependencies optional. Normal installs do not require ASE,
MACE, or torch for core rule-based functionality.

## Install MLIP Extras

Using pip:

```bash
pip install -e ".[mlip]"
```

Using uv:

```bash
uv pip install -e ".[mlip]"
```

The helper script is cautious by default and does not install anything unless
`--install` is passed:

```bash
python scripts/setup_mlip_env.py
python scripts/setup_mlip_env.py --install --pip
python scripts/setup_mlip_env.py --install --uv
```

## Check The Environment

```bash
python scripts/check_mlip_env.py \
  --output reports/mlip_env_report.json
```

The check exits successfully when core `import mendel` works, even if optional
MLIP dependencies are missing. Missing optional dependencies are reported with:

```text
Install MLIP extras with: pip install -e '.[mlip]'
```

## Platform Notes

On macOS, MPS availability depends on the installed torch and MACE support.
CPU is acceptable for tiny smoke tests.

On NVIDIA systems, CUDA is preferred for larger MACE-OFF runs.

## Scope Warning

The MACE-OFF benchmark path is optional and experimental. This setup does not
train MLIP, does not download QO2Mol data, does not run DFT, and does not run
NEB, IRC, MD, transition-state search, or barrier prediction.
