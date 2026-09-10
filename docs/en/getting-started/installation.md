# Installation

```bash
pip install visualmetrics
```

The base install is deliberately light - NumPy and SciPy - because every lab
runs its own numerics. A regression, a HAC covariance matrix or a Monte Carlo
size study all work without a heavier scientific stack.

## Extras

```bash
pip install "visualmetrics[gui]"    # the interactive application
pip install "visualmetrics[all]"    # everything
```

| Extra | Enables |
|---|---|
| `gui` | The NiceGUI application |
| `viz` | Plotly and Matplotlib rendering |
| `symbolic` | SymPy derivations |
| `econometrics` | statsmodels, linearmodels, arch cross-checks |
| `causal` | networkx DAG tooling |
| `ai` | scikit-learn, SHAP |
| `proofs` | Manim, for exported proof animations |
| `notebook` | ipywidgets controls |
| `data` | openpyxl, pyarrow loaders |
| `export` | kaleido, for static image export |

A missing extra disables only what needs it. The error names the package and
the exact install command rather than failing with an import error.

## Checking the environment

```bash
visualmetrics doctor
```

This reports the Python version, every optional backend with its status and
version, which extra provides it, the available languages, the catalogue counts
and whether the config and cache directories are writable.

A package that is installed but *unusable* - a NumPy ABI mismatch, for instance
- is reported as unusable rather than as missing, because the two problems have
different fixes.

## From source

```bash
git clone https://github.com/merwanroudane/visualmetrics
cd visualmetrics
pip install -e ".[dev,gui]"
pytest -m "not slow"
```

## Supported versions

Python 3.11, 3.12 and 3.13, on Linux, macOS and Windows.
