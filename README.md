# Spatial Augmentation Analysis for Long-Term MTS Forecasting

Official implementation of **"How Much Does the Spatial Module Really Help? A Systematic Analysis for Long-Term MTS Forecasting"**

[![SAICSIT 2026](https://img.shields.io/badge/SAICSIT-2026-blue.svg)](https://saicsit.ac.za/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Paper: H. T. Moges and D. Moodley — SAICSIT 2026, CCIS, Springer
Contact: ht.moges@gmail.com  
Homepage: https://htmoges.github.io

## Get Started

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Download Data
Datasets from [Time-Series-Library](https://github.com/thuml/Time-Series-Library). Place in `./data/`:
```
data/
├── electricity.csv
├── traffic.csv
└── weather.csv
```

### 3. Reproduce Results
```bash
bash scripts/run_linear.sh       # Table 4 — DLinear / NLinear / RLinear
bash scripts/run_patchtst.sh     # Table 5 — PatchTST
bash scripts/run_horizon.sh      # Table 6 — horizon sweep
```

Or run a single experiment:
```bash
python -m gdf.train --config configs/linear/dlinear_electricity.yaml --seeds 42 1 2 3
```

## Evaluation Configurations (TTS Design)

The study uses four configurations formed by selectively enabling or disabling the prediction head (PH), input skip connection (R), and spatial module (S) within a shared TTS architecture. This isolates each component's contribution independently of the others:

| Config   | Pred. Head | Input Residual | Spatial |
|----------|:---:|:---:|:---:|
| L        | ✗  | ✗  | ✗  |
| L+PH     | ✓  | ✗  | ✗  |
| L+PH+R   | ✓  | ✓  | ✗  |
| L+PH+R+S | ✓  | ✓  | ✓  |

Same protocol for PatchTST: Base / +R / +R+S / +S, evaluating spatial behaviour as a function of the residual connection.

## Citation
```bibtex
@inproceedings{moges2026spatial,
  title={How Much Does the Spatial Module Really Help? A Systematic Analysis for Long-Term MTS Forecasting},
  author={Moges, H.T. and Moodley, D.},
  booktitle={Proceedings of SAICSIT 2026, CCIS, Springer},
  year={2026}
}
```

## Acknowledgements

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library) — benchmark datasets and baselines
- [Lite-STGNN](https://github.com/htmoges/Lite-STGNN) — STGNN modeling framework

## License

MIT License
