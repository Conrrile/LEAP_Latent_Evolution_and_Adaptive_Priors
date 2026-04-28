# LEAP — Quick Start

## 1. How to run (quick examples)

This repository uses `LEAP_Extension_main.py` as the main entry point. Use the `--model_name` flag to select either `LEAP` or `LEAP_Extension`.

Run commands from the repository root. Use `--device` to force CPU/GPU if needed.

### 1.1 Running `LEAP` (basic)

- Pretraining:

```powershell
python LEAP_Extension_main.py --mode pretrain --model_name LEAP --dataset HAR --output_root ./extension_experiments --experiment_name myexp
```
```powershell
python LEAP_Extension_main.py --mode pretrain --model_name LEAP --dataset HAR --augment 1 --output_root ./extension_experiments --experiment_name myexp
```
```powershell
python LEAP_Extension_main.py --mode pretrain --model_name LEAP --dataset HAR --augment 1 --augment_both_sides 1 --output_root ./extension_experiments --experiment_name myexp
```
- Downstream finetuning (linear probe):

```powershell
python LEAP_Extension_main.py --mode finetune --model_name LEAP --dataset HAR --load_path path\to\pretrained.pth --finetune_strategy probe --label_ratio 0.01 --output_root ./extension_experiments --experiment_name myexp
```

Notes:
- `--mode`: `pretrain` or `finetune`.
- `--model_name`: `LEAP` or `LEAP_Extension`.
- `--load_path`: path to pretrained checkpoint (required for probe mode).

### 1.2 Running `LEAP_Extension` (basic)

- Pretraining with common LEAP options:

```powershell
python LEAP_Extension_main.py --mode pretrain --model_name LEAP_Extension --dataset HAR --leap_ode_dynamics linear --leap_ode_solver rk4 --leap_ode_step_size 0.1 --leap_horizon 5 --output_root ./extension_experiments --experiment_name myext
```
```powershell
python LEAP_Extension_main.py --mode pretrain --model_name LEAP_Extension --dataset HAR --augment 1 --output_root ./extension_experiments --experiment_name myexp
```
```powershell
python LEAP_Extension_main.py --mode pretrain --model_name LEAP_Extension --dataset HAR --augment 1 --augment_both_sides 1 --output_root ./extension_experiments --experiment_name myexp
```
- Downstream finetuning example (using pretrained checkpoint):

```powershell
python LEAP_Extension_main.py --mode finetune --model_name LEAP_Extension --dataset HAR --load_path path\to\pretrained.pth \\
  --finetune_strategy probe --label_ratio 0.01 --output_root ./extension_experiments --experiment_name myext
```

Common extra options:
- `--leap_ode_solver`: `euler` / `rk2` / `rk4` / `linear_map` (`linear_map` uses matrix exponential).
- `--leap_ode_dynamics`: `linear` / `mlp`.
- `--leap_ode_step_size`: ODE step size (e.g. `0.1`).
- `--leap_temperature`: temperature for LEAP alignment loss (optional).
- `--augment`: applies only to `pretrain` (`--augment 1` enables augmentation on only single view).
- `--augment_both_sides`: enable two-sided augmentation.

## 2. Evironment:

- Python 3.9.x (example: 3.9.2)
- PyTorch 2.8.0+cu128 (CUDA 12.8 build) — verify with:


```powershell
pip install -r requirements.txt
```

## 3. Datasets
Download datasets from the links below and place them in the `datasets/` directory. The code will automatically look for the datasets in `datasets/` and preprocess them if needed.
[Download Dataset](https://zenodo.org/records/18970358?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY2YmI1YzgzLThlNjAtNGJhNC1iN2MzLTRjOTQ0NDI5NGI2YiIsImRhdGEiOnt9LCJyYW5kb20iOiI5N2ZiNzk4NmY3NWQyZGZjZGRjOTEzY2FlOWFiYTRjMyJ9.dHHLYVlQJPH9li4dv5DBFV3vJZL-_11g3qtUaSxtjc2UrVkwzyRENnuBSWKB_R8NN3QYKHIl640ZMJUVwdJB6Q)


