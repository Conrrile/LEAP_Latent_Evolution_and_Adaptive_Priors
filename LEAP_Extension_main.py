import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import csv
import datetime

from config import Config
from data_factory import load_data
from models.LEAP_Extension import LEAP_Extension
from models.LEAP import LEAP
from models.heads import Linear_Classifier, MLP_Classifier
from utils import set_seed, save_checkpoint, load_state_dict_compatible, compute_metrics

# CSV logging (detailed experiment results)
def save_detailed_results(cfg, metrics, filename="all_results.csv"):
    """Save detailed experiment results to CSV, recording key flags."""
    save_dir = cfg.output_dir
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    # Fields to record
    row_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": getattr(cfg, 'experiment_name', 'default'),
        "dataset": cfg.dataset.replace("_MEAN", "").split("_ratio")[0],
        "model_name": cfg.model_name,
        "mode": getattr(cfg, 'mode', 'unknown'),
        "finetune_strategy": getattr(cfg, 'finetune_strategy', 'N/A'),
        "label_ratio": getattr(cfg, 'label_ratio', 1.0),
        "sensitivity_note": getattr(cfg, 'sensitivity_note', 'N/A'),
        # LEAP-specific
        "loss_type": getattr(cfg, 'leap_loss_type', 'N/A'),
        "backbone": getattr(cfg, 'leap_backbone_type', 'N/A'),
        "solver": getattr(cfg, 'leap_ode_solver', 'N/A'),
        "dynamics": getattr(cfg, 'leap_ode_dynamics', 'N/A'),
        "horizon": getattr(cfg, 'leap_horizon', 'N/A'),
        "step_size": getattr(cfg, 'leap_ode_step_size', 'N/A'),
        "noise_level": getattr(cfg, 'noise_level', 0.0),
        # training
        "seed": getattr(cfg, 'seed', 0),
        "batch_size": cfg.batch_size,
        "epochs": cfg.finetune_epochs if hasattr(cfg, 'finetune_epochs') else cfg.pretrain_epochs,
        "lr": cfg.probe_lr if hasattr(cfg, 'probe_lr') else cfg.lr,
        # temperature
        "temperature": getattr(cfg, 'leap_temperature', 'N/A'),
        # results
        "acc_mean": f"{metrics.get('acc', 0):.4f}",
        "acc_std": f"{getattr(cfg, 'std_acc', 0):.4f}",
        "f1_mean": f"{metrics.get('macro_f1', 0):.4f}",
        "f1_std": f"{getattr(cfg, 'std_f1', 0):.4f}"
    }

    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    print(f"[Info] Detailed results saved to: {os.path.abspath(file_path)}")


# Model factory: only support LEAP and LEAP_Extension
def get_model(cfg):
    model_name = cfg.model_name.lower()
    if 'leap_extension' in model_name:
        return LEAP_Extension(cfg).to(cfg.device)
    elif model_name == 'leap':
        return LEAP(cfg).to(cfg.device)
    else:
        raise ValueError(f"Unknown model: {model_name}. Only 'LEAP' and 'LEAP_Extension' are supported.")


# Pre-training routines
def train_one_epoch_pretrain(model, dataloader, optimizer, online_probe, probe_optimizer, cfg, epoch, alphas_list=None):
    model.train()
    online_probe.train()

    total_loss = 0
    total_probe_loss = 0
    correct = 0
    total = 0

    use_amp = getattr(cfg, 'use_amp', False)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    criterion_probe = nn.CrossEntropyLoss()

    # collect per-batch alphas for this epoch
    alphas_epoch = []
    for x1, x2, y in dataloader:
        x1, x2, y = x1.to(cfg.device), x2.to(cfg.device), y.to(cfg.device)

        if alphas_list is not None:
            with torch.no_grad():
                if hasattr(model, 'criterion') and hasattr(model.criterion, 'gate_net'):
                    try:
                        logits_gate = model.criterion.gate_net(x1)
                        freq_weight = torch.sigmoid(logits_gate).squeeze(-1).detach().cpu().numpy()
                        alphas_epoch.append(freq_weight)
                    except Exception:
                        pass

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            loss_ssl = model(x1, x2)

        scaler.scale(loss_ssl).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(cfg, 'clip_grad_norm', 5.0))
        scaler.step(optimizer)

        # Online probe
        probe_optimizer.zero_grad()
        with torch.no_grad():
            feat = model.extract_features(x1)
            feat = feat.detach()

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = online_probe(feat)
            loss_probe = criterion_probe(logits, y)

        scaler.scale(loss_probe).backward()
        scaler.step(probe_optimizer)
        scaler.update()

        if hasattr(model, 'update_moving_average'):
            model.update_moving_average()

        total_loss += loss_ssl.item()
        total_probe_loss += loss_probe.item()

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    if alphas_list is not None:
        if len(alphas_epoch) > 0:
            import numpy as _np
            alphas_list.append(_np.concatenate(alphas_epoch, axis=0))
        else:
            alphas_list.append(_np.array([]))

    return total_loss / len(dataloader), correct / total


def validate_online_probe(model, dataloader, online_probe, cfg):
    model.eval()
    online_probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x1, _, y in dataloader:
            x1, y = x1.to(cfg.device), y.to(cfg.device)
            feat = model.extract_features(x1)
            logits = online_probe(feat)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def pretrain(cfg):
    print(f"=== Starting Pre-training: {cfg.model_name} ===")
    print(f"Dataset: {cfg.dataset} | Loss: {getattr(cfg, 'leap_loss_type', 'N/A')} | Backbone: {getattr(cfg, 'leap_backbone_type', 'N/A')}")

    train_dl, val_dl, _ = load_data(cfg, mode='pretrain')

    model = get_model(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    online_probe = Linear_Classifier(model.n_features, cfg.n_class).to(cfg.device)
    probe_optimizer = optim.Adam(online_probe.parameters(), lr=1e-3)

    # build save path
    save_dir = os.path.join(cfg.output_dir, cfg.dataset, cfg.model_name, 'pretrain')
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Info] Checkpoints will be saved to: {save_dir}")

    best_acc = 0.0

    collect_alphas = getattr(cfg, 'save_alpha_path', None) is not None
    alphas_accum = [] if collect_alphas else None

    for epoch in range(1, cfg.pretrain_epochs + 1):
        loss, train_acc = train_one_epoch_pretrain(
            model, train_dl, optimizer, online_probe, probe_optimizer, cfg, epoch, alphas_list=alphas_accum
        )

        val_acc = validate_online_probe(model, val_dl, online_probe, cfg)

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.pretrain_epochs:
            print(f"Epoch {epoch}/{cfg.pretrain_epochs} | SSL Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                os.path.join(save_dir, 'model_best.pth'),
                model,
                epoch,
                optimizer
            )
            if epoch % 10 == 0 or epoch == 1 or epoch == cfg.pretrain_epochs:
                print(f"--> Saved Best Model (Acc: {best_acc:.4f})")

        save_checkpoint(
            os.path.join(save_dir, 'model_last.pth'),
            model,
            epoch,
            optimizer
        )
        torch.save(online_probe.state_dict(), os.path.join(save_dir, 'online_probe_last.pth'))

    print(f"Pre-training finished. Best Val Acc: {best_acc:.4f}")
    if collect_alphas and alphas_accum is not None:
        import numpy as _np
        out_path = getattr(cfg, 'save_alpha_path')
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        save_dict = {}
        means = []
        for i, arr in enumerate(alphas_accum):
            key = f'epoch_{i+1}'
            save_dict[key] = arr
            means.append(float(arr.mean()) if arr.size > 0 else float('nan'))

        npz_path = out_path.replace('.npy', '') + '_per_epoch.npz'
        _np.savez_compressed(npz_path, **save_dict)
        mean_path = out_path.replace('.npy', '') + '_mean_per_epoch.npy'
        _np.save(mean_path, _np.array(means))

        try:
            all_alpha = _np.concatenate([a for a in alphas_accum if a.size > 0], axis=0)
            _np.save(out_path, all_alpha)
        except Exception:
            _np.save(out_path, _np.array([]))

        print(f"[Info] Saved pretrain gating alpha per-epoch npz: {os.path.abspath(npz_path)}")
        print(f"[Info] Saved pretrain gating alpha mean per-epoch: {os.path.abspath(mean_path)}")
        print(f"[Info] Saved concatenated alpha array to: {os.path.abspath(out_path)} (shape={all_alpha.shape if 'all_alpha' in locals() else (0,)})")
    return best_acc


# Finetuning utilities
def test_classifier(model, classifier, loader, cfg):
    model.eval()
    classifier.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            feat = model.extract_features(x)
            out = classifier(feat)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            trues.extend(y.cpu().numpy())
    return compute_metrics(trues, preds)


def save_gating_alpha(model, dataloader, cfg, out_path):
    """Compute and save gating alpha (freq_weight) for all samples in dataloader."""
    model.eval()
    alphas = []
    device = cfg.device
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            if not hasattr(model, 'criterion') or not hasattr(model.criterion, 'gate_net'):
                raise RuntimeError('Model/criterion does not expose gate_net')
            logits = model.criterion.gate_net(x)
            freq_weight = torch.sigmoid(logits)
            alphas.append(freq_weight.squeeze(-1).detach().cpu().numpy())

    if len(alphas) == 0:
        raise RuntimeError('No samples found in dataloader to compute alphas')

    import numpy as _np
    all_alpha = _np.concatenate(alphas, axis=0)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    _np.save(out_path, all_alpha)
    print(f"[Info] Saved gating alpha array to: {os.path.abspath(out_path)} (shape={all_alpha.shape})")


def run_finetune_once(cfg, run_id):
    """Execute a single finetune/probe run."""
    mode_str = "Linear Probe" if cfg.finetune_strategy == 'probe' else "Full Fine-tuning"
    print(f"\n--- Run {run_id + 1} | {mode_str}: {cfg.model_name} (Ratio: {cfg.label_ratio}) ---")

    train_dl, val_dl, test_dl = load_data(cfg, mode='finetune', ratio=cfg.label_ratio)

    model = get_model(cfg)

    # enhanced model loading logic
    path = cfg.load_path
    if not path:
        auto_path = os.path.join(cfg.output_dir, cfg.dataset, cfg.model_name, 'pretrain', 'model_best.pth')
        print(f"[Info] 'load_path' not specified. Attempting to load from experiment dir: {auto_path}")
        path = auto_path

    if os.path.exists(path):
        print(f"[Info] Loading checkpoint from: {path}")
        try:
            state = torch.load(path, map_location=cfg.device)
            if isinstance(state, dict):
                if 'model_state_dict' in state:
                    load_state_dict_compatible(model, state['model_state_dict'])
                    print(f"[Info] Success: Loaded 'model_state_dict' (Epoch: {state.get('epoch', 'unknown')})")
                else:
                    try:
                        load_state_dict_compatible(model, state)
                        print("[Info] Success: Loaded checkpoint as raw state dict")
                    except Exception as e:
                        print(f"[Error] Failed to match keys. Training from scratch. Error: {e}")
            else:
                print("[Warning] Checkpoint is not a dict. Training from scratch.")
        except Exception as e:
            print(f"[Error] Failed to load file {path}: {e}")
            if cfg.finetune_strategy == 'probe':
                raise RuntimeError(f"Cannot Linear Probe without a valid pretrained model! Check path: {path}")
    else:
        print(f"[Warning] No checkpoint found at {path}")
        if cfg.finetune_strategy == 'probe':
            raise RuntimeError(f"Cannot Linear Probe: Pretrained model not found at {path}")
        print("[Info] Proceeding with random initialization (Training from scratch)...")

    # configure classifier and optimizer
    if cfg.finetune_strategy == 'probe':
        for param in model.parameters():
            param.requires_grad = False
        classifier = Linear_Classifier(model.n_features, cfg.n_class).to(cfg.device)
        optimizer = optim.AdamW(classifier.parameters(), lr=cfg.probe_lr, weight_decay=1e-4)
    else:
        for param in model.parameters():
            param.requires_grad = True
        classifier = MLP_Classifier(model.n_features, cfg.n_class).to(cfg.device)
        optimizer = optim.AdamW([
            {'params': model.parameters(), 'lr': cfg.finetune_lr},
            {'params': classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4)

    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model Stats] Total Params: {total_params:,} | Trainable: {trainable_params:,}")
    except:
        pass

    criterion = nn.CrossEntropyLoss()

    # set output directory for this run
    ratio_str = f"_ratio{cfg.label_ratio}" if cfg.label_ratio < 1.0 else ""
    run_dir = os.path.join(cfg.output_dir, cfg.dataset, cfg.model_name, f"{cfg.finetune_strategy}{ratio_str}", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    best_acc = 0
    best_model_state = None

    epoch_iter = range(1, cfg.finetune_epochs + 1)

    for epoch in epoch_iter:
        model.train() if cfg.finetune_strategy == 'finetune' else model.eval()
        classifier.train()

        epoch_loss = 0.0
        batch_count = 0

        for x, y in train_dl:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()

            # decide whether to compute gradients for backbone
            if cfg.finetune_strategy == 'probe':
                with torch.no_grad():
                    feat = model.extract_features(x)
            else:
                feat = model.extract_features(x)

            logits = classifier(feat)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        mean_loss = epoch_loss / batch_count if batch_count > 0 else 0.0

        val_metrics = test_classifier(model, classifier, val_dl, cfg)
        if epoch % 5 == 0 or epoch == 1 or epoch == cfg.finetune_epochs:
            print(f"Epoch {epoch}/{cfg.finetune_epochs} | Loss: {mean_loss:.4f} | Val Acc: {val_metrics['acc']:.4f}")

        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            best_model_state = {
                'encoder': model.state_dict(),
                'classifier': classifier.state_dict()
            }

    if best_model_state:
        model.load_state_dict(best_model_state['encoder'])
        classifier.load_state_dict(best_model_state['classifier'])

    test_metrics = test_classifier(model, classifier, test_dl, cfg)
    print(f"Run {run_id+1} Test Acc: {test_metrics['acc']:.4f}")

    return test_metrics


def finetune_main(cfg):
    """Main entry for downstream tasks: aggregate results and save CSV."""
    accuracies = []
    f1s = []

    base_seed = cfg.seed

    for r in range(cfg.repeat):
        cfg.seed = base_seed + r
        set_seed(cfg.seed)

        metrics = run_finetune_once(cfg, r)
        accuracies.append(metrics['acc'])
        f1s.append(metrics['macro_f1'])
        cfg.std_acc = 0.0
        cfg.std_f1 = 0.0
        save_detailed_results(cfg, metrics)

    print(f"\n{'='*60}")
    print(f"Summary: {cfg.model_name} on {cfg.dataset} (Loss: {getattr(cfg, 'leap_loss_type', 'N/A')})")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)

    print(f"Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Mean F1 : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"{'='*60}")

    cfg.std_acc = std_acc
    cfg.std_f1 = std_f1

    original_dataset = cfg.dataset
    cfg.dataset = f"{original_dataset}_MEAN_ratio{cfg.label_ratio}_runs{cfg.repeat}"
    save_detailed_results(cfg, {'acc': mean_acc, 'macro_f1': mean_f1})
    cfg.dataset = original_dataset
    if hasattr(cfg, 'std_acc'):
        delattr(cfg, 'std_acc')
    if hasattr(cfg, 'std_f1'):
        delattr(cfg, 'std_f1')


# Experiment runners
def run_experiment_ode_vs_linear(cfg):
    import copy
    print(f"[Experiment] Running ODE vs Matched Linear Baseline")
    results = {}
    orig_dyn = getattr(cfg, 'leap_ode_dynamics', 'linear')

    for dyn in (orig_dyn, 'linear_map'):
        c = copy.deepcopy(cfg)
        c.leap_ode_dynamics = dyn
        c.model_name = f"LEAP_Extension_{dyn}"
        c.output_dir = os.path.join(cfg.output_dir, 'ode_vs_linear')

        print(f"[Experiment] Pretraining with dynamics={dyn}")
        best_val = pretrain(c)
        results[dyn] = best_val

        pretrain_ckpt = os.path.join(cfg.output_dir, c.dataset, c.model_name, 'pretrain', 'model_best.pth')
        if os.path.exists(pretrain_ckpt):
            c.load_path = pretrain_ckpt
            print(f"[Experiment] Running downstream evaluation for {dyn}")
            finetune_main(c)
        else:
            print(f"[Warning] Pretrained checkpoint not found at {pretrain_ckpt}; skipping downstream eval")

    out_csv = os.path.join(cfg.output_dir, 'ode_vs_linear_summary.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dynamics', 'best_val_acc'])
        for k, v in results.items():
            writer.writerow([k, v])
    print(f"[Experiment] ODE vs Linear results saved to {out_csv}")


def run_experiment_irregular_sampling(cfg):
    import copy
    print(f"[Experiment] Running Irregular Sampling / Time-Reversal tests: variant={getattr(cfg,'irregular_variant',None)}")
    c = copy.deepcopy(cfg)
    c.model_name = f"LEAP_Extension_{c.leap_ode_dynamics}"
    c.output_dir = os.path.join(cfg.output_dir, 'irregular_sampling')
    best_val = pretrain(c)

    v = getattr(cfg, 'irregular_variant', 'subsample')
    if v == 'time_reverse':
        c.irregular_test_mode = 'reverse'
    elif v == 'subsample':
        c.irregular_test_mode = 'irregular'
    elif v == 'missing_chunk':
        c.irregular_test_mode = 'missing'
    else:
        c.irregular_test_mode = None

    c.irregular_keep_prob = getattr(cfg, 'irregular_keep_prob', 0.7)
    c.missing_chunk_frac = getattr(cfg, 'missing_chunk_frac', 0.2)

    pretrain_ckpt = os.path.join(cfg.output_dir, c.dataset, c.model_name, 'pretrain', 'model_best.pth')
    if os.path.exists(pretrain_ckpt):
        c.load_path = pretrain_ckpt
        print(f"[Experiment] Running downstream evaluation under {c.irregular_test_mode}")
        finetune_main(c)
    else:
        print(f"[Warning] Pretrained checkpoint not found at {pretrain_ckpt}; skipping downstream eval")


def run_experiment_step_size_sensitivity(cfg):
    import copy
    print(f"[Experiment] Running ODE Step Size Sensitivity")
    c = copy.deepcopy(cfg)
    c.model_name = f"LEAP_Extension_{c.leap_ode_dynamics}"
    c.output_dir = os.path.join(cfg.output_dir, 'step_size_sensitivity')

    pretrain_ckpt = os.path.join(cfg.output_dir, c.dataset, c.model_name, 'pretrain', 'model_best.pth')
    if not os.path.exists(pretrain_ckpt):
        print(f"[Experiment] Pretrained checkpoint not found, running pretrain first")
        pretrain(c)

    train_dl, val_dl, _ = load_data(c, mode='pretrain')

    model = get_model(c)
    if os.path.exists(pretrain_ckpt):
        try:
            state = torch.load(pretrain_ckpt, map_location=c.device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                load_state_dict_compatible(model, state['model_state_dict'])
            else:
                load_state_dict_compatible(model, state)
        except Exception as e:
            print(f"[Warning] Failed to load pretrained model: {e}")

    model.to(c.device)
    model.eval()

    dt_list = getattr(cfg, 'dt_values', None)
    if dt_list is None:
        dt_list = [0.01, 0.05, 0.1, 0.5, 1.0]

    results = []
    for dt in dt_list:
        if hasattr(model, 'evolver'):
            if getattr(model.evolver, 'use_linear_map', False):
                model.evolver.dt_scalar = dt
            else:
                with torch.no_grad():
                    model.evolver.dt_gate.data.fill_(dt)

        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for x1, x2, y in val_dl:
                x1 = x1.to(c.device)
                z0 = model.encoder(x1)
                traj = model.evolver(z0)
                loss = model.criterion(x1, traj)
                total_loss += float(loss.cpu().item())
                count += 1
        mean_loss = total_loss / max(1, count)
        results.append((dt, mean_loss))
        print(f"dt={dt:.5f} | Mean Align Loss: {mean_loss:.6f}")

    out_csv = os.path.join(cfg.output_dir, 'step_size_sensitivity.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dt', 'mean_alignment_loss'])
        for dt, loss in results:
            writer.writerow([dt, loss])
    print(f"[Experiment] Step size sensitivity results saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LEAP Extension Training Script')

    # Basic options
    parser.add_argument('--dataset', type=str, default='HAR', help='Dataset name')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune'], required=True)
    parser.add_argument('--experiment_name', type=str, default='default', help='Name of the experiment group')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)

    # Model settings
    parser.add_argument('--model_name', type=str, default='LEAP_Extension')
    parser.add_argument('--leap_model_size', type=str, default='base', choices=['small', 'base', 'large'])

    # LEAP extension flags
    parser.add_argument('--leap_loss_type', type=str, default='kl', choices=['kl', 'mse', 'mae', 'huber', 'js'], help='Loss function type')
    parser.add_argument('--leap_ode_solver', type=str, default='rk4', choices=['euler', 'rk2', 'rk4', 'linear_map'],
                        help='ODE integrator for latent evolver; linear_map invokes matrix exponential (exact)')
    parser.add_argument('--leap_ode_dynamics', type=str, default='linear', choices=['linear', 'mlp'],
                        help='Latent dynamics (linear or mlp).')
    parser.add_argument('--leap_ode_step_size', type=float, default=0.1)
    parser.add_argument('--leap_horizon', type=int, default=5)
    parser.add_argument('--noise_level', type=float, default=0.0)
    parser.add_argument('--leap_temperature', type=float, default=None, help='Temperature for LEAP alignment loss')

    # Test-time irregularity / robustness
    parser.add_argument('--irregular_test_mode', type=str, choices=['none','irregular','reverse','missing'], default='none', help='Apply test-time irregular transform')
    parser.add_argument('--irregular_keep_prob', type=float, default=0.7, help='Fraction of points kept for irregular resample')
    parser.add_argument('--missing_chunk_frac', type=float, default=0.2, help='Fraction of time length to zero out for missing-chunk test')
    parser.add_argument('--extension_experiment', type=str, default=None, choices=['ode_vs_linear','irregular_sampling','step_size_sensitivity'], help='Run special experiments (pretrain+eval flows)')
    parser.add_argument('--irregular_variant', type=str, default='subsample', choices=['time_reverse','subsample','missing_chunk'], help='Irregular test variant')
    parser.add_argument('--dt_values', type=str, default=None, help='Comma separated dt values for step-size sensitivity (e.g. 0.01,0.05,0.1).')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None, help='Pretrain LR')
    parser.add_argument('--probe_lr', type=float, default=1e-3, help='Probe/Classifier LR')
    parser.add_argument('--finetune_lr', type=float, default=3e-5, help='Backbone LR during finetuning')

    # Augmentation
    parser.add_argument('--augment', type=int, choices=[0,1], default=None,
                        help='Toggle data augmentation (0 = off, 1 = on). Overrides augment_jitter and augment_scaling.')
    parser.add_argument('--augment_both_sides', type=int, choices=[0,1], default=None,
                        help='Use both-sided augmentation (1) or one-sided (0).')

    # Downstream settings
    parser.add_argument('--finetune_strategy', type=str, choices=['probe', 'full'], default='probe')
    parser.add_argument('--label_ratio', type=float, default=1.0)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--load_path', type=str, default=None, help='Explicit path to pretrained model')

    # Other
    parser.add_argument('--resample', type=int, default=None)
    parser.add_argument('--save_alpha_path', type=str, default=None, help='If set, load checkpoint and save gating alpha (.npy) for test set to this path')
    parser.add_argument('--alpha_split', type=str, default='val', choices=['train','val','all'], help='Which pretrain split to use when saving alphas')
    parser.add_argument('--sensitivity_note', type=str, default='N/A', help='Note about the sensitivity experiment being performed')
    parser.add_argument('--output_root', type=str, default='./extension_experiments', help='Root directory for saving results')

    # Additional flags
    parser.add_argument('--leap_entropy_weight', type=float, default=None)
    parser.add_argument('--leap_cross_weight', type=float, default=None)
    parser.add_argument('--leap_latent_dim', type=int, default=None)
    parser.add_argument('--leap_ema_decay', type=float, default=None)

    args = parser.parse_args()

    cfg = Config(args.dataset)

    # unify output directory
    cfg.output_dir = os.path.join(args.output_root, args.experiment_name)

    cfg.update(args)

    cfg.sensitivity_note = getattr(args, 'sensitivity_note', 'N/A')

    if args.leap_ode_step_size is not None:
        cfg.leap_dt_init = args.leap_ode_step_size
        cfg.leap_ode_step_size = args.leap_ode_step_size

    if args.leap_horizon is not None:
        cfg.leap_horizon = args.leap_horizon
    cfg.leap_horizon = args.leap_horizon
    cfg.noise_level = args.noise_level

    if args.leap_entropy_weight is not None: cfg.leap_entropy_weight = args.leap_entropy_weight
    if args.leap_cross_weight is not None: cfg.leap_cross_weight = args.leap_cross_weight
    if args.leap_latent_dim is not None: cfg.leap_latent_dim = args.leap_latent_dim
    if args.leap_ema_decay is not None: cfg.leap_ema_decay = args.leap_ema_decay

    cfg.irregular_test_mode = args.irregular_test_mode
    cfg.irregular_keep_prob = args.irregular_keep_prob
    cfg.missing_chunk_frac = args.missing_chunk_frac
    cfg.leap_dropout = 0.1
    cfg.load_path = args.load_path
    cfg.leap_dt_init = args.leap_ode_step_size
    cfg.extension_experiment = args.extension_experiment
    cfg.irregular_variant = args.irregular_variant
    if args.dt_values:
        try:
            cfg.dt_values = [float(x.strip()) for x in args.dt_values.split(',') if x.strip()]
        except Exception:
            cfg.dt_values = None

    if getattr(args, 'leap_temperature', None) is not None:
        cfg.leap_temperature = args.leap_temperature

    cfg.pin_memory = torch.cuda.is_available()

    if cfg.model_name.lower() == 'leap_extension':
        config_parts = ['LEAP_Extension']
        if args.leap_loss_type != 'kl':
            config_parts.append(f"loss_{args.leap_loss_type}")
        if getattr(args, 'leap_backbone_type', cfg.leap_backbone_type) != 'shallow_cnn':
            config_parts.append(f"bk_{getattr(args, 'leap_backbone_type', cfg.leap_backbone_type)}")
        if args.leap_ode_solver != 'rk4':
            config_parts.append(f"sol_{args.leap_ode_solver}")
        if args.leap_horizon != 5:
            config_parts.append(f"hz_{args.leap_horizon}")
        config_parts.append(args.leap_ode_dynamics)
        cfg.model_name = '_'.join(config_parts)

    if getattr(args, 'augment', None) is not None:
        if args.mode == 'pretrain':
            aug_on = bool(args.augment)
            cfg.augment_jitter = aug_on
            cfg.augment_scaling = aug_on
            cfg.pretrain_augment = aug_on
        else:
            print("[Info] --augment applies only to pretrain mode; ignoring for finetune runs.")

    if getattr(args, 'augment_both_sides', None) is not None:
        cfg.augment_both_sides = bool(args.augment_both_sides)
        print(f"[Info] Augment both sides set to: {cfg.augment_both_sides}")

    set_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.epochs:
        if args.mode == 'pretrain': cfg.pretrain_epochs = args.epochs
        else: cfg.finetune_epochs = args.epochs

    print(f"Output Directory: {os.path.abspath(cfg.output_dir)}")

    if getattr(args, 'save_alpha_path', None) and args.mode != 'pretrain':
        print(f"[Info] Saving gating alpha to: {args.save_alpha_path} (using pretrain data, split={args.alpha_split})")
        train_dl, val_dl, _ = load_data(cfg, mode='pretrain')
        model = get_model(cfg)
        path = cfg.load_path
        if not path:
            auto_path = os.path.join(cfg.output_dir, cfg.dataset, cfg.model_name, 'pretrain', 'model_best.pth')
            path = auto_path
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location=cfg.device)
                if isinstance(state, dict):
                    if 'model_state_dict' in state:
                        load_state_dict_compatible(model, state['model_state_dict'])
                    else:
                        load_state_dict_compatible(model, state)
                else:
                    print(f"[Warning] Checkpoint format unexpected; attempted to load raw state dict")
            except Exception as e:
                print(f"[Error] Failed to load checkpoint {path}: {e}")
                raise
        else:
            raise RuntimeError(f"Pretrained checkpoint not found at {path}; cannot compute alphas.")

        model.to(cfg.device)
        if args.alpha_split == 'train':
            dl = train_dl
        elif args.alpha_split == 'val':
            dl = val_dl
        else:
            class ConcatLoader:
                def __init__(self, a, b):
                    self.a = a
                    self.b = b
                def __iter__(self):
                    for x in self.a: yield x
                    for x in self.b: yield x
                def __len__(self):
                    return len(self.a) + len(self.b)
            dl = ConcatLoader(train_dl, val_dl)

        save_gating_alpha(model, dl, cfg, args.save_alpha_path)
        print("[Info] Done saving alpha; exiting.")
        import sys
        sys.exit(0)

    if getattr(args, 'save_alpha_path', None) and args.mode == 'pretrain':
        cfg.save_alpha_path = args.save_alpha_path
        cfg.alpha_split = args.alpha_split

    if args.mode == 'pretrain':
        if cfg.extension_experiment == 'ode_vs_linear':
            run_experiment_ode_vs_linear(cfg)
        elif cfg.extension_experiment == 'irregular_sampling':
            run_experiment_irregular_sampling(cfg)
        elif cfg.extension_experiment == 'step_size_sensitivity':
            run_experiment_step_size_sensitivity(cfg)
        else:
            pretrain(cfg)
    else:
        if cfg.extension_experiment == 'irregular_sampling':
            run_experiment_irregular_sampling(cfg)
        elif cfg.extension_experiment == 'ode_vs_linear':
            run_experiment_ode_vs_linear(cfg)
        elif cfg.extension_experiment == 'step_size_sensitivity':
            run_experiment_step_size_sensitivity(cfg)
        else:
            finetune_main(cfg)
