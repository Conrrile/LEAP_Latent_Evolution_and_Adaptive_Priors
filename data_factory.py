import torch
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import random 


def jitter(x, sigma=0.02):
    return x + torch.randn_like(x) * sigma

def add_robust_noise(x, sigma=0.05):
    """Explicitly for robustness testing"""
    return x + torch.randn_like(x) * sigma

def scaling(x, sigma=0.1):
    factor = 1.0 + (torch.randn(x.size(0), 1, 1, device=x.device) * sigma)
    factor = factor.clamp(min=0.01)
    return x * factor

def random_mask(x, prob=0.25):
    mask = (torch.rand_like(x) > prob).float()
    return x * mask

def resample_tensor(x, target_length):
    if x.shape[-1] == target_length: return x
    return F.interpolate(x, size=target_length, mode='linear', align_corners=False)


def _irregular_resample_sample(sample, keep_prob=0.7, seed=None):
    """
    Given a sample tensor of shape (C, L), simulate irregular sampling by
    selecting a subset of time indices and linearly interpolating back to L.
    keep_prob in (0,1] controls fraction of points kept.
    """
    import numpy as _np
    C, L = sample.shape
    rng = _np.random.default_rng(seed)
    k = max(2, int(L * max(0.05, min(1.0, keep_prob))))
    idx = _np.sort(rng.choice(_np.arange(L), size=k, replace=False))
    t_sel = idx.astype(_np.float32) / float(L - 1)
    t_target = _np.linspace(0.0, 1.0, num=L, dtype=_np.float32)

    out = _np.zeros((C, L), dtype=_np.float32)
    for ci in range(C):
        y = sample[ci].cpu().numpy().astype(_np.float32)
        y_sel = y[idx]
        out[ci] = _np.interp(t_target, t_sel, y_sel)
    return torch.from_numpy(out)


def _missing_chunk_sample(sample, missing_frac=0.2, seed=None):
    """Zero out a contiguous chunk of the sample along time axis."""
    import numpy as _np
    C, L = sample.shape
    rng = _np.random.default_rng(seed)
    k = max(1, int(L * max(0.0, min(1.0, missing_frac))))
    start = int(rng.integers(0, max(1, L - k + 1)))
    out = sample.clone()
    out[:, start:start + k] = 0.0
    return out


def seed_worker(worker_id):
    """
    DataLoader Worker initialization function.
    Ensure each worker has a different but reproducible seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class PretrainDataset(Dataset):
    """
    Standardized pretraining dataset.
    Note: The data loader provides two views (x1, x2) and labels y (for Online Probe monitoring).
    """
    def __init__(self, x, y, cfg):
        self.x = x
        self.y = y
        self.cfg = cfg

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx].float()
        label = self.y[idx].long()
        
        x1 = sample.unsqueeze(0)  
        x2 = sample.unsqueeze(0)

        # apply augmentations according to config
        # special strong/weak mode: both augment types on x1, only jitter on x2
        if getattr(self.cfg, 'augment_strong_weak', False):
            sigma_j = getattr(self.cfg, 'jitter_sigma', 0.02)
            sigma_s = getattr(self.cfg, 'scaling_sigma', 0.1)
            x1 = jitter(x1, sigma=sigma_j)
            x1 = scaling(x1, sigma=sigma_s)
            x2 = jitter(x2, sigma=sigma_j)
        else:
            if getattr(self.cfg, 'augment_jitter', False):
                sigma = getattr(self.cfg, 'jitter_sigma', 0.02)
                x1 = jitter(x1, sigma=sigma)
                if getattr(self.cfg, 'augment_both_sides', False):
                    x2 = jitter(x2, sigma=sigma)

            if getattr(self.cfg, 'augment_scaling', False):
                sigma = getattr(self.cfg, 'scaling_sigma', 0.1)
                x1 = scaling(x1, sigma=sigma)
                if getattr(self.cfg, 'augment_both_sides', False):
                    x2 = scaling(x2, sigma=sigma)

        x1 = x1.squeeze(0)
        x2 = x2.squeeze(0)
        return x1, x2, label


def get_stratified_indices(y, ratio, seed):
    """Stratified sampling for few-shot/semi-supervised learning"""
    if ratio >= 1.0: return np.arange(len(y))
    n_samples = len(y)
    indices = np.arange(n_samples)
    selected = []
    rng = np.random.default_rng(seed)
    for c in np.unique(y):
        c_idx = indices[y == c]
        n_sel = max(1, int(len(c_idx) * ratio))
        selected.extend(rng.choice(c_idx, n_sel, replace=False))
    return np.array(selected)

def load_data(cfg, mode='finetune', ratio=1.0):
    base_dir = os.path.join('datasets', cfg.dataset)
    paths = [os.path.join(base_dir, f"{cfg.dataset}.pt"), f"{cfg.dataset}.pt"]
    
    data = None
    for p in paths:
        if os.path.exists(p):
            data = torch.load(p)
            break
            
    if data is None:
        try:
            train = torch.load(os.path.join(base_dir, 'train.pt'))
            val = torch.load(os.path.join(base_dir, 'val.pt'))
            test = torch.load(os.path.join(base_dir, 'test.pt'))
            train_x, train_y = train['samples'], train['labels']
            val_x, val_y = val['samples'], val['labels']
            test_x, test_y = test['samples'], test['labels']
        except:
            raise FileNotFoundError(f"Data not found for {cfg.dataset} in {base_dir}")
    else:
        train_x, train_y = data['train']['samples'], data['train']['labels']
        val_x, val_y = data['val']['samples'], data['val']['labels']
        test_x, test_y = data['test']['samples'], data['test']['labels']

    to_tensor = lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x)
    train_x, val_x, test_x = map(lambda t: to_tensor(t).float(), [train_x, val_x, test_x])
    train_y, val_y, test_y = map(lambda t: to_tensor(t).long(), [train_y, val_y, test_y])

    def _ensure_channel(t):
        if t.dim() == 2:
            return t.unsqueeze(1)
        elif t.dim() == 3:
            cfg_ch = getattr(cfg, 'n_channel', None)
            if cfg_ch is not None and t.size(1) == cfg_ch:
                return t
            if cfg_ch is not None and t.size(2) == cfg_ch:
                return t.permute(0, 2, 1).contiguous()
            if t.size(1) <= 16 and t.size(2) > 16:
                return t
            return t.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"Unexpected data tensor shape: {t.shape}")

    train_x, val_x, test_x = map(_ensure_channel, [train_x, val_x, test_x])

    if train_x.size(1) != getattr(cfg, 'n_channel', train_x.size(1)):
        print(f"[Data] Warning: dataset channel mismatch: got {train_x.size(1)}, expected {cfg.n_channel}")

    if cfg.resample_length is not None and train_x.shape[-1] != cfg.resample_length:
        print(f"[Data] Resampling {train_x.shape[-1]} -> {cfg.resample_length}")
        train_x = resample_tensor(train_x, cfg.resample_length)
        val_x = resample_tensor(val_x, cfg.resample_length)
        test_x = resample_tensor(test_x, cfg.resample_length)
        cfg.n_length = cfg.resample_length

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    if mode == 'pretrain':
        train_ds = PretrainDataset(train_x, train_y, cfg)
        val_ds = PretrainDataset(val_x, val_y, cfg) 
        
        train_dl = DataLoader(
            train_ds, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers, 
            drop_last=True, 
            pin_memory=cfg.pin_memory,
            worker_init_fn=seed_worker, 
            generator=g                 
        )
        
        val_dl = DataLoader(
            val_ds, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=cfg.num_workers, 
            drop_last=False, 
            pin_memory=cfg.pin_memory,
            worker_init_fn=seed_worker, 
            generator=g                 
        )

        return train_dl, val_dl, None

    idxs = get_stratified_indices(train_y.numpy(), ratio, cfg.seed)
    train_x_l, train_y_l = train_x[idxs], train_y[idxs]

    downstream_bs = getattr(cfg, 'downstream_batch_size', cfg.batch_size)

    train_dl = DataLoader(
        TensorDataset(train_x_l, train_y_l), 
        batch_size=downstream_bs, 
        shuffle=True, 
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory,
        worker_init_fn=seed_worker, 
        generator=g                 
    )
    
    val_dl = DataLoader(
        TensorDataset(val_x, val_y), 
        batch_size=downstream_bs, 
        shuffle=False, 
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory,
        worker_init_fn=seed_worker, 
        generator=g                 
    )
    
    test_dl = DataLoader(
        TensorDataset(test_x, test_y), 
        batch_size=downstream_bs, 
        shuffle=False, 
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory,
        worker_init_fn=seed_worker, 
        generator=g                 
    )


    if hasattr(cfg, 'irregular_test_mode') and cfg.irregular_test_mode is not None and cfg.irregular_test_mode != 'none':
        mode_flag = str(cfg.irregular_test_mode).lower()
        print(f"[Data] Applying test-time transform: {mode_flag}")
        test_x_mod = test_x.clone()
        seed_base = int(getattr(cfg, 'seed', 0))
        for i in range(test_x_mod.size(0)):
            sample = test_x_mod[i]  # (C, L)
            if mode_flag == 'irregular':
                kp = float(getattr(cfg, 'irregular_keep_prob', 0.7))
                test_x_mod[i] = _irregular_resample_sample(sample, keep_prob=kp, seed=seed_base + i)
            elif mode_flag == 'reverse':
                test_x_mod[i] = sample.flip(-1)
            elif mode_flag == 'missing':
                frac = float(getattr(cfg, 'missing_chunk_frac', 0.2))
                test_x_mod[i] = _missing_chunk_sample(sample, missing_frac=frac, seed=seed_base + i)
            else:
                # unknown -> no op
                pass

        test_dl = DataLoader(
            TensorDataset(test_x_mod, test_y),
            batch_size=downstream_bs,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            worker_init_fn=seed_worker,
            generator=g
        )
    
    # Add noise to test set if specified
    if hasattr(cfg, 'noise_level') and cfg.noise_level > 0:
        print(f"[Data] Adding noise to test set with level {cfg.noise_level}")
        test_x_noisy = test_x + torch.randn_like(test_x) * cfg.noise_level
        test_dl = DataLoader(
            TensorDataset(test_x_noisy, test_y), 
            batch_size=downstream_bs, 
            shuffle=False, 
            num_workers=cfg.num_workers, 
            pin_memory=cfg.pin_memory,
            worker_init_fn=seed_worker, 
            generator=g                 
        )
    
    if mode == 'semi':
        unlab_idxs = np.setdiff1d(np.arange(len(train_y)), idxs)
        if len(unlab_idxs) > 0:
            unlab_dl = DataLoader(
                TensorDataset(train_x[unlab_idxs], train_y[unlab_idxs]), 
                batch_size=downstream_bs, 
                shuffle=True, 
                drop_last=True, 
                num_workers=cfg.num_workers, 
                pin_memory=cfg.pin_memory,
                worker_init_fn=seed_worker, 
                generator=g                 
            )
            from utils import SemiLoader
            return SemiLoader(train_dl, unlab_dl), val_dl, test_dl

    return train_dl, val_dl, test_dl