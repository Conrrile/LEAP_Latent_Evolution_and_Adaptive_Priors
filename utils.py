import torch
import numpy as np
import random
import os
import csv
from datetime import datetime


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed) 
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)      
    torch.cuda.manual_seed_all(seed)
    
    # 确保完全确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

class SemiLoader:
    def __init__(self, labeled_dl, unlabeled_dl):
        self.labeled_dl = labeled_dl
        self.unlabeled_dl = unlabeled_dl
        self.iter_l = iter(labeled_dl)
        self.iter_u = iter(unlabeled_dl)
    
    def __iter__(self):
        self.iter_l = iter(self.labeled_dl)
        self.iter_u = iter(self.unlabeled_dl)
        return self
    
    def __next__(self):
        try:
            xl, yl = next(self.iter_l)
        except StopIteration:
            self.iter_l = iter(self.labeled_dl)
            xl, yl = next(self.iter_l)
            
        try:
            xu, _ = next(self.iter_u)
        except StopIteration:
            self.iter_u = iter(self.unlabeled_dl)
            xu, _ = next(self.iter_u)
        return xl, yl, xu

def save_checkpoint(path, model, epoch, optimizer=None, save_encoder_only=False, encoder_path=None, extra=None):
    """Save a checkpoint.

    Args:
        path (str): Path to save checkpoint.
        model (nn.Module): Model to save.
        epoch (int): Current epoch number.
        optimizer (Optimizer, optional): Optimizer to save state for.
        save_encoder_only (bool, optional): If True and model has `encoder`, save only encoder state.
        encoder_path (str, optional): Path to save encoder-only checkpoint.
        extra (dict, optional): Additional metadata to include in the checkpoint.
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    if save_encoder_only and hasattr(model, 'encoder'):
        enc_state = model.encoder.state_dict()
        state = {'epoch': epoch, 'encoder_state_dict': enc_state, 'saved_as': 'encoder_only'}
        if optimizer: state['optimizer_state_dict'] = optimizer.state_dict()
        if extra and isinstance(extra, dict):
            state.update(extra)
        save_path = encoder_path if encoder_path else path
        torch.save(state, save_path)
        return

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'saved_as': 'full_model'
    }

    if hasattr(model, 'encoder'):
        try:
            state['encoder_state_dict'] = model.encoder.state_dict()
        except Exception:
            pass

    if optimizer: state['optimizer_state_dict'] = optimizer.state_dict()
    if extra and isinstance(extra, dict):
        state.update(extra)
    torch.save(state, path)

def log_results(path, cfg, metrics):
    """
    Logs results to CSV.
    Updated to include Label Ratio and Finetune Strategy.
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    file_exists = os.path.exists(path)
    
    mode = getattr(cfg, 'mode', 'pretrain')
    if mode == 'finetune':
        mode = getattr(cfg, 'finetune_strategy', 'N/A')
    
    strategy = getattr(cfg, 'finetune_strategy', 'N/A')
    label_ratio = getattr(cfg, 'label_ratio', 1.0)
    
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Dataset', 'Model', 'Mode', 'Strategy', 'Label_Ratio', 'Pretrain_Aug', 'Acc', 'F1', 'Std_Acc', 'Std_F1', 'Source', 'Target', 'Robustness_Test', 'Robustness_Intensity', 'Ablation_Type'])
            
        pre_aug = getattr(cfg, 'pretrain_augment', None)
        if pre_aug is None:
            aj = getattr(cfg, 'augment_jitter', None)
            ascl = getattr(cfg, 'augment_scaling', None)
            if aj is None and ascl is None:
                pre_aug = None
            else:
                pre_aug = bool(aj or ascl)

        # Format std values
        std_acc_val = getattr(cfg, 'std_acc', None)
        std_f1_val = getattr(cfg, 'std_f1', None)
        
        std_acc_str = f"{std_acc_val:.4f}" if std_acc_val is not None else ""
        std_f1_str = f"{std_f1_val:.4f}" if std_f1_val is not None else ""
        
        # Determine ablation type
        ablation_type = 'baseline'
        if getattr(cfg, 'leap_no_evolver', False):
            ablation_type = 'no_evolver'
        elif not getattr(cfg, 'leap_use_projector', True):
            ablation_type = 'no_projector'
        elif getattr(cfg, 'leap_fix_domain', None) == 'time':
            ablation_type = 'time_only'
        elif getattr(cfg, 'leap_fix_domain', None) == 'freq':
            ablation_type = 'freq_only'
        elif getattr(cfg, 'leap_entropy_weight', 0.0) == 0.0:
            ablation_type = 'no_entropy'
        elif getattr(cfg, 'leap_cross_weight', 0.0) == 0.0:
            ablation_type = 'no_cross_view'
        
        # Robustness fields (if present on cfg) for easier downstream inspection
        robustness_test = getattr(cfg, 'robustness_test', 'none')
        robustness_intensity = getattr(cfg, 'robustness_intensity', 'N/A')

        # Optional source/target metadata (useful for transfer-learning logging)
        source_field = getattr(cfg, 'source', '')
        target_field = getattr(cfg, 'target', '')

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            cfg.dataset, 
            cfg.model_name,
            mode,
            strategy,
            label_ratio,
            pre_aug,
            f"{metrics['acc']:.4f}", 
            f"{metrics['macro_f1']:.4f}", 
            std_acc_str,
            std_f1_str,
            source_field,
            target_field,
            robustness_test,
            robustness_intensity,
            ablation_type
        ])

def load_state_dict_compatible(model, state_dict, strict=False):
    def strip_prefix(d, prefix):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in d.items()}

    try:
        model.load_state_dict(state_dict, strict=strict)
        return True, 'loaded_direct'
    except Exception:
        pass

    if any(k.startswith('module.') for k in state_dict.keys()):
        sd = strip_prefix(state_dict, 'module.')
        try:
            model.load_state_dict(sd, strict=strict)
            return True, 'loaded_strip_module'
        except Exception:
            pass

    prefixes = ['encoder.', 'online_encoder.']
    for p in prefixes:
        if any(k.startswith(p) for k in state_dict.keys()):
            sd = strip_prefix(state_dict, p)
            try:
                model.load_state_dict(sd, strict=strict)
                return True, f'loaded_strip_{p}'
            except Exception:
                pass

    try:
        model.load_state_dict(state_dict, strict=False)
        return True, 'loaded_partial'
    except Exception as e:
        return False, f'failed: {e}'

def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score
    return {
        'acc': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro')
    }