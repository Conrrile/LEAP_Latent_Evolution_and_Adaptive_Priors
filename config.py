
import torch
from typing import Dict


DATASET_CONFIG = {
    'HAR':      {'n_channel': 9, 'n_length': 128,  'n_class': 6},
    'HAPT':     {'n_channel': 6, 'n_length': 200,  'n_class': 12},
    'UniMiB':   {'n_channel': 3, 'n_length': 151,  'n_class': 17},
    'FordA':    {'n_channel': 1, 'n_length': 500,  'n_class': 2},
    'FordB':    {'n_channel': 1, 'n_length': 500,  'n_class': 2},
    'SleepEDF': {'n_channel': 1, 'n_length': 3000, 'n_class': 5},
    'CharacterTrajectories': {'n_channel': 3, 'n_length': 205, 'n_class': 20},
    'PTB-XL': {'n_channel': 12, 'n_length': 1000, 'n_class': 5},
    'UWaveGestureLibrary': {'n_channel': 3, 'n_length': 315, 'n_class': 8},
}


class Config:
    def __init__(self, dataset: str = 'HAR'):
        # Basic
        self.dataset = dataset
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.seed = 42
        self.resample_length = None

        # Derived from dataset
        self.match_dataset_params(dataset)

        # Training defaults
        self.batch_size = 128
        self.lr = 3e-4
        self.weight_decay = 1e-4
        self.pretrain_epochs = 50
        self.finetune_epochs = 60
        self.probe_lr = 1e-3
        self.finetune_lr = 1e-4
        self.num_workers = 4
        self.pin_memory = True
        

        self.augment_jitter = True
        self.jitter_sigma = 0.02
        self.augment_scaling = True
        self.scaling_sigma = 0.1


        
        self.output_dir = './experiments/'
        self.load_path = None


        # LEAP specific defaults
        self.leap_latent_dim = 128
        self.leap_use_projector = True
        self.leap_hidden_dims = [32, 64, 128]
        self.leap_horizon = 5
        self.leap_dt_init = 0.1
        self.leap_ema_decay = 0.1
        self.leap_dropout = 0.1
        self.leap_var_weight = 1.0
        self.leap_cov_weight = 0.1
        self.leap_cross_weight = 1.0
        self.leap_entropy_weight = 0.05
        self.leap_fix_domain = 'none'
        self.leap_no_evolver = False
        self.leap_ode_solver = 'rk4'
        self.leap_ode_dynamics = 'linear'
        self.leap_backbone_type = 'shallow_cnn'
        self.leap_model_size = 'base'
        self.leap_loss_type = 'kl'
        self.leap_enc_kernel = 3
        self.leap_temperature = 0.1

    def match_dataset_params(self, dataset: str):
        ds_lower = dataset.lower()
        matched = None
        if dataset in DATASET_CONFIG:
            matched = DATASET_CONFIG[dataset]
        else:
            for k, v in DATASET_CONFIG.items():
                if k.lower() in ds_lower:
                    matched = v
                    break
        if matched:
            self.n_channel = matched['n_channel']
            self.n_length = matched['n_length']
            self.n_class = matched['n_class']
            self.resample_length = matched.get('resample_to', None)
        else:
            self.n_channel = 1
            self.n_length = 200
            self.n_class = 2

    def update(self, args):
        """Apply argparse.Namespace values to config when attribute exists."""
        for k, v in vars(args).items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, v)

        # special mapping: CLI leap_ode_step_size -> internal dt
        if hasattr(args, 'leap_ode_step_size') and args.leap_ode_step_size is not None:
            self.leap_dt_init = args.leap_ode_step_size
            self.leap_ode_step_size = args.leap_ode_step_size