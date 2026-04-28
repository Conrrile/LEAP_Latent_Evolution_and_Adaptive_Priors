import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .losses import StructureAlignmentLoss

class LeapEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_dims: Tuple[int, ...], latent_dim: int, dropout: float = 0.0, use_projector: bool = True):
        super().__init__()
        layers = []
        in_dim = input_channels
        for out_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool1d(2)
            ])
            in_dim = out_dim

        self.backbone = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.use_projector = use_projector
        if self.use_projector:
            self.projector = nn.Sequential(
                nn.Linear(hidden_dims[-1], latent_dim),
                nn.LayerNorm(latent_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim)
            )
            self.n_features = latent_dim
        else:
            self.n_features = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.global_pool(feat).squeeze(-1)
        if self.use_projector:
            z0 = self.projector(feat)
        else:
            z0 = feat
        return z0

class LatentEvolver(nn.Module):
    def __init__(self, latent_dim, horizon, dt_init):
        super().__init__()
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.vector_field = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dt_gate = nn.Parameter(torch.full((1, latent_dim), dt_init))
        nn.init.orthogonal_(self.vector_field.weight, gain=0.9)

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        trajectory = []
        curr_state = z0
        dt = F.softplus(self.dt_gate)
        for _ in range(self.horizon):
            k1 = self.vector_field(curr_state)
            k2 = self.vector_field(curr_state + 0.5 * dt * k1)
            k3 = self.vector_field(curr_state + 0.5 * dt * k2)
            k4 = self.vector_field(curr_state + dt * k3)
            curr_state = curr_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(curr_state)
        return torch.stack(trajectory, dim=1)

class LEAP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        model_size = getattr(cfg, 'leap_model_size', 'base')
        preset_map = {
            'small': tuple(getattr(cfg, 'leap_hidden_dims_small', (16, 32))),
            'base': tuple(getattr(cfg, 'leap_hidden_dims', (32, 64, 128))),
            'large': tuple(getattr(cfg, 'leap_hidden_dims_large', (64, 128, 256)))
        }
        hidden_dims = preset_map.get(model_size, tuple(getattr(cfg, 'leap_hidden_dims', (32, 64, 128))))
        latent_dim = getattr(cfg, 'leap_latent_dim', getattr(cfg, 'd_model', 128))
        horizon = getattr(cfg, 'leap_horizon', 5)
        dt_init = getattr(cfg, 'leap_dt_init', 0.1)
        ema_decay = getattr(cfg, 'leap_ema_decay', 0.1)
        dropout = getattr(cfg, 'leap_dropout', 0.1)
        
        entropy_weight = getattr(cfg, 'leap_entropy_weight', 0.05) 
        fix_domain = getattr(cfg, 'leap_fix_domain', 'none')
        use_projector = getattr(cfg, 'leap_use_projector', True)

        self.encoder = LeapEncoder(cfg.n_channel, hidden_dims, latent_dim, dropout=dropout, use_projector=use_projector)
        self.no_evolver = getattr(cfg, 'leap_no_evolver', False)
        if not self.no_evolver:
            self.evolver = LatentEvolver(self.encoder.n_features, horizon, dt_init)
        
        self.n_features = self.encoder.n_features
        
        self.criterion = StructureAlignmentLoss(
            seq_len=cfg.n_length, 
            input_channels=cfg.n_channel, 
            ema_decay=ema_decay, 
            entropy_weight=entropy_weight, 
            fix_domain=fix_domain
        )

    def _compute_cross_view_loss(self, student_traj: torch.Tensor, teacher_mean_rep: torch.Tensor) -> torch.Tensor:
        """
        Cross-View Alignment:
        Implement Trajectory (Step-wise) <-> Teacher (Mean)
        Reuse the metric (get_pred_dist) in self.criterion for consistency
        """
        with torch.no_grad():
            log_target_dist = self.criterion.get_pred_dist(teacher_mean_rep.detach())
            target_prob = torch.exp(log_target_dist)

        total_kl = 0.0
        horizon = student_traj.shape[1]
        
        for t in range(horizon):
            student_step = student_traj[:, t, :]
            log_pred_step = self.criterion.get_pred_dist(student_step)
            total_kl += F.kl_div(log_pred_step, target_prob, reduction='batchmean')
            
        return total_kl / horizon

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        z0_1 = self.encoder(x1)
        if self.no_evolver:
            z_traj1 = z0_1.unsqueeze(1).expand(-1, getattr(self.cfg, 'leap_horizon', 5), -1)
        else:
            z_traj1 = self.evolver(z0_1)
        
        loss = self.criterion(x1, z_traj1)

        if x2 is not None:
            z0_2 = self.encoder(x2)
            if self.no_evolver:
                z_traj2 = z0_2.unsqueeze(1).expand(-1, getattr(self.cfg, 'leap_horizon', 5), -1)
            else:
                z_traj2 = self.evolver(z0_2)
            
            loss_v2 = self.criterion(x2, z_traj2)

            
            z_mean1 = z_traj1.mean(dim=1)
            z_mean2 = z_traj2.mean(dim=1)
            
            loss_cv1 = self._compute_cross_view_loss(z_traj1, z_mean2)
            
            loss_cv2 = self._compute_cross_view_loss(z_traj2, z_mean1)
            
            loss_cross = 0.5 * (loss_cv1 + loss_cv2)

            cross_weight = getattr(self.cfg, 'leap_cross_weight', 1)
            loss = 0.5 * (loss + loss_v2) + cross_weight * loss_cross

        return loss

    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
            """
            Correct implementation for Finetuning support.
            Do NOT use @torch.no_grad() here.
            Do NOT use self.eval() here.
            """
            z0 = self.encoder(x)
            if hasattr(self, "no_evolver") and self.no_evolver: return z0
            traj = self.evolver(z0)
            return traj.mean(dim=1)