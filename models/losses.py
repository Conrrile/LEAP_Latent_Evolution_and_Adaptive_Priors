import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StructureAlignmentLoss(nn.Module):
    """
    Structure Alignment Loss with Input-Dependent Domain Gating.
    Dynamically determine the fusion weights for time and frequency domains based on input samples.
    """
    def __init__(self, seq_len: int, input_channels: int, ema_decay: float, temp: float = 0.1, entropy_weight: float = 0.05, fix_domain: str = 'none'):
        super().__init__()
        self.temp = temp
        self.entropy_weight = entropy_weight
        self.fix_domain = fix_domain

        kernel_weights = self._build_ema_kernel_weights(ema_decay)
        self.register_buffer("ema_kernel", kernel_weights.view(1, 1, -1))
        self.kernel_size = self.ema_kernel.shape[-1]
        self.padding = self.kernel_size - 1

        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def _build_ema_kernel_weights(self, alpha, threshold=1e-6):
        weights = []
        w = alpha
        decay = 1.0 - alpha
        while w > threshold:
            weights.append(w)
            w *= decay
        return torch.tensor(weights, dtype=torch.float32).flip(0)

    def _compute_gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        return torch.matmul(x_norm, x_norm.T)

    def _get_dist(self, x: torch.Tensor, sharpen: bool = False) -> torch.Tensor:
        """
        Channel-wise normalization to adapt to sensor data of different scales
        """
        eps = 1e-8

        if x.dim() == 3:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            x = (x - mean) / (std + eps)

            x_flat = x.flatten(start_dim=1)
        else:
            x_flat = x

        x_norm = F.normalize(x_flat, p=2, dim=1)
        sim = torch.matmul(x_norm, x_norm.T) / self.temp

        mask = torch.eye(x.size(0), device=x.device).bool()
        sim.masked_fill_(mask, -9e15)

        dist = F.softmax(sim, dim=-1)
        if sharpen:
            dist = F.normalize(torch.pow(dist, 2), p=1, dim=-1)
        return dist

    def get_pred_dist(self, z_rep: torch.Tensor) -> torch.Tensor:
        sim = self._compute_gram_matrix(z_rep) / self.temp
        mask = torch.eye(z_rep.size(0), device=z_rep.device).bool()
        sim.masked_fill_(mask, -9e15)
        return F.log_softmax(sim, dim=-1)

    def forward(self, raw_input: torch.Tensor, latent_trajectory: torch.Tensor) -> torch.Tensor:
        if raw_input.shape[0] <= 1:
            return torch.tensor(0.0, device=raw_input.device)

        with torch.no_grad():
            b, c, l = raw_input.shape

            x_reshaped = raw_input.view(b * c, 1, l)
            ema_feat = F.conv1d(x_reshaped, self.ema_kernel, padding=self.padding)[..., :l]
            ema_feat = ema_feat.view(b, c, l)
            p_time = self._get_dist(ema_feat, sharpen=True)

            fft_feat = torch.abs(torch.fft.rfft(raw_input, dim=-1))
            p_freq = self._get_dist(fft_feat, sharpen=True)

            if self.fix_domain == 'time':
                freq_weight = torch.zeros(b, 1, device=raw_input.device)
            elif self.fix_domain == 'freq':
                freq_weight = torch.ones(b, 1, device=raw_input.device)
            else:
                freq_weight = torch.sigmoid(self.gate_net(raw_input))

            target_dist = (1 - freq_weight) * p_time + freq_weight * p_freq

        z_rep = latent_trajectory.mean(dim=1)
        log_pred_dist = self.get_pred_dist(z_rep)

        kl_loss = F.kl_div(log_pred_dist, target_dist, reduction='batchmean')

        pred_probs = torch.exp(log_pred_dist)
        entropy = -torch.sum(pred_probs * log_pred_dist, dim=-1).mean()

        return kl_loss + self.entropy_weight * entropy


class StructureAlignmentLoss_Extension(nn.Module):
    """
    Extension of Structure Alignment Loss.
    - loss_type='kl': Strictly aligns with the original LEAP implementation (mixes probabilities).
    - loss_type='mse'/'mae': Regresses on the raw similarity matrices (mixes logits).
    """
    def __init__(self, 
                 seq_len: int, 
                 input_channels: int, 
                 ema_decay: float, 
                 temp: float = 0.1, 
                 entropy_weight: float = 0.05, 
                 fix_domain: str = 'none', 
                 loss_type: str = 'kl'):
        super().__init__()
        self.temp = temp
        self.entropy_weight = entropy_weight
        self.fix_domain = fix_domain
        self.loss_type = loss_type

        kernel_weights = self._build_ema_kernel_weights(ema_decay)
        self.register_buffer("ema_kernel", kernel_weights.view(1, 1, -1))
        self.kernel_size = self.ema_kernel.shape[-1]
        self.padding = self.kernel_size - 1

        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def _build_ema_kernel_weights(self, alpha, threshold=1e-6):
        weights = []
        w = alpha
        decay = 1.0 - alpha
        while w > threshold:
            weights.append(w)
            w *= decay
        return torch.tensor(weights, dtype=torch.float32).flip(0)

    def _compute_gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Standard Cosine Similarity"""
        x_norm = F.normalize(x, p=2, dim=1)
        return torch.matmul(x_norm, x_norm.T)

    def _get_raw_logits_teacher(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes raw similarity logits with Instance Normalization (Teacher specific).
        Does NOT apply mask or softmax yet.
        """
        eps = 1e-8
        if x.dim() == 3:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            x = (x - mean) / (std + eps)
            x_flat = x.flatten(start_dim=1)
        else:
            x_flat = x

        x_norm = F.normalize(x_flat, p=2, dim=1)
        return torch.matmul(x_norm, x_norm.T) / self.temp

    def _process_dist_from_logits(self, logits: torch.Tensor, sharpen: bool = False) -> torch.Tensor:
        """
        Converts raw logits to distribution (Mask -> Softmax -> Sharpen).
        Matches the logic inside the original _get_dist.
        """
        mask = torch.eye(logits.size(0), device=logits.device).bool()
        logits_masked = logits.masked_fill(mask, -9e15)

        dist = F.softmax(logits_masked, dim=-1)
        if sharpen:
            dist = F.normalize(torch.pow(dist, 2), p=1, dim=-1)
        return dist

    def get_pred_dist(self, z_rep: torch.Tensor) -> torch.Tensor:
        """For KL: Returns Log Softmax"""
        sim = self._compute_gram_matrix(z_rep) / self.temp
        sim.masked_fill_(torch.eye(z_rep.size(0), device=z_rep.device).bool(), -9e15)
        return F.log_softmax(sim, dim=-1)

    def forward(self, raw_input: torch.Tensor, latent_trajectory: torch.Tensor) -> torch.Tensor:
        if raw_input.shape[0] <= 1:
            return torch.tensor(0.0, device=raw_input.device)

        with torch.no_grad():
            b, c, l = raw_input.shape

            x_reshaped = raw_input.view(b * c, 1, l)
            ema_feat = F.conv1d(x_reshaped, self.ema_kernel, padding=self.padding)[..., :l]
            ema_feat = ema_feat.view(b, c, l)
            logits_time = self._get_raw_logits_teacher(ema_feat)

            fft_feat = torch.abs(torch.fft.rfft(raw_input, dim=-1))
            logits_freq = self._get_raw_logits_teacher(fft_feat)

            if self.fix_domain == 'time':
                freq_weight = torch.zeros(b, 1, device=raw_input.device)
            elif self.fix_domain == 'freq':
                freq_weight = torch.ones(b, 1, device=raw_input.device)
            else:
                freq_weight = torch.sigmoid(self.gate_net(raw_input))

        if self.loss_type == 'kl':
            with torch.no_grad():
                p_time = self._process_dist_from_logits(logits_time, sharpen=True)
                p_freq = self._process_dist_from_logits(logits_freq, sharpen=True)

                target_dist = (1 - freq_weight) * p_time + freq_weight * p_freq

            z_rep = latent_trajectory.mean(dim=1)
            log_pred_dist = self.get_pred_dist(z_rep)

            kl_loss = F.kl_div(log_pred_dist, target_dist, reduction='batchmean')

            pred_probs = torch.exp(log_pred_dist)
            entropy = -torch.sum(pred_probs * log_pred_dist, dim=-1).mean()

            return kl_loss + self.entropy_weight * entropy

        elif self.loss_type in ['mse', 'mae', 'huber']:
            with torch.no_grad():
                target_logits = (1 - freq_weight) * logits_time + freq_weight * logits_freq

            z_rep = latent_trajectory.mean(dim=1)
            z_norm = F.normalize(z_rep, p=2, dim=1)
            pred_logits = torch.matmul(z_norm, z_norm.T) / self.temp

            mask_diag = torch.eye(b, device=raw_input.device).bool()
            valid_mask = ~mask_diag

            pred_valid = pred_logits[valid_mask]
            target_valid = target_logits[valid_mask]

            if self.loss_type == 'mse':
                return F.mse_loss(pred_valid, target_valid)
            elif self.loss_type == 'mae':
                return F.l1_loss(pred_valid, target_valid)
            elif self.loss_type == 'huber':
                return F.smooth_l1_loss(pred_valid, target_valid)
            elif self.loss_type == 'js':
                with torch.no_grad():
                    p = self._process_dist_from_logits(target_logits, sharpen=False, temperature=1.0)

                q = torch.exp(self.get_pred_dist(z_rep))

                m = 0.5 * (p + q)

                js_part1 = F.kl_div(torch.log(p + 1e-8), m, reduction='batchmean')
                js_part2 = F.kl_div(torch.log(q + 1e-8), m, reduction='batchmean')

                return 0.5 * (js_part1 + js_part2)

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
