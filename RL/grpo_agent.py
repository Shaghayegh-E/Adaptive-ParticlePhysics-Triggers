# RL/grpo_agent.py
"""
GRPO-style (Group Relative Policy Optimization) for discrete action spaces.

We treat each decision as a bandit-style step:
  - For the same observation s, sample a *group* of candidate actions.
  - Evaluate reward for each candidate.
  - Use relative advantage: A_i = r_i - mean(r_group).
  - Update a categorical policy with PPO-style clipping + optional KL to a reference policy.

Bonus points here are Rewards are cheap to evaluate
for multiple candidate deltas at the same micro-step.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except Exception as e:
    raise SystemExit(
        "PyTorch is required.\nInstall: pip install torch\n\n"
        f"Import error: {e}"
    )


@dataclass
class GRPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    beta_kl: float = 0.02          # KL penalty strength (to reference)
    ent_coef: float = 0.01         # entropy bonus
    train_epochs: int = 2
    batch_size: int = 256
    ref_update_interval: int = 200 # how often to sync reference
    max_grad_norm: float = 1.0
    device: str = "cpu"


class SeqPolicy(nn.Module):
    """
    Simple sequence -> logits policy.
    Input: (B, K, F)
    Output: (B, A)
    """
    def __init__(self, feat_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_actions = n_actions

        # Pool sequence with mean/max + last token, then MLP
        in_dim = 3 * feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,K,F)
        mean = x.mean(dim=1)
        mx, _ = x.max(dim=1)
        last = x[:, -1, :]
        h = torch.cat([mean, mx, last], dim=-1)
        return self.mlp(h)


class GRPOBuffer:
    """
    Stores group samples: (obs, action, old_logp, adv)
    """
    def __init__(self, capacity: int = 200_000):
        self.capacity = int(capacity)
        self.reset()

    def reset(self):
        self.obs = []
        self.act = []
        self.old_logp = []
        self.adv = []

    def push(self, obs: np.ndarray, act: int, old_logp: float, adv: float):
        if len(self.obs) >= self.capacity:
            # simple FIFO
            self.obs.pop(0); self.act.pop(0); self.old_logp.pop(0); self.adv.pop(0)
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.act.append(int(act))
        self.old_logp.append(float(old_logp))
        self.adv.append(float(adv))

    def __len__(self):
        return len(self.obs)

    def as_tensors(self, device: str):
        obs = torch.tensor(np.stack(self.obs), dtype=torch.float32, device=device)   # (B,K,F)
        act = torch.tensor(np.array(self.act), dtype=torch.long, device=device)     # (B,)
        old_logp = torch.tensor(np.array(self.old_logp), dtype=torch.float32, device=device)  # (B,)
        adv = torch.tensor(np.array(self.adv), dtype=torch.float32, device=device)  # (B,)
        return obs, act, old_logp, adv


class GRPOAgent:
    def __init__(self, seq_len: int, feat_dim: int, n_actions: int, cfg: GRPOConfig, seed: int = 0):
        self.seq_len = int(seq_len)
        self.feat_dim = int(feat_dim)
        self.n_actions = int(n_actions)
        self.cfg = cfg

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device(cfg.device)
        self.pi = SeqPolicy(feat_dim=feat_dim, n_actions=n_actions).to(self.device)
        self.pi_ref = SeqPolicy(feat_dim=feat_dim, n_actions=n_actions).to(self.device)
        self.pi_ref.load_state_dict(self.pi.state_dict())
        for p in self.pi_ref.parameters():
            p.requires_grad = False

        self.opt = optim.Adam(self.pi.parameters(), lr=cfg.lr)
        self.buf = GRPOBuffer()
        self._update_count = 0

    @torch.no_grad()
    def dist(self, obs: np.ndarray) -> torch.distributions.Categorical:
        x = torch.tensor(obs[None, ...], dtype=torch.float32, device=self.device)  # (1,K,F)
        logits = self.pi(x)
        return torch.distributions.Categorical(logits=logits)

    @torch.no_grad()
    def sample_group_actions(self, obs: np.ndarray, group_size: int, temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          actions: (G,)
          logp:    (G,) under current policy
        """
        x = torch.tensor(obs[None, ...], dtype=torch.float32, device=self.device)  # (1,K,F)
        logits = self.pi(x) / max(1e-6, float(temperature))
        dist = torch.distributions.Categorical(logits=logits)

        # sample with replacement; group can contain repeats (fine)
        acts = dist.sample((group_size,)).squeeze(-1)  # (G,)
        logp = dist.log_prob(acts)                     # (G,)

        return acts.cpu().numpy(), logp.cpu().numpy()

    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray) -> int:
        d = self.dist(obs)
        return int(torch.argmax(d.probs).item())

    def update(self) -> Optional[float]:
        """
        PPO-style clipped objective + entropy + KL(pi||pi_ref).
        Returns mean loss (float) if update happens, else None.
        """
        if len(self.buf) < max(self.cfg.batch_size, 128):
            return None

        obs, act, old_logp, adv = self.buf.as_tensors(device=str(self.device))

        # Normalize advantages globally (still relative, but helps optimizer)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        B = obs.shape[0]
        idx = torch.randperm(B, device=self.device)

        losses = []
        for _ in range(int(self.cfg.train_epochs)):
            for start in range(0, B, int(self.cfg.batch_size)):
                j = idx[start:start + int(self.cfg.batch_size)]
                ob = obs[j]
                ac = act[j]
                olp = old_logp[j]
                ad = adv[j]

                logits = self.pi(ob)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(ac)
                ratio = torch.exp(logp - olp)

                # PPO clip
                unclipped = ratio * ad
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * ad
                loss_pg = -torch.mean(torch.minimum(unclipped, clipped))

                # Entropy bonus
                loss_ent = -self.cfg.ent_coef * torch.mean(dist.entropy())

                # KL to reference policy
                with torch.no_grad():
                    logits_ref = self.pi_ref(ob)
                dist_ref = torch.distributions.Categorical(logits=logits_ref)
                kl = torch.distributions.kl_divergence(dist, dist_ref).mean()
                loss_kl = self.cfg.beta_kl * kl

                loss = loss_pg + loss_ent + loss_kl

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                losses.append(loss.item())

        self._update_count += 1
        if self._update_count % int(self.cfg.ref_update_interval) == 0:
            self.pi_ref.load_state_dict(self.pi.state_dict())

        # Clear buffer after update
        self.buf.reset()
        return float(np.mean(losses)) if losses else None
