# rl/dqn_agent.py
"""
Minimal DQN module (PyTorch) for threshold-control tasks.

Exports:
  - DQNAgent
  - make_obs(...)
  - shield_delta(...)
  - compute_reward(...)

No domain-specific code here.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import random
import numpy as np
import math

# --- torch import guarded so main script can error nicely if missing ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required.\nInstall: pip install torch\n\n"
        f"Import error: {e}"
    )

# ------------------------ replay buffer ------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.capacity = int(capacity)
        self.data = []
        self.i = 0

    def push(self, s, a, r, sp, done: bool):
        item = (
            np.asarray(s, np.float32),
            int(a),
            float(r),
            np.asarray(sp, np.float32),
            float(done),
        )
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.i] = item
        self.i = (self.i + 1) % self.capacity

    def sample(self, batch_size: int = 128):
        batch = random.sample(self.data, batch_size)
        s, a, r, sp, done = zip(*batch)
        return (
            np.stack(s),
            np.asarray(a, np.int64),
            np.asarray(r, np.float32),
            np.stack(sp),
            np.asarray(done, np.float32),
        )

    def __len__(self):
        return len(self.data)

# ------------------------ networks ------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)

# ------------------------ agent ------------------------
@dataclass
class DQNConfig:
    lr: float = 5e-4
    gamma: float = 0.95
    batch_size: int = 128
    target_update: int = 200
    buffer_capacity: int = 50_000
    grad_clip: float = 5.0

class DQNAgent:
    """
    Vanilla Double-DQN with:
      - SmoothL1 (Huber)
      - target network
      - replay buffer
      - epsilon-greedy action selection
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        seed: int = 0,
        device: Optional[str] = None,
        cfg: Optional[DQNConfig] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.cfg = cfg or DQNConfig()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.q = QNet(self.obs_dim, self.n_actions).to(self.device)
        self.tgt = QNet(self.obs_dim, self.n_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buf = ReplayBuffer(capacity=self.cfg.buffer_capacity)

        self.train_steps = 0

    def act(self, obs: np.ndarray, eps: float = 0.1) -> int:
        """Epsilon-greedy."""
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q(x)[0]
            return int(torch.argmax(qvals).item())

    def train_step(self) -> Optional[float]:
        """One gradient step. Returns loss or None if not enough data."""
        bs = self.cfg.batch_size
        if len(self.buf) < bs:
            return None

        s, a, r, sp, done = self.buf.sample(bs)

        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        sp = torch.tensor(sp, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)

        # Double DQN target
        with torch.no_grad():
            a_star = torch.argmax(self.q(sp), dim=1, keepdim=True)
            q_sp = self.tgt(sp).gather(1, a_star)
            y = r + (1.0 - done) * self.cfg.gamma * q_sp

        loss = nn.SmoothL1Loss()(q_sa, y)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update == 0:
            self.tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())

# ------------------------ observation + reward helpers ------------------------
def make_obs(
    bg_rate: float,
    prev_bg_rate: float,
    cut: float,
    cut_mid: float,
    cut_span: float,
    target: float,
) -> np.ndarray:
    """
    Default 3D observation used for RL agent.:
      [ normalized_error, normalized_delta_error, normalized_cut ]
    """
    cut_span = max(1e-12, float(cut_span))
    target = max(1e-12, float(target))
    x_rate = (float(bg_rate) - target) / target
    x_drate = (float(bg_rate) - float(prev_bg_rate)) / target
    x_cut = (float(cut) - float(cut_mid)) / cut_span
    return np.array([x_rate, x_drate, x_cut], dtype=np.float32)

def _downsample_or_pad(x: np.ndarray, K: int) -> np.ndarray:
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return np.zeros(K, dtype=np.float32)
    if n >= K:
        idx = np.linspace(0, n - 1, K).astype(int)
        return x[idx].astype(np.float32)
    # pad by repeating last value
    out = np.empty(K, dtype=np.float32)
    out[:n] = x.astype(np.float32)
    out[n:] = float(x[-1])
    return out

def make_event_seq_ht(
    bht: np.ndarray,
    bnpv: np.ndarray,
    bg_rate: float,
    prev_bg_rate: float,
    cut: float,
    ht_mid: float,
    ht_span: float,
    target: float,
    K: int = 128,
) -> np.ndarray:
    """
    HT agent event-level state: (K, F)
    Per-event features: [norm_ht, norm_npv, pass_ht]
    Global features (repeated each step): [norm_err, norm_drate, norm_cut]
    Total F = 6
    """
    ht_span = max(1e-12, float(ht_span))
    target = max(1e-12, float(target))

    ht = _downsample_or_pad(bht, K)
    npv = _downsample_or_pad(bnpv, K)

    norm_ht  = (ht - ht_mid) / ht_span
    norm_npv = (npv - np.mean(npv)) / max(1e-6, float(np.std(npv)))
    pass_ht  = (ht >= float(cut)).astype(np.float32)

    # global (PID-like) signals, but now the agent also sees events
    g_err  = (float(bg_rate) - target) / target
    g_derr = (float(bg_rate) - float(prev_bg_rate)) / target
    g_cut  = (float(cut) - ht_mid) / ht_span
    g = np.array([g_err, g_derr, g_cut], dtype=np.float32)
    G = np.tile(g[None, :], (K, 1))

    X = np.stack([norm_ht, norm_npv, pass_ht], axis=1)  # (K,3)
    return np.concatenate([X, G], axis=1).astype(np.float32)  # (K,6)

def make_event_seq_as(
    bas: np.ndarray,
    bnpv: np.ndarray,
    bg_rate: float,
    prev_bg_rate: float,
    cut: float,
    as_mid: float,
    as_span: float,
    target: float,
    K: int = 128,
) -> np.ndarray:
    """
    AS agent event-level state: (K, F)
    Per-event features: [norm_as, norm_npv, pass_as]
    Global features:    [norm_err, norm_drate, norm_cut]
    Total F = 6
    """
    as_span = max(1e-12, float(as_span))
    target = max(1e-12, float(target))

    a = _downsample_or_pad(bas, K)
    npv = _downsample_or_pad(bnpv, K)

    norm_as  = (a - as_mid) / as_span
    norm_npv = (npv - np.mean(npv)) / max(1e-6, float(np.std(npv)))
    pass_as  = (a >= float(cut)).astype(np.float32)

    g_err  = (float(bg_rate) - target) / target
    g_derr = (float(bg_rate) - float(prev_bg_rate)) / target
    g_cut  = (float(cut) - as_mid) / as_span
    g = np.array([g_err, g_derr, g_cut], dtype=np.float32)
    G = np.tile(g[None, :], (K, 1))

    X = np.stack([norm_as, norm_npv, pass_as], axis=1)  # (K,3)
    return np.concatenate([X, G], axis=1).astype(np.float32)  # (K,6)


def shield_delta(
    bg_rate: float,
    target: float,
    tol: float,
    max_delta: float,
) -> Optional[float]:
    """
    If agent is too far from target, force a strong move in the correct direction.
      - bg too high => increase cut (positive delta)
      - bg too low  => decrease cut (negative delta)
    """
    if bg_rate > target + tol:
        return +float(max_delta)
    if bg_rate < target - tol:
        return -float(max_delta)
    return None

def compute_reward(
    bg_rate: float,
    target: float,
    tol: float,
    sig_rate_1: float,
    sig_rate_2: float,
    delta_applied: float,
    max_delta: float,
    alpha: float = 0.2,
    beta: float = 0.02,
    clip: Tuple[float, float] = (-10.0, 10.0),
    prev_bg_rate: Optional[float] = None,
    gamma_stab: float = 0.25, # weight for stability penalty 0.25 default
) -> float:
    """
    sig_rate_1: first signal rate (e.g. TTbar) focuses more on TTbar
    sig_rate_2: second signal rate (e.g. HToAATo4B)

    Generic reward:
      + in-band tracking bonus (encourages holding)
      - out-of-band penalty grows smoothly
    #   - background penalty: |bg-target|/tol
      + signal bonus: alpha * mean(sig)/100
      - movement penalty: beta * |delta|/max_delta
    """
    tol = max(1e-12, float(tol))
    max_delta = max(1e-12, float(max_delta))

    # normalized error
    e = (float(bg_rate) - float(target)) / tol
    ae = abs(e)

    # Tracking: reward being within tolerance, penalize being outside
    if ae <= 1.0:
        # max +1 at center; smoothly decreases to 0 at band edge
        track = 1.0 - ae**2
    else:
        # linear penalty outside band, continuous at ae=1
        track = - (ae - 1.0)


    bg_pen = abs(float(bg_rate) - float(target)) / tol
    sig_term = 0.5 * (2 * float(sig_rate_1) + float(sig_rate_2)) / 100.0

    move_pen = abs(float(delta_applied)) / max_delta

    if prev_bg_rate is None:
        stab_pen = 0.0
    else:
        db = abs(float(bg_rate) - float(prev_bg_rate)) / tol
        stab_pen = db * db
    # r = -bg_pen + alpha * sig_term - beta * move_pen
    r = track + alpha * sig_term - beta * move_pen - gamma_stab * stab_pen
    lo, hi = clip 
    return float(np.clip(r, lo, hi))




### Encode event-level sequences with a GRU-based Q-network ###
# ------------------------ sequence network ------------------------
class SeqQNet(nn.Module):
    """
    Q-network for event-level sequences.
    Input:  x of shape (B, K, F)
    Output: Q-values of shape (B, n_actions)
    """
    def __init__(self, feat_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        # x: (B, K, F)
        _, h = self.gru(x)      # h: (1, B, hidden)
        h = h[-1]               # (B, hidden)
        return self.head(h)


class ReplayBufferSeq:
    def __init__(self, capacity: int = 50_000):
        self.capacity = int(capacity)
        self.data = []
        self.i = 0

    def push(self, s_seq, a, r, sp_seq, done: bool):
        item = (
            np.asarray(s_seq, np.float32),   # (K, F)
            int(a),
            float(r),
            np.asarray(sp_seq, np.float32),  # (K, F)
            float(done),
        )
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.i] = item
        self.i = (self.i + 1) % self.capacity

    def sample(self, batch_size: int = 128):
        batch = random.sample(self.data, batch_size)
        s, a, r, sp, done = zip(*batch)
        return (
            np.stack(s),  # (B, K, F)
            np.asarray(a, np.int64),
            np.asarray(r, np.float32),
            np.stack(sp), # (B, K, F)
            np.asarray(done, np.float32),
        )

    def __len__(self):
        return len(self.data)


class SeqDQNAgent:
    """
    Double-DQN for event-level sequences.
    Same API style as DQNAgent, but obs is (K, F) instead of (obs_dim,).
    """
    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        n_actions: int,
        seed: int = 0,
        device: Optional[str] = None,
        cfg: Optional[DQNConfig] = None,
    ):
        self.seq_len = int(seq_len)
        self.feat_dim = int(feat_dim)
        self.n_actions = int(n_actions)
        self.cfg = cfg or DQNConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.q = SeqQNet(self.feat_dim, self.n_actions).to(self.device)
        self.tgt = SeqQNet(self.feat_dim, self.n_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buf = ReplayBufferSeq(capacity=self.cfg.buffer_capacity)
        self.train_steps = 0

    def act(self, obs_seq: np.ndarray, eps: float = 0.1) -> int:
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,K,F)
            qvals = self.q(x)[0]
            return int(torch.argmax(qvals).item())

    def train_step(self) -> Optional[float]:
        bs = self.cfg.batch_size
        if len(self.buf) < bs:
            return None

        s, a, r, sp, done = self.buf.sample(bs)

        s = torch.tensor(s, dtype=torch.float32, device=self.device)            # (B,K,F)
        sp = torch.tensor(sp, dtype=torch.float32, device=self.device)          # (B,K,F)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)

        with torch.no_grad():
            a_star = torch.argmax(self.q(sp), dim=1, keepdim=True)
            q_sp = self.tgt(sp).gather(1, a_star)
            y = r + (1.0 - done) * self.cfg.gamma * q_sp

        loss = nn.SmoothL1Loss()(q_sa, y)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update == 0:
            self.tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())
