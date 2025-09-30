# ============================================
# Open In Colab
# DQN on ALE/Q*bert with guaranteed 15–30s videos
# NOTE: Only the VIDEO RECORDING BLOCK changed to force duration.
# ============================================

# ---- Installs (Colab-friendly) ----
!pip -q install "gymnasium[atari,accept-rom-license]" autorom stable-baselines3 imageio[ffmpeg]
!AutoROM --accept-license -y

# ---- Imports & setup ----
import os, time, glob, collections
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import typing as tt
import numpy as np
import imageio.v2 as imageio

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common import atari_wrappers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

# Ensure ALE can find ROMs
try:
    autorom_rom_dirs = glob.glob("/usr/local/lib/python*/dist-packages/AutoROM/roms")
    if autorom_rom_dirs:
        os.environ["ALE_PY_ROM_DIR"] = autorom_rom_dirs[0]
except Exception:
    pass

# (Optional) Mount Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_OK = True
except Exception:
    print("Google Drive not available; will save locally only.")
    DRIVE_OK = False

# ---- Dirs ----
save_dir_drive = "/content/drive/MyDrive/PUBLIC/Models" if DRIVE_OK else None
save_dir_local = "saved_models"
video_dir = "videos"
os.makedirs(save_dir_local, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
if save_dir_drive:
    os.makedirs(save_dir_drive, exist_ok=True)

# ---- Model (DQN) ----
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        self.fc = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def forward(self, x: torch.ByteTensor):
        x = x.float() / 255.0
        return self.fc(self.conv(x))

# ---- Wrappers ----
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs = self.observation_space
        assert isinstance(obs, gym.spaces.Box) and len(obs.shape) == 3
        new_shape = (obs.shape[-1], obs.shape[0], obs.shape[1])
        self.observation_space = gym.spaces.Box(
            low=obs.low.min(), high=obs.high.max(),
            shape=new_shape, dtype=obs.dtype
        )
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super().__init__(env)
        obs = env.observation_space
        assert isinstance(obs, spaces.Box)
        self.observation_space = gym.spaces.Box(
            obs.low.repeat(n_steps, axis=0),
            obs.high.repeat(n_steps, axis=0),
            dtype=obs.dtype
        )
        self.buffer = collections.deque(maxlen=n_steps)
    def reset(self, *, seed: tt.Optional[int] = None, options: tt.Optional[dict[str, tt.Any]] = None):
        for _ in range(self.buffer.maxlen):
            self.buffer.append(np.zeros_like(self.env.observation_space.low))
        obs, extra = self.env.reset(seed=seed, options=options)
        return self.observation(obs), extra
    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.buffer.append(observation)
        return np.concatenate(self.buffer)

def make_env(env_name: str, n_steps=4, render_mode=None, **kwargs):
    print(f"Creating environment {env_name}")
    env = gym.make(env_name, render_mode=render_mode, **kwargs)
    env = atari_wrappers.AtariWrapper(env, clip_reward=False, noop_max=0)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=n_steps)
    return env

# ---- Hyperparams ----
DEFAULT_ENV_NAME = "ALE/Qbert-v5"
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10_000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 500
REPLAY_START_SIZE = 1_000
SAVE_EPSILON = 0.5
EPSILON_DECAY_LAST_FRAME = 10_000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

# Typing
State = np.ndarray
Action = int
BatchTensors = tt.Tuple[torch.ByteTensor, torch.LongTensor, torch.Tensor, torch.BoolTensor, torch.ByteTensor]

# ---- Replay & Agent ----
@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool
    new_state: State

class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)
    def __len__(self):
        return len(self.buffer)
    def append(self, experience: Experience):
        self.buffer.append(experience)
    def sample(self, batch_size: int) -> tt.List[Experience]:
        idx = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

class Agent:
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state: tt.Optional[np.ndarray] = None
        self._reset()
    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0
    @torch.no_grad()
    def play_step(self, net: DQN, device: torch.device, epsilon: float = 0.0) -> tt.Optional[float]:
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            s = torch.as_tensor(self.state).to(device).unsqueeze(0)
            q = net(s)
            action = int(torch.argmax(q, dim=1).item())
        new_state, reward, is_done, is_tr, _ = self.env.step(action)
        self.total_reward += reward
        self.exp_buffer.append(Experience(self.state, action, float(reward), bool(is_done or is_tr), new_state))
        self.state = new_state
        if is_done or is_tr:
            done_reward = self.total_reward
            self._reset()
        return done_reward

# ---- Training utils ----

def batch_to_tensors(batch: tt.List[Experience], device: torch.device) -> BatchTensors:
    states = torch.as_tensor(np.asarray([e.state for e in batch])).to(device)
    actions = torch.LongTensor([e.action for e in batch]).to(device)
    rewards = torch.FloatTensor([e.reward for e in batch]).to(device)
    dones = torch.BoolTensor([e.done_trunc for e in batch]).to(device)
    new_states = torch.as_tensor(np.asarray([e.new_state for e in batch])).to(device)
    return states, actions, rewards, dones, new_states


def calc_loss(batch: tt.List[Experience], net: DQN, tgt_net: DQN, device: torch.device) -> torch.Tensor:
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)
    state_action_values = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(new_states_t).max(1)[0]
        next_state_values[dones_t] = 0.0
    expected = rewards_t + 0.99 * next_state_values
    return nn.MSELoss()(state_action_values, expected)

# ---- Build env & nets ----
env_name = DEFAULT_ENV_NAME
safe_env_name = env_name.replace('/', '_')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_comment = f"test_epsdec{EPSILON_DECAY_LAST_FRAME}_rs{REPLAY_START_SIZE}_sync{SYNC_TARGET_FRAMES}"

env = make_env(env_name)
net = DQN(env.observation_space.shape, env.action_space.n).to(device)
tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
writer = SummaryWriter(comment=f"-{env_name}-{model_comment}")
print(net)

buffer = ExperienceBuffer(REPLAY_SIZE)
agent = Agent(env, buffer)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

total_rewards, frame_idx, ts_frame = [], 0, 0
ts, best_m_reward, start_time = time.time(), None, time.time()

# ---- Helper: save best ----

def save_model_if_best(m_reward):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    fname = f"{safe_env_name}-best_{int(m_reward)}-{timestamp}-{model_comment}.dat"
    p_local = os.path.join(save_dir_local, fname)
    torch.save(net.state_dict(), p_local)
    print(f"\n💾 Model saved:\n - Local: {p_local}")
    if save_dir_drive:
        p_drive = os.path.join(save_dir_drive, fname)
        torch.save(net.state_dict(), p_drive)
        print(f" - Google Drive: {p_drive}")

# ======================================================
# 🎥 VIDEO RECORDING BLOCK — FORCE 15–30s VIA FRAME BUDGET
# Records ~target_seconds at fps (defaults: 20s @ 60fps => ~1200 frames).
# Continues across episodes until enough frames captured.
# ======================================================

def record_policy_video(net, epsilon: float, name_prefix: str, target_seconds: int = 20, fps: int = 60, max_episodes: int = 50):
    """Record a ~fixed-length clip to ./videos/<name_prefix>.mp4.
    Captures frames across multiple episodes if needed to hit duration.
    """
    target_frames = int(target_seconds * fps)

    # Build an eval env graph that mirrors training but preserves rgb_array rendering
    base = gym.make(env_name, render_mode='rgb_array')
    eval_env = atari_wrappers.AtariWrapper(base, clip_reward=False, noop_max=0)
    eval_env = ImageToPyTorch(eval_env)
    eval_env = BufferWrapper(eval_env, n_steps=4)

    frames = []
    episodes = 0
    obs, info = eval_env.reset()
    done = trunc = False

    with torch.no_grad():
        while len(frames) < target_frames and episodes < max_episodes:
            # Action selection
            if np.random.rand() < epsilon:
                action = eval_env.action_space.sample()
            else:
                s = torch.as_tensor(obs).unsqueeze(0).to(device)
                q = net(s)
                action = int(torch.argmax(q, dim=1).item())

            # Step + capture frame from base renderer
            obs, reward, done, trunc, _ = eval_env.step(action)
            frame = base.render()
            if frame is not None:
                frames.append(frame)

            if done or trunc:
                episodes += 1
                obs, info = eval_env.reset()
                done = trunc = False

            # Safety break
            if episodes >= max_episodes:
                break

    # Write out MP4 at requested fps
    out_path = Path(video_dir) / f"{name_prefix}.mp4"
    if frames:
        writer = imageio.get_writer(out_path, fps=fps)
        for fr in frames[:target_frames]:
            writer.append_data(fr)
        writer.close()
        print(f"🎞️ Saved {name_prefix} video to: {out_path} | frames={len(frames[:target_frames])} (~{len(frames[:target_frames])/fps:.1f}s)")
    else:
        print(f"⚠️ No frames captured for {name_prefix}.")
    eval_env.close()
    return str(out_path)

# ---- Training (brief) + capture videos ----
try:
    eps = EPSILON_START
    while True:
        frame_idx += 1
        eps = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, device, eps)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / max(1e-8, (time.time() - ts))
            m_reward = float(np.mean(total_rewards[-100:]))
            ts_frame, ts = frame_idx, time.time()
            writer.add_scalar('epsilon', eps, frame_idx)
            writer.add_scalar('speed', speed, frame_idx)
            writer.add_scalar('reward_100', m_reward, frame_idx)
            writer.add_scalar('reward', reward, frame_idx)
            if (best_m_reward is None) or (m_reward > best_m_reward + SAVE_EPSILON):
                save_model_if_best(m_reward)
                best_m_reward = m_reward if best_m_reward is None else max(best_m_reward, m_reward)
            if len(total_rewards) >= 20:  # keep runs short
                break
        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device)
        loss_t.backward()
        optimizer.step()

    # Record two clips ~20–25s each
    early_video = record_policy_video(net, epsilon=1.0, name_prefix='early', target_seconds=20, fps=60)
    later_video = record_policy_video(net, epsilon=0.05, name_prefix='later', target_seconds=25, fps=60)
    print("\nUpload these files to your GitHub repo and reference them in README.md:")
    print(f" - {early_video}")
    print(f" - {later_video}")

except KeyboardInterrupt:
    print("\n⏹️ Interrupted. Saving current weights…")
    ts_save = datetime.now().strftime('%Y%m%d-%H%M')
    fname = f"{safe_env_name}-current-{ts_save}-{model_comment}.dat"
    torch.save(net.state_dict(), os.path.join(save_dir_local, fname))
    if save_dir_drive:
        torch.save(net.state_dict(), os.path.join(save_dir_drive, fname))
    print("✅ Current model saved.")
finally:
    env.close()
    writer.close()
    print("Done.")
