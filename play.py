"""Xem mô hình PPO đã train điều khiển BipedalWalker.

Cách chạy:
    python play.py

Checkpoint mới (huấn luyện bằng maint1.py bản cập nhật) có lưu sẵn thống kê
chuẩn hóa observation (obs_rms) -> chạy thử khớp ngay, không cần warmup.
Checkpoint cũ (thiếu obs_rms) sẽ tự động dùng cơ chế warmup (kém chính xác hơn).
"""

import sys
import gymnasium as gym
import numpy as np
import torch
from ppot1 import PPO

# Console Windows mặc định cp1252 không in được tiếng Việt -> ép UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

device = torch.device("cpu")  # render: chạy CPU là đủ

OBS_CLIP = 10.0
MODEL_PATH = "ppo_bipedal_vec.pth"
NUM_EPISODES = 5  # số lượt xem
WARMUP_STEPS = 5000  # chỉ dùng khi checkpoint KHÔNG có obs_rms
DETERMINISTIC = True  # True: hành động trung bình (mượt); False: lấy mẫu ngẫu nhiên


def build_env(render=False):
    env = gym.make("BipedalWalker-v3", render_mode="human" if render else None)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    return env


@torch.no_grad()
def select_action(policy, obs):
    state = torch.FloatTensor(np.clip(obs, -OBS_CLIP, OBS_CLIP)).to(device)
    if DETERMINISTIC:
        action = policy.actor(state)  # hành động xác định = trung bình phân phối
    else:
        action, _, _, _ = policy.get_action_and_value(state)
    return action.cpu().numpy()


def main():
    tmp = build_env(False)
    obs_dim = tmp.observation_space.shape[0]
    act_dim = tmp.action_space.shape[0]
    tmp.close()

    agent = PPO(obs_dim, act_dim)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    agent.policy.load_state_dict(ckpt["policy_state_dict"])
    agent.policy.eval()
    print(f"[*] Đã nạp mô hình từ {MODEL_PATH}")

    env = build_env(True)

    if "obs_rms_mean" in ckpt:
        # Dùng thống kê chuẩn hóa đã lưu -> chính xác, không cần warmup
        env.obs_rms.mean = np.asarray(ckpt["obs_rms_mean"], dtype=np.float64)
        env.obs_rms.var = np.asarray(ckpt["obs_rms_var"], dtype=np.float64)
        env.obs_rms.count = float(ckpt["obs_rms_count"])
        env.update_running_mean = False
        print("[*] Đã nạp obs_rms từ checkpoint (chuẩn xác).")
    else:
        # Checkpoint cũ: hâm nóng obs normalization bằng env phụ rồi đóng băng
        print("[!] Checkpoint không có obs_rms -> dùng warmup (kém chính xác).")
        warm = build_env(False)
        obs, _ = warm.reset()
        for _ in range(WARMUP_STEPS):
            obs, _, te, tr, _ = warm.step(select_action(agent.policy, obs))
            if te or tr:
                obs, _ = warm.reset()
        env.obs_rms = warm.obs_rms
        env.update_running_mean = False
        warm.close()

    rewards = []
    for ep in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            obs, reward, term, trunc, _ = env.step(select_action(agent.policy, obs))
            total += float(reward)
            steps += 1
            done = term or trunc
        rewards.append(total)
        print(f"Episode {ep}: reward = {total:7.1f} | số bước = {steps}")

    print(f"\n[*] Trung bình {NUM_EPISODES} lượt: {np.mean(rewards):.1f}")
    env.close()


if __name__ == "__main__":
    main()
