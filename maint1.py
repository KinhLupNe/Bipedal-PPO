import sys
import gymnasium as gym
import torch
import numpy as np
import os
import functools
from collections import deque
from ppot1 import PPO
from torch.utils.tensorboard import SummaryWriter

# Console Windows mặc định cp1252 không in được tiếng Việt -> ép UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mạng nhỏ nên giới hạn thread torch để nhường core cho các tiến trình môi trường
torch.set_num_threads(4)

OBS_CLIP = 10.0  # clip observation sau khi normalize (chuẩn CleanRL)
REW_CLIP = 10.0  # clip reward sau khi normalize


def make_single_env(gamma):
    """Tạo MỘT môi trường con đầy đủ wrapper.

    Hàm ở cấp module (không dùng lambda/closure) để PICKLE được ->
    cần thiết cho AsyncVectorEnv trên Windows (multiprocessing spawn).
    """
    env = gym.make("BipedalWalker-v3")
    # RecordEpisodeStatistics đặt TRƯỚC normalize -> ghi lại REWARD THẬT
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    # NormalizeObservation/Reward được áp dụng ở CẤP VECTOR (xem main) để có
    # một obs_rms DUY NHẤT, dễ lưu lại cùng checkpoint cho lúc chạy thử.
    return env


def make_env(gamma):
    """Trả về một factory picklable (functools.partial) tạo môi trường con."""
    return functools.partial(make_single_env, gamma)


def main():
    # ----- Siêu tham số -----
    gamma = 0.99
    gae_lambda = 0.95
    base_lr = 3e-4
    num_envs = 8  # số môi trường chạy SONG SONG (CPU 12 core: 8 env + 4 thread torch)
    num_steps = 256  # số bước mỗi env trong một rollout
    batch_size = num_envs * num_steps  # = 2048 mẫu/lần cập nhật
    minibatch_size = 64
    total_timesteps = 3_000_000
    num_updates = total_timesteps // batch_size

    # ----- Tạo các môi trường song song -----
    # AsyncVectorEnv: mỗi env một tiến trình -> bước Box2D chạy SONG SONG trên nhiều core.
    # Nếu lỗi (vd môi trường Windows đặc thù) thì tự lùi về SyncVectorEnv (tuần tự).
    env_fns = [make_env(gamma) for _ in range(num_envs)]
    try:
        envs = gym.vector.AsyncVectorEnv(env_fns)
        print(f"[*] Dùng AsyncVectorEnv với {num_envs} tiến trình song song.")
    except Exception as e:
        print(f"[!] AsyncVectorEnv lỗi ({e}); lùi về SyncVectorEnv.")
        envs = gym.vector.SyncVectorEnv(env_fns)

    # Chuẩn hóa observation & reward ở CẤP VECTOR (một obs_rms duy nhất -> lưu được)
    envs = gym.wrappers.vector.NormalizeObservation(envs)
    obs_rms_env = envs  # giữ tham chiếu để lấy obs_rms khi lưu checkpoint
    envs = gym.wrappers.vector.NormalizeReward(envs, gamma=gamma)

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    ppo_agent = PPO(
        obs_dim,
        act_dim,
        lr=base_lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        minibatch_size=minibatch_size,
    )
    writer = SummaryWriter(log_dir="runs/Bipedal_Vec")
    model_path = "ppo_bipedal_vec.pth"

    # ----- (Tùy chọn) khôi phục từ checkpoint -----
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        ppo_agent.policy.load_state_dict(ckpt["policy_state_dict"])
        ppo_agent.policy_old.load_state_dict(ckpt["policy_state_dict"])
        if "optimizer_state_dict" in ckpt:
            ppo_agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "obs_rms_mean" in ckpt:
            obs_rms_env.obs_rms.mean = np.asarray(
                ckpt["obs_rms_mean"], dtype=np.float64
            )
            obs_rms_env.obs_rms.var = np.asarray(ckpt["obs_rms_var"], dtype=np.float64)
            obs_rms_env.obs_rms.count = float(ckpt["obs_rms_count"])
        print(f"[*] Đã khôi phục trạng thái huấn luyện từ {model_path}.")

    # ----- Bộ nhớ rollout dạng [num_steps, num_envs, ...] -----
    obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions = torch.zeros((num_steps, num_envs, act_dim), device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    global_step = 0
    recent_rewards = deque(maxlen=50)  # 50 episode gần nhất (reward THẬT)

    next_obs, _ = envs.reset(seed=0)
    next_obs = torch.FloatTensor(np.clip(next_obs, -OBS_CLIP, OBS_CLIP)).to(device)
    next_done = torch.zeros(num_envs, device=device)

    def save_checkpoint():
        # Lưu cả thống kê chuẩn hóa observation (obs_rms) để lúc chạy thử khớp
        torch.save(
            {
                "policy_state_dict": ppo_agent.policy.state_dict(),
                "optimizer_state_dict": ppo_agent.optimizer.state_dict(),
                # Lưu dạng tensor/float để torch.load(weights_only=True) nạp được
                "obs_rms_mean": torch.as_tensor(
                    obs_rms_env.obs_rms.mean, dtype=torch.float32
                ),
                "obs_rms_var": torch.as_tensor(
                    obs_rms_env.obs_rms.var, dtype=torch.float32
                ),
                "obs_rms_count": float(obs_rms_env.obs_rms.count),
            },
            model_path,
        )

    try:
        for update in range(1, num_updates + 1):
            # --- Anneal learning rate tuyến tính về 0 ---
            frac = 1.0 - (update - 1) / num_updates
            lrnow = frac * base_lr
            for pg in ppo_agent.optimizer.param_groups:
                pg["lr"] = lrnow

            # ===== 1) Thu thập rollout song song =====
            for step in range(num_steps):
                global_step += num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = (
                        ppo_agent.policy_old.get_action_and_value(next_obs)
                    )
                values[step] = value.reshape(-1)
                actions[step] = action
                logprobs[step] = logprob

                next_obs_np, reward, term, trunc, info = envs.step(action.cpu().numpy())
                done = np.logical_or(term, trunc)

                rewards[step] = torch.tensor(
                    np.clip(reward, -REW_CLIP, REW_CLIP),
                    dtype=torch.float32,
                    device=device,
                )
                next_obs = torch.FloatTensor(
                    np.clip(next_obs_np, -OBS_CLIP, OBS_CLIP)
                ).to(device)
                next_done = torch.tensor(done, dtype=torch.float32, device=device)

                # Ghi lại REWARD THẬT mỗi khi một env kết thúc episode
                if "episode" in info:
                    mask = info["_episode"]
                    for i in range(num_envs):
                        if mask[i]:
                            ep_r = float(info["episode"]["r"][i])
                            recent_rewards.append(ep_r)
                            writer.add_scalar(
                                "Training/Reward_Per_Episode", ep_r, global_step
                            )

            # ===== 2) Tính GAE trên toàn bộ [num_steps, num_envs] =====
            with torch.no_grad():
                next_value = ppo_agent.policy.get_value(next_obs).reshape(-1)
                advantages = torch.zeros_like(rewards).to(device)
                last_gae = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        next_val = next_value
                    else:
                        next_nonterminal = 1.0 - dones[t + 1]
                        next_val = values[t + 1]
                    delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
                    advantages[t] = last_gae = (
                        delta + gamma * gae_lambda * next_nonterminal * last_gae
                    )
                returns = advantages + values

            # ===== 3) Làm phẳng batch & tối ưu PPO =====
            b_obs = obs.reshape(-1, obs_dim)
            b_actions = actions.reshape(-1, act_dim)
            b_logprobs = logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            ppo_agent.optimize_batch(
                b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values
            )

            # ===== 4) Log & lưu checkpoint =====
            if recent_rewards:
                avg_reward = float(np.mean(recent_rewards))
                writer.add_scalar("Training/Avg_Reward_50_Eps", avg_reward, global_step)
                print(
                    f"Update {update:4d} | Steps {global_step:8d} | "
                    f"Avg(50 eps): {avg_reward:7.2f} | lr {lrnow:.2e}"
                )

            if update % 50 == 0:
                save_checkpoint()
                print(f"[*] Đã lưu checkpoint tại update {update}")

        print(f"\n[*] Đã hoàn thành huấn luyện {total_timesteps} bước!")

    except KeyboardInterrupt:
        print("\n\n[CẢNH BÁO] Đã nhận Ctrl+C. Đang lưu checkpoint an toàn...")
        save_checkpoint()
        print(f"[*] Đã lưu checkpoint vào: {model_path}")

    finally:
        envs.close()
        writer.close()


if __name__ == "__main__":
    main()
