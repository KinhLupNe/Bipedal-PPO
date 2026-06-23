import gymnasium as gym
import torch
import numpy as np
import os
from collections import deque
from ppot1 import PPO, RolloutBuffer
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    gamma = 0.99
    base_lr = 3e-4
    OBS_CLIP = 10.0   # clip observation sau khi normalize (chuẩn CleanRL)
    REW_CLIP = 10.0   # clip reward sau khi normalize

    # Dùng môi trường BipedalWalker NGUYÊN BẢN (reward gốc của Gym, không reward shaping)
    env = gym.make("BipedalWalker-v3")
    # RecordEpisodeStatistics đặt TRƯỚC NormalizeReward để ghi lại REWARD THẬT (không bị chuẩn hóa)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Khởi tạo thuật toán (Đã thêm Minibatch size)
    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr=base_lr,
        gamma=gamma,
        gae_lambda=0.95,
        minibatch_size=64,
    )
    memory = RolloutBuffer()
    writer = SummaryWriter(log_dir="runs/Bipedal_Final")

    max_episodes = 10000
    update_timestep = 2048          # chuẩn CleanRL cho continuous control
    total_timesteps = 3_000_000     # dùng để anneal learning rate
    num_updates = total_timesteps // update_timestep
    update_idx = 0
    time_step = 0
    model_path = "ppo_bipedal_final.pth"

    # 1. TẢI LẠI TRẠNG THÁI TỪ FILE CŨ (Bao gồm cả Optimizer)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        ppo_agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        # Đảm bảo tương thích ngược nếu checkpoint cũ không có optimizer
        if "policy_old_state_dict" in checkpoint:
            ppo_agent.policy_old.load_state_dict(checkpoint["policy_old_state_dict"])
        else:
            ppo_agent.policy_old.load_state_dict(checkpoint["policy_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            ppo_agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"[*] Đã khôi phục trạng thái huấn luyện từ {model_path}.")

    # Hàng đợi lưu điểm số 50 tập gần nhất để vẽ đồ thị trung bình
    recent_rewards = deque(maxlen=50)

    try:  # 2. BỌC TRONG KHỐI TRY ĐỂ BẮT SỰ KIỆN CTRL+C
        for episode in range(1, max_episodes + 1):
            state, _ = env.reset()
            state = np.clip(state, -OBS_CLIP, OBS_CLIP)
            episode_reward = 0.0

            while True:  # Để env tự quyết định điểm kết thúc (terminated/truncated)
                time_step += 1
                state_tensor = torch.FloatTensor(state).to(device)

                with torch.no_grad():
                    action, logprob, _, _ = ppo_agent.policy_old.get_action_and_value(
                        state_tensor
                    )

                action_np = action.cpu().numpy()
                next_state, reward, terminated, truncated, info = env.step(
                    np.clip(action_np, -1.0, 1.0)
                )
                done = terminated or truncated

                # Clip observation & reward (sau khi đã normalize) — chuẩn CleanRL
                next_state = np.clip(next_state, -OBS_CLIP, OBS_CLIP)
                reward_train = float(np.clip(float(reward), -REW_CLIP, REW_CLIP))

                memory.states.append(state_tensor)
                memory.actions.append(action)
                memory.logprobs.append(logprob)
                memory.rewards.append(reward_train)
                # Chỉ lưu 'terminated' cho GAE: khi truncated (hết giờ) vẫn phải bootstrap
                memory.is_terminals.append(terminated)

                state = next_state

                if time_step % update_timestep == 0:
                    # Anneal learning rate tuyến tính về 0
                    update_idx += 1
                    frac = max(0.0, 1.0 - (update_idx - 1) / num_updates)
                    for pg in ppo_agent.optimizer.param_groups:
                        pg["lr"] = frac * base_lr

                    next_state_tensor = torch.FloatTensor(next_state).to(device)
                    # bootstrap flag cho bước cuối rollout cũng chỉ dùng terminated
                    ppo_agent.update(memory, next_state_tensor, terminated)
                    memory.clear()

                if done:
                    # Lấy REWARD THẬT (trước normalize) từ RecordEpisodeStatistics
                    if "episode" in info:
                        episode_reward = float(info["episode"]["r"])
                    break

            # 3. VẼ ĐỒ THỊ TRÊN TENSORBOARD
            recent_rewards.append(episode_reward)
            avg_reward = np.mean(recent_rewards)

            writer.add_scalar("Training/Reward_Per_Episode", episode_reward, time_step)
            writer.add_scalar("Training/Avg_Reward_50_Eps", avg_reward, time_step)

            if episode % 10 == 0:
                print(
                    f"Episode {episode:4d} | Reward: {episode_reward:7.2f} | Avg (50 eps): {avg_reward:7.2f}"
                )

            # Lưu mô hình định kỳ (100 tập/lần)
            if episode % 100 == 0:
                checkpoint = {
                    "policy_state_dict": ppo_agent.policy.state_dict(),
                    "policy_old_state_dict": ppo_agent.policy_old.state_dict(),
                    "optimizer_state_dict": ppo_agent.optimizer.state_dict(),
                }
                torch.save(checkpoint, model_path)
                print(f"[*] Đã lưu Checkpoint định kỳ tại episode {episode}")

        print(f"\n[*] Đã hoàn thành huấn luyện {max_episodes} tập!")

    except KeyboardInterrupt:
        # 4. LƯU MÔ HÌNH KHI NHẤN CTRL + C
        print("\n\n[CẢNH BÁO] Đã nhận lệnh ngắt từ bàn phím (Ctrl+C).")
        print("Đang lưu lại Checkpoint an toàn để lần sau học tiếp...")
        checkpoint = {
            "policy_state_dict": ppo_agent.policy.state_dict(),
            "policy_old_state_dict": ppo_agent.policy_old.state_dict(),
            "optimizer_state_dict": ppo_agent.optimizer.state_dict(),
        }
        torch.save(checkpoint, model_path)
        print(f"[*] Đã lưu Checkpoint an toàn vào: {model_path}")

    finally:
        env.close()
        writer.close()


if __name__ == "__main__":
    main()
