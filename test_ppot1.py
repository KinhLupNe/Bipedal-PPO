import gymnasium as gym
import torch
import numpy as np
import os
import cv2  # Thêm thư viện xử lý ảnh OpenCV
from ppot1 import PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test():
    # Dùng môi trường BipedalWalker NGUYÊN BẢN (reward gốc, không reward shaping)
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    
    # Phải bọc giống hệt lúc train để normalize state
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.ClipAction(env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo_agent = PPO(state_dim, action_dim)
    model_path = "ppo_bipedal_final.pth"

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        ppo_agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"[*] Đã tải thành công mô hình từ {model_path}!")
    else:
        print(f"[LỖI] Không tìm thấy file {model_path}. Hãy chạy huấn luyện trước!")
        env.close()
        return

    ppo_agent.policy.eval()
    print(
        "[*] Đang chạy mô phỏng... Nhấn phím 'ESC' trên cửa sổ video hoặc Ctrl+C ở Terminal để thoát."
    )

    try:
        episode = 0
        while True:
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                with torch.no_grad():
                    action_mean = ppo_agent.policy.actor(state_tensor)

                action_np = action_mean.squeeze(0).cpu().numpy()
                next_state, reward, terminated, truncated, _ = env.step(
                    np.clip(action_np, -1.0, 1.0)
                )
                done = terminated or truncated

                # =================================================================
                # KHU VỰC RENDER VÀ VẼ HUD (Heads-Up Display)
                # =================================================================

                # 1. Lấy khung hình môi trường dưới dạng ma trận điểm ảnh (RGB)
                frame = env.render()

                # 2. Truy cập vào phần lõi (unwrapped) để lấy tọa độ X thực tế của phần thân (hull)
                distance_x = env.unwrapped.hull.position.x

                # 3. Chuyển hệ màu RGB của Gym sang BGR của OpenCV để hiển thị đúng màu
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 4. Ghi text lên màn hình (Text, Tọa độ, Font, Kích thước, Màu BGR, Độ dày)
                cv2.putText(
                    frame_bgr,
                    f"Khoang cach (X): {distance_x:.2f} m",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )  # Màu Xanh dương

                cv2.putText(
                    frame_bgr,
                    f"Diem thuong: {episode_reward:.2f}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )  # Màu Đỏ

                # 5. Hiển thị cửa sổ video
                cv2.imshow("Bipedal Walker AI - Test", frame_bgr)

                # 6. Đợi 1ms và bắt sự kiện phím bấm. Nếu nhấn ESC (mã 27) thì thoát.
                if cv2.waitKey(20) & 0xFF == 27:
                    print("\n[*] Người dùng nhấn ESC. Đang thoát...")
                    env.close()
                    cv2.destroyAllWindows()
                    return
                # =================================================================

                state = next_state
                episode_reward += float(reward)

            episode += 1
            print(
                f"Episode {episode} kết thúc với Reward: {episode_reward:.2f} | Khoảng cách: {distance_x:.2f}m"
            )

    except KeyboardInterrupt:
        print("\n[*] Đã nhận lệnh thoát (Ctrl+C). Đang đóng môi trường...")

    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test()
