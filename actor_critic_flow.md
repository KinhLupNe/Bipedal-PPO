# Luồng dữ liệu Actor-Critic trong `ppot1.py`

Tài liệu giải thích cách hàm `PPO.update()` hoạt động và mối quan hệ giữa **Actor** và **Critic**.

---

## Tổng quan

| Thành phần | Vai trò | Học từ loss |
|-----------|---------|-------------|
| **ACTOR** | "Người chơi" — quyết định hành động | `policy_loss` (+ `entropy_loss`) |
| **CRITIC** | "Người chấm điểm" — đoán giá trị của state | `value_loss` |

> Critic **không** cập nhật actor trực tiếp. Nó cung cấp `advantages` để actor biết hành động nào tốt/xấu. Cả hai mạng cùng được cập nhật trong một bước `optimizer.step()` vì chung `self.policy.parameters()`.

---

## 1. Giai đoạn THU THẬP dữ liệu (rollout — trước khi gọi `update`)

```
                    state (s_t)
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
     ┌─────────┐               ┌─────────┐
     │  ACTOR  │               │ CRITIC  │
     └────┬────┘               └────┬────┘
          │                         │
     action_mean + std         value V(s_t)
          │                         │
          ▼                         │
    Normal(mean, std)               │
          │                         │
     ┌────┴────┐                    │
   action   log_prob                │
     │         │                    │
     ▼         ▼                    ▼
 (môi trường)  └──────► lưu vào RolloutBuffer ◄──────┘
     │                  (states, actions, logprobs,
   reward, done ──────►  rewards, is_terminals)
```

---

## 2. `update` — Vòng for #1: Critic giúp tính Advantage (GAE)

> Chạy **ngược** từ bước cuối về đầu. Nằm trong `torch.no_grad()` → chỉ tính số liệu, không sinh gradient.

```
   rewards[t] ──┐
                ▼
   values[t]   delta = r_t + γ·V(s_{t+1})·(1-done) − V(s_t)    ← TD error
   (từ critic)  │                                               (dùng V của CRITIC)
                ▼
            GAE (cộng dồn ngược)
                │
        ┌───────┴────────┐
        ▼                ▼
   advantages         returns = advantages + values
   (cho ACTOR)        (mục tiêu cho CRITIC)
```

---

## 3. `update` — Vòng for #2: Huấn luyện (lặp `K_epochs` lần)

```
   states, actions (dữ liệu cũ)
          │
   ┌──────┴───────┐
   ▼              ▼
┌───────┐     ┌────────┐
│ ACTOR │     │ CRITIC │
└───┬───┘     └───┬────┘
    │             │
new_logprobs   new_values
entropy           │
    │             │
    ▼             │
ratio = exp(new_logprobs − logprobs)
    │             │
    ▼             │
clip + advantages │
    │             │
    ▼             ▼
policy_loss   value_loss = MSE(new_values, returns)
entropy_loss      │
    │             │
    └──────┬──────┘
           ▼
   loss = policy_loss + value_loss − coef·entropy_loss
           │
           ▼
   loss.backward()   →  gradient chảy về CẢ actor + critic
           │
           ▼
   optimizer.step()  →  cập nhật trọng số CẢ HAI mạng
```

---

## 4. Vòng lặp lớn (toàn cảnh)

```
   ┌─────────────────────────────────────────────────────────┐
   │                                                           │
   ▼                                                           │
[Thu thập] ──► [Tính Advantage] ──► [Huấn luyện] ──► [đồng bộ policy_old]
 (rollout)      (vòng for #1)        (vòng for #2)    load_state_dict
   ▲                                                           │
   └───────────────────────────────────────────────────────────┘
```

---

## Ghi chú nhanh

- `values` (vòng #1) nằm trong `no_grad` → chỉ là số liệu cố định để tính advantage.
- `new_values` (vòng #2) **có gradient** → đây mới là cái thực sự huấn luyện critic.
- `clamp` trong `policy_loss` giới hạn policy không thay đổi quá mạnh mỗi bước → ổn định (trái tim của PPO).
- Cuối cùng `policy_old.load_state_dict(policy.state_dict())` đồng bộ để chuẩn bị cho lần rollout kế tiếp.
