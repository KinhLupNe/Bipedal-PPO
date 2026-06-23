import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim), # Bỏ Tanh() để tránh Vanishing Gradient
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Khởi tạo trọng số mạng (Orthogonal Init) - Tip rất mạnh của PPO
        self.apply(self._init_weights)
        # Áp dụng Gain nhỏ cho lớp diễn viên cuối để robot khởi đầu an toàn
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.414)
            nn.init.constant_(m.bias, 0.0)

    def get_value(self, state):
        # Tự động thêm chiều Batch nếu thiếu
        is_single = state.dim() == 1
        if is_single:
            state = state.unsqueeze(0)
        value = self.critic(state)
        # Gỡ chiều Batch ra nếu đầu vào là single
        return value.squeeze(0) if is_single else value

    def get_action_and_value(self, state, action=None):
        is_single = state.dim() == 1
        if is_single:
            state = state.unsqueeze(0)

        action_mean = self.actor(state)
        # Giới hạn log_std để tránh policy quá ngẫu nhiên hoặc quá tất định
        log_std = torch.clamp(self.actor_log_std, -5, 2)
        action_std = torch.exp(log_std.expand_as(action_mean))
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
        else:
            if is_single and action.dim() == 1:
                action = action.unsqueeze(0)

        log_prob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)
        value = self.critic(state)

        # Trả về kích thước chuẩn hóa không bị dư dấu ngoặc vuông
        if is_single:
            return (
                action.squeeze(0),
                log_prob.squeeze(0),
                entropy.squeeze(0),
                value.squeeze(0),
            )
        return action, log_prob, entropy, value


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4, # Chuẩn mực cho PPO
        gamma=0.99, # BipedalWalker cần tầm nhìn xa
        gae_lambda=0.95,
        epsilon_clip=0.2,
        entropy_coef=0.0,   # CleanRL dùng 0.0 cho continuous control
        vf_coef=0.5,        # hệ số value loss
        max_grad_norm=0.5,  # chặn nổ gradient
        clip_vloss=True,    # clip value loss như CleanRL
        target_kl=0.03,     # KL early stopping để tránh update quá đà
        K_epochs=10,
        minibatch_size=64,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.clip_vloss = clip_vloss
        self.target_kl = target_kl
        self.K_epochs = K_epochs
        self.minibatch_size = minibatch_size

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory, next_state_tensor, next_done):
        states = torch.stack(memory.states).to(device).detach()
        actions = torch.stack(memory.actions).to(device).detach()
        logprobs = torch.stack(memory.logprobs).to(device).detach()
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device).detach()
        dones = torch.tensor(memory.is_terminals).to(device).detach()

        with torch.no_grad():
            values = self.policy.get_value(states).squeeze()
            next_value = self.policy.get_value(next_state_tensor).squeeze()

            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0

            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - float(next_done)
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - float(dones[t])
                    next_val = values[t + 1]

                # ap dung bellman equation de vua khong nhieu, vua ko  lech
                delta = (
                    rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
                )
                advantages[t] = last_gae_lam = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                )

            returns = advantages + values

        # Tối ưu trên batch đã làm phẳng (dùng chung cho single-env và vector env)
        self.optimize_batch(states, actions, logprobs, advantages, returns, values)

    def optimize_batch(self, states, actions, logprobs, advantages, returns, values):
        """Chạy K epoch tối ưu PPO trên batch đã làm phẳng [batch, ...]."""
        batch_size = states.shape[0]
        for _ in range(self.K_epochs):
            # Xáo trộn thứ tự dữ liệu
            indices = torch.randperm(batch_size)
            approx_kl = 0.0

            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_logprobs = logprobs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_values = values[mb_indices]

                _, new_logprobs, entropy, new_values = self.policy.get_action_and_value(
                    mb_states, mb_actions
                )
                new_values = new_values.squeeze()

                logratio = new_logprobs - mb_logprobs
                ratio = torch.exp(logratio)

                # Ước lượng KL để theo dõi / early stopping (không tạo gradient)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()

                # Chuẩn hóa advantage theo TỪNG minibatch (chuẩn CleanRL)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss có clip (giảm dao động critic)
                if self.clip_vloss:
                    v_unclipped = (new_values - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        new_values - mb_values, -self.epsilon_clip, self.epsilon_clip
                    )
                    v_clipped = (v_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    value_loss = 0.5 * self.MseLoss(new_values, mb_returns)

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    - self.entropy_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient Clipping: bắt buộc với PPO để tránh nổ gradient
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

            # KL early stopping: nếu chính sách đổi quá nhiều thì dừng epoch sớm
            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        self.policy_old.load_state_dict(self.policy.state_dict())


class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.rewards, self.is_terminals = (
            [],
            [],
            [],
            [],
            [],
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
