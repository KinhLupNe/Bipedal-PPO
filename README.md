# BipedalWalker-v3 Solved with PPO

A PyTorch implementation of Proximal Policy Optimization (PPO) to train a bipedal robot to walk in the `BipedalWalker-v3` Gymnasium environment.

## 🎥 Demo & Results

**Trained Agent in Action:**
![BipedalWalker Demo](https://github.com/user-attachments/assets/e5be1518-9371-4bbe-9d89-24631829c603)

**Training Convergence (TensorBoard):**
![Training Curve](https://github.com/user-attachments/assets/4ef4726a-3997-4f4a-be31-851d1df115cc)

## ✨ Key Features
- **Custom PPO from scratch:** Implemented entirely in PyTorch.
- **GAE (Generalized Advantage Estimation):** Smooth advantage calculation ($\lambda = 0.95$).
- **Minibatch SGD Updates:** Stable policy updates using shuffled mini-batches.
- **Modern RL Tricks:** Incorporates Orthogonal Weight Initialization and Observation Normalization for fast convergence.
- **Real-time HUD:** Custom OpenCV rendering to display metrics during evaluation.

## 🛠️ Installation
Install the required dependencies:
```bash
pip install torch numpy gymnasium[box2d] opencv-python tensorboard
```

## 🚀 Usage

### 1. Training
To train the agent from scratch, run:
```bash
python maint1.py
```
*Note: The model is automatically saved to `ppo_bipedal_final.pth` every 100 episodes, or instantly if you interrupt the process with `Ctrl+C`.*

### 2. Testing / Evaluation
To watch the trained agent walk (with custom OpenCV HUD metrics):
```bash
python test_ppot1.py
```

### 3. Monitoring (TensorBoard)
To view the training convergence curves and reward metrics:
```bash
tensorboard --logdir=runs
```
Then navigate to `http://localhost:6006` in your web browser.

## 📊 Hyperparameters
- **Learning Rate:** 3e-4
- **Gamma ($\gamma$):** 0.99
- **GAE Lambda ($\lambda$):** 0.95
- **Clip Range ($\epsilon$):** 0.2
- **K Epochs:** 10
- **Minibatch Size:** 64
- **Update Timestep:** 4000
