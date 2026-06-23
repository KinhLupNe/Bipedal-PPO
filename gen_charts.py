# -*- coding: utf-8 -*-
import glob
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUTDIR = r"E:\nam4\ky2\AI\project\Báo cáo\files"

fs = glob.glob("runs/Bipedal_Vec/**/events*", recursive=True)
fs.sort(key=os.path.getmtime)
# size_guidance={'scalars': 0} -> nạp TOÀN BỘ điểm (không downsample ngẫu nhiên)
ea = EventAccumulator(fs[-1], size_guidance={"scalars": 0})
ea.Reload()
avg = ea.Scalars("Training/Avg_Reward_50_Eps")
ep = ea.Scalars("Training/Reward_Per_Episode")
ax = np.array([e.step for e in avg]) / 1e6
ay = np.array([e.value for e in avg])
ex = np.array([e.step for e in ep]) / 1e6
ey = np.array([e.value for e in ep])

# Fig 1: duong cong hoi tu
plt.figure(figsize=(7, 4))
plt.plot(ax, ay, color="#1f77b4", lw=1.8)
plt.axhline(0, color="gray", ls="--", lw=0.8)
plt.axhline(300, color="green", ls=":", lw=0.9, label="Mức giải (≈300)")
plt.xlabel("Số bước môi trường (triệu)")
plt.ylabel("Reward trung bình (50 episode)")
plt.title("Đường cong huấn luyện PPO – BipedalWalker-v3")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "figure_eval_train.png"), dpi=150)
plt.close()

# Fig 2: reward tung episode
plt.figure(figsize=(7, 4))
plt.scatter(ex, ey, s=4, alpha=0.25, color="#888888", label="Reward mỗi episode")
plt.plot(ax, ay, color="#d62728", lw=1.8, label="Trung bình 50 episode")
plt.axhline(0, color="gray", ls="--", lw=0.8)
plt.xlabel("Số bước môi trường (triệu)")
plt.ylabel("Reward mỗi episode")
plt.title("Phân bố reward theo episode trong huấn luyện")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "figure_eval_episodes.png"), dpi=150)
plt.close()


def first_cross(x, y, thr):
    idx = np.where(y >= thr)[0]
    return round(float(x[idx[0]]), 2) if len(idx) else None


print("points:", len(avg))
print("avg cuoi:", round(float(ay[-1]), 1), "| max avg:", round(float(ay.max()), 1),
      "tai", round(float(ax[ay.argmax()]), 2), "M")
print("vuot 0 tai ~", first_cross(ax, ay, 0), "M | vuot 200 tai ~", first_cross(ax, ay, 200), "M")
print("SAVED 2 FIGURES")
