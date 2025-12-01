import math
import random
from typing import Tuple, List

import os  # <-- NEW
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------
# Environment: 2-context stochastic bandit
# ---------------------------------------------------------------------


def sample_step(
    theta: List[float],
    d0: float = 0.95,
    mu0: Tuple[float, float] = (0.0, 0.1),
    mu1: Tuple[float, float] = (0.0, 2.0),
    sigma0: float = 1.0,
    sigma1: float = 1.0,
) -> Tuple[int, int, float, float]:
    """
    Draw one (state, action, reward) sample.

    Returns:
        s: state index (0 or 1)
        a: action (0 or 1)
        r: reward
        p: πθ(a=1 | s) at this state
    """
    # sample context / state
    s = 0 if random.random() < d0 else 1

    # policy
    p = sigmoid(theta[s])
    a = 1 if random.random() < p else 0

    # reward
    if s == 0:
        mean = mu0[a]
        r = random.gauss(mean, sigma0)
    else:
        mean = mu1[a]
        r = random.gauss(mean, sigma1)

    return s, a, r, p


def batch_natural_gradient(
    theta: List[float],
    batch_size: int = 256,
    d0: float = 0.95,
) -> Tuple[np.ndarray, float]:
    """
    Monte-Carlo estimate of natural policy gradient for this batch.

    For Bernoulli with logit θ_s:
      ∂ log π / ∂θ_s = a - p
      FIM_s = p(1-p)
    Natural gradient sample for state s is:
      g_s ≈ (a - p) * r / FIM_s
    """
    grad = np.zeros(2, dtype=float)
    rewards = []

    for _ in range(batch_size):
        s, a, r, p = sample_step(theta, d0=d0)
        rewards.append(r)

        # Avoid division by zero when p is extremely close to 0 or 1
        denom = p * (1.0 - p)
        if denom < 1e-8:
            invF = 0.0
        else:
            invF = 1.0 / denom

        grad[s] += (a - p) * r * invF

    grad /= float(batch_size)
    avg_reward = float(np.mean(rewards))
    return grad, avg_reward


def eval_policy(
    theta: List[float],
    d0: float = 0.95,
    mu0: Tuple[float, float] = (0.0, 0.1),
    mu1: Tuple[float, float] = (0.0, 2.0),
) -> float:
    """
    Analytic expected reward under current policy (using reward means only).
    """
    p0 = sigmoid(theta[0])
    p1 = sigmoid(theta[1])

    # E[r | s] = (1 - p) * μ(s,0) + p * μ(s,1)
    er0 = (1.0 - p0) * mu0[0] + p0 * mu0[1]
    er1 = (1.0 - p1) * mu1[0] + p1 * mu1[1]
    return d0 * er0 + (1.0 - d0) * er1


def run_algorithm(
    momentum: bool,
    steps: int = 200,
    alpha: float = 0.05,
    beta: float = 0.9,
    batch_size: int = 256,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run original NPG (momentum=False) or NPG+momentum (momentum=True).

    Returns:
        values[t]: expected reward at iteration t
        probs[t, s]: πθ(a=1 | state s) at iteration t
    """
    random.seed(seed)
    np.random.seed(seed)

    # two logits: θ_0, θ_1
    theta = [0.0, 0.0]  # start at π=0.5 for both contexts
    momentum_vec = np.zeros(2, dtype=float)

    values = np.zeros(steps, dtype=float)
    probs = np.zeros((steps, 2), dtype=float)

    for t in range(steps):
        values[t] = eval_policy(theta)
        probs[t, 0] = sigmoid(theta[0])
        probs[t, 1] = sigmoid(theta[1])

        grad, _ = batch_natural_gradient(theta, batch_size=batch_size)

        if momentum:
            # heavy-ball momentum on natural gradient
            momentum_vec = beta * momentum_vec + grad
            theta[0] += alpha * momentum_vec[0]
            theta[1] += alpha * momentum_vec[1]
        else:
            # original method: plain NPG
            theta[0] += alpha * grad[0]
            theta[1] += alpha * grad[1]

    return values, probs


def experiment(
    steps: int = 200,
    seeds: int = 20,
    save_dir: str = "figs",
    save_name: str = "momentum_vs_plain.png",
) -> None:
    """
    Compare original NPG vs NPG + momentum over multiple random seeds,
    and save the figure into a folder.
    """
    avg_val_plain = np.zeros(steps, dtype=float)
    avg_val_mom = np.zeros(steps, dtype=float)

    for s in range(seeds):
        v_plain, _ = run_algorithm(momentum=False, steps=steps, seed=s)
        v_mom, _ = run_algorithm(momentum=True, steps=steps, seed=1000 + s)
        avg_val_plain += v_plain
        avg_val_mom += v_mom

    avg_val_plain /= float(seeds)
    avg_val_mom /= float(seeds)

    # Optimal value: always take a=1 in both contexts → θ → +∞
    opt_value = eval_policy([10.0, 10.0])  # 10 is “large enough” logit

    plt.figure()
    plt.plot(avg_val_plain, label="Original NPG")
    plt.plot(avg_val_mom, label="NPG + momentum")
    plt.axhline(opt_value, linestyle="--", label="Optimal value")
    plt.xlabel("Iteration")
    plt.ylabel("Expected reward")
    plt.title("Momentum NPG vs original NPG on skewed context distribution")
    plt.legend()
    plt.tight_layout()

    # --- NEW: make folder and save fig ---
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, save_name)
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")

    # Optional: still show it on screen
    plt.show()


if __name__ == "__main__":
    experiment()
