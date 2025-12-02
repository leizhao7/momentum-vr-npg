#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Hunter–Prey Markov Game (grid_size x grid_size, slip on prey)
# ============================================================

class HunterPreyGame:
    """
    Two-player zero-sum Markov game.

    - Grid: grid_size x grid_size
    - State: (h_r, h_c, p_r, p_c)
    - Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
    - Hunter (max player) wants to capture the prey.
    - Prey (min player) moves but sometimes slips and stays.
    - Reward to hunter:
        +10 on capture, -0.1 per time step otherwise.

    In THIS version, we always treat the state in canonical form:
      hunter is at (0,0), prey anywhere in {0,1}×{0,1}.
    After every step we canonicalize the next state.
    """

    def __init__(self, grid_size=2, slip_prob=0.25, gamma=0.9):
        self.grid_size = grid_size
        self.slip_prob = slip_prob
        self.gamma = gamma
        self.actions = [0, 1, 2, 3, 4]   # stay, up, down, left, right
        self.n_actions = len(self.actions)

    # ----- basic environment dynamics -----

    def reset(self):
        """Fixed canonical start state: hunter (0,0), prey bottom-right."""
        g = self.grid_size - 1
        return (0, 0, g, g)  # already canonical

    def _move(self, r, c, a):
        nr, nc = r, c
        if a == 1: nr -= 1
        elif a == 2: nr += 1
        elif a == 3: nc -= 1
        elif a == 4: nc += 1
        nr = max(0, min(self.grid_size - 1, nr))
        nc = max(0, min(self.grid_size - 1, nc))
        return nr, nc

    def step(self, state, a_h, a_p):
        """
        One environment step, then canonicalize.

        Input/Output states are always treated **canonically**:
          hunter effectively at (0,0) after canonicalization.
        """
        h_r, h_c, p_r, p_c = state

        # prey slips to "stay" with slip_prob
        real_a_p = a_p
        if np.random.rand() < self.slip_prob:
            real_a_p = 0

        nh_r, nh_c = self._move(h_r, h_c, a_h)
        np_r, np_c = self._move(p_r, p_c, real_a_p)

        caught = (nh_r == np_r and nh_c == np_c)
        swapped = (nh_r == p_r and nh_c == p_c and
                   np_r == h_r and np_c == h_c)

        if caught or swapped:
            reward = 10.0
            next_state_raw = self.reset()
            done = True
        else:
            reward = -0.1
            next_state_raw = (nh_r, nh_c, np_r, np_c)
            done = False

        # canonicalize the next state
        canonical_next, _ = canonicalize_state(next_state_raw, self.grid_size)
        return canonical_next, reward, done


# ============================================================
# 1.5 Rotational canonicalization (compress 16 states -> 4)
# ============================================================

def rotate_coords(r, c, rot, grid_size):
    """
    Rotate coordinates (r, c) around the grid center by:
      rot = 0: 0 degrees (identity)
      rot = 1: 90 degrees clockwise
      rot = 2: 180 degrees
      rot = 3: 270 degrees clockwise
    """
    G = grid_size
    if rot == 0:
        return r, c
    elif rot == 1:  # 90° clockwise
        return c, G - 1 - r
    elif rot == 2:  # 180°
        return G - 1 - r, G - 1 - c
    elif rot == 3:  # 270° clockwise
        return G - 1 - c, r
    else:
        raise ValueError("rot must be in {0,1,2,3}")


def canonicalize_state(state, grid_size=2):
    """
    Rotate the full state so that the hunter is always at (0,0).

    Input state: (h_r, h_c, p_r, p_c) in original coordinates.
    Output:
      canonical_state: (0, 0, p_r', p_c')
      rot: which rotation (0,1,2,3) was used.
    """
    h_r, h_c, p_r, p_c = state

    for rot in range(4):
        rh, ch = rotate_coords(h_r, h_c, rot, grid_size)
        if rh == 0 and ch == 0:
            rp, cp = rotate_coords(p_r, p_c, rot, grid_size)
            return (0, 0, rp, cp), rot

    raise ValueError(f"Cannot canonicalize state {state} for grid_size={grid_size}")


# ============================================================
# 2. State features for log-linear function approximation
# ============================================================

def state_features(state, grid_size=2):
    """
    One-hot encoding of the *canonical* state.

    We rotate the board so that the hunter is always at (0,0),
    and then we only encode the prey position. On a 2x2 grid,
    this gives only grid_size^2 = 4 distinct canonical states.

      Canonical states (hunter fixed at (0,0)):
        prey at (0,0) -> index 0
        prey at (0,1) -> index 1
        prey at (1,0) -> index 2
        prey at (1,1) -> index 3
    """
    canonical_state, _ = canonicalize_state(state, grid_size=grid_size)
    h_r, h_c, p_r, p_c = canonical_state

    assert h_r == 0 and h_c == 0  # must be canonical

    idx = p_r * grid_size + p_c  # 0..3 for a 2x2 grid
    total_states = grid_size ** 2

    features = np.zeros(total_states, dtype=np.float32)
    features[idx] = 1.0
    return features


# ============================================================
# 3. Log-linear policy class (for both players)
# ============================================================

class LogLinearPolicy:
    """
    π_θ(a | s) ∝ exp( θ_a^T φ(s) ), where θ_a is the row for action a.

    We store θ as (n_actions, f_dim). All gradients are in the same shape.
    """

    def __init__(self, n_actions, f_dim, grid_size=2, temperature=1.0):
        self.n_actions = n_actions
        self.f_dim = f_dim
        self.theta = np.zeros((n_actions, f_dim), dtype=np.float32)
        self.temperature = temperature
        self.grid_size = grid_size

    # ----- policy evaluation -----

    def logits(self, state):
        phi = state_features(state, grid_size=self.grid_size)
        return (self.theta @ phi) / self.temperature

    def probs(self, state):
        logits = self.logits(state)
        logits -= np.max(logits)
        exps = np.exp(logits)
        return exps / np.sum(exps)

    def sample(self, state):
        p = self.probs(state)
        return np.random.choice(self.n_actions, p=p)

    # ----- gradient of log π -----

    def grad_log(self, state, action):
        """
        Gradient ∇_θ log π(a | s) as a tensor of same shape as θ.

        For each action a':
          grad[a'] = (1_{a'=action} - π(a'|s)) * φ(s)
        """
        phi = state_features(state, grid_size=self.grid_size)
        p = self.probs(state)
        grad = np.zeros_like(self.theta)

        for a in range(self.n_actions):
            coeff = (1.0 if a == action else 0.0) - p[a]
            grad[a, :] = coeff * phi

        return grad


# ============================================================
# 4. Value approximation (linear in φ(s))
# ============================================================

def V_value(v_param, state, grid_size=2):
    """Linear value function V(s) = v^T φ(s)."""
    if v_param is None:
        return 0.0
    return float(v_param @ state_features(state, grid_size=grid_size))


# ============================================================
# 5. Episodic sampling oracle (for ν0, ν_{x,f}, and Q-hat)
# ============================================================

def sample_state_uniform(grid_size):
    """
    σ(s): uniform over all *canonical* states:
      (0,0,p_r,p_c) for p_r,p_c in {0,1}.
    """
    states = []
    for p_r in range(grid_size):
        for p_c in range(grid_size):
            states.append((0, 0, p_r, p_c))
    idx = np.random.randint(len(states))
    return states[idx]

def sample_nu0(env):
    """
    ν0(s,a,b) = σ(s) / |A|^2 with σ uniform over canonical states.
    """
    s = sample_state_uniform(env.grid_size)
    a = np.random.randint(env.n_actions)
    b = np.random.randint(env.n_actions)
    return s, a, b

def sample_from_nu_t(env, x_policy, f_policy):
    s, a, b = sample_nu0(env)
    while True:
        # geometric stopping with parameter 1-γ
        if np.random.rand() < (1.0 - env.gamma):
            return s, a, b

        s_next, r, done = env.step(s, a, b)
        # ignore `done` – env.step already reset() and canonicalizes
        a = x_policy.sample(s_next)
        b = f_policy.sample(s_next)
        s = s_next

def oracle_Q(env, x_policy, f_policy, s, a, b):
    """
    Unbiased Q estimate for the continuing discounted game
    using a geometric stopping time with parameter (1 - gamma).
    """
    G = 0.0
    state = s
    a_curr = a
    b_curr = b
    
    while True:
        next_state, r, done = env.step(state, a_curr, b_curr)
        G += r  # no explicit discount; handled by geometric horizon

        # Geometric stopping with parameter (1 - gamma)
        if np.random.rand() < (1.0 - env.gamma):
            break

        state = next_state
        a_curr = x_policy.sample(state)
        b_curr = f_policy.sample(state)
        
    return G
    

# ============================================================
# 5.5 Tabular analyzer: exact Nash + exploitability (4 canonical states)
# ============================================================

class TabularAnalyzer:
    """
    Exact tabular computations for the 4-state canonical Markov game.
    """

    def __init__(self, env: HunterPreyGame):
        self.env = env
        self.grid_size = env.grid_size
        self.actions = env.actions
        self.n_actions = len(self.actions)

        # enumerate ONLY canonical states
        self.states = []
        self.state_to_idx = {}
        for p_r in range(self.grid_size):
            for p_c in range(self.grid_size):
                s = (0, 0, p_r, p_c)
                idx = len(self.states)
                self.states.append(s)
                self.state_to_idx[s] = idx

        self.n_states = len(self.states)
        self.start_idx = self.state_to_idx[env.reset()]  # reset is canonical

    # ---------- environment expectation helpers ----------

    def expected_transition(self, s_idx, a_h, a_p):
        """
        Deterministic expectation over prey slip.
        Returns a list of (prob, next_state_idx, reward)
        using the SAME canonicalization convention as env.step.
        """
        env = self.env
        s = self.states[s_idx]
        h_r, h_c, p_r, p_c = s

        nh_r, nh_c = env._move(h_r, h_c, a_h)

        res = []

        def branch(prob, effective_a_p):
            np_r, np_c = env._move(p_r, p_c, effective_a_p)
            caught = (nh_r == np_r and nh_c == np_c)
            swapped = (
                nh_r == p_r and nh_c == p_c and
                np_r == h_r and np_c == h_c
            )
            if caught or swapped:
                reward = 10.0
                next_state_raw = env.reset()  # canonical start
            else:
                reward = -0.1
                next_state_raw = (nh_r, nh_c, np_r, np_c)

            # canonicalize next_state
            canonical_next, _ = canonicalize_state(next_state_raw, env.grid_size)
            next_idx = self.state_to_idx[canonical_next]
            res.append((prob, next_idx, reward))

        prob_norm = 1.0 - env.slip_prob
        prob_slip = env.slip_prob

        branch(prob_norm, a_p)   # normal prey move
        branch(prob_slip, 0)     # slip -> stay

        return res

    def expected_Q_one_step(self, s_idx, a_h, a_p, V):
        total = 0.0
        gamma = self.env.gamma
        for prob, next_idx, reward in self.expected_transition(s_idx, a_h, a_p):
            total += prob * (reward + gamma * V[next_idx])
        return total

    # ---------- static zero-sum matrix game solver ----------

    def _solve_row_player_lp(self, A):
        """
        Solve max_p min_q p^T A q via LP, returning optimal row strategy p
        and game value v.
        """
        try:
            from scipy.optimize import linprog
        except ImportError as e:
            raise ImportError(
                "scipy is required for exact LP Nash solving. "
                "Install with `pip install scipy`."
            ) from e

        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape

        # decision variables: x = [p_0, ..., p_{m-1}, v]
        # objective: minimize c^T x = -v
        c = np.zeros(m + 1)
        c[-1] = -1.0

        # inequality constraints: -A^T p + v <= 0  (one per column)
        A_ub = np.zeros((n, m + 1))
        b_ub = np.zeros(n)
        for j in range(n):
            A_ub[j, :m] = -A[:, j]
            A_ub[j, m] = 1.0
            b_ub[j] = 0.0

        # equality: sum_i p_i = 1
        A_eq = np.zeros((1, m + 1))
        A_eq[0, :m] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, 1.0)] * m + [(None, None)]  # p_i in [0,1], v free

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            raise RuntimeError(f"LP for zero-sum game failed: {res.message}")

        x_opt = res.x
        p = x_opt[:m]
        v = x_opt[m]
        p = np.maximum(p, 0.0)
        s = p.sum()
        if s > 0:
            p /= s
        return p, v

    def solve_matrix_game(self, Q):
        """
        Solve the 5x5 zero-sum matrix game with payoff matrix Q
        for the row player (hunter).

        Returns:
          p_h: row player mixed strategy (size n_actions)
          p_p: column player mixed strategy (size n_actions)
          v  : game value
        """
        p_h, v = self._solve_row_player_lp(Q)
        p_p, v_dual = self._solve_row_player_lp(-Q.T)
        return p_h, p_p, v

    # ---------- Nash value only ----------

    def solve_nash_value(self, tol=1e-6, max_iter=10000):
        nS = self.n_states
        nA = self.n_actions
        V = np.zeros(nS, dtype=np.float64)

        for _ in range(max_iter):
            V_new = np.zeros_like(V)
            delta = 0.0
            for s_idx in range(nS):
                Q = np.zeros((nA, nA))
                for i, a in enumerate(self.actions):
                    for j, b in enumerate(self.actions):
                        Q[i, j] = self.expected_Q_one_step(s_idx, a, b, V)

                _, _, v = self.solve_matrix_game(Q)
                V_new[s_idx] = v
                delta = max(delta, abs(V_new[s_idx] - V[s_idx]))
            V = V_new
            if delta < tol:
                break
        return V

    # ---------- Nash value + Nash policies, with printing ----------

    def solve_nash_equilibrium(self, tol=1e-6, max_iter=10000, print_all=False):
        """
        Full Nash solution on the 4 canonical states.
        """
        nS = self.n_states
        nA = self.n_actions

        V_star = self.solve_nash_value(tol=tol, max_iter=max_iter)

        pi_hunter = np.zeros((nS, nA), dtype=np.float64)
        pi_prey   = np.zeros((nS, nA), dtype=np.float64)
        state_values = np.zeros(nS, dtype=np.float64)

        for s_idx in range(nS):
            Q = np.zeros((nA, nA), dtype=np.float64)
            for i, a in enumerate(self.actions):
                for j, b in enumerate(self.actions):
                    Q[i, j] = self.expected_Q_one_step(s_idx, a, b, V_star)

            p_h, p_p, v = self.solve_matrix_game(Q)
            pi_hunter[s_idx, :] = p_h
            pi_prey[s_idx, :]   = p_p
            state_values[s_idx] = v

            if print_all:
                print(f"\nState {s_idx} {self.states[s_idx]}:")
                print("  Hunter π_h:", p_h)
                print("  Prey   π_p:", p_p)
                print("  Stage-game value:", v)

        s0_idx = self.start_idx
        print("\n=== Exact Nash Equilibrium (canonical 4-state game, LP) ===")
        print(f"Start state: index {s0_idx}, state {self.states[s0_idx]}")
        print(f"Game value V*(start) ≈ {V_star[s0_idx]:.6f}")
        print("Hunter equilibrium π_h(· | start):", pi_hunter[s0_idx])
        print("Prey   equilibrium π_p(· | start):", pi_prey[s0_idx])

        return V_star, pi_hunter, pi_prey

    # ---------- other utilities ----------

    def best_response_value(self, x_policy, tol=1e-6, max_iter=10000):
        """
        Compute V_{x, f_BR(x)} via value iteration where prey plays
        a per-state best response against fixed hunter policy x.
        """
        nS = self.n_states
        nA = self.n_actions
        V = np.zeros(nS, dtype=np.float64)

        for _ in range(max_iter):
            V_new = np.zeros_like(V)
            delta = 0.0
            for s_idx, s in enumerate(self.states):
                p_a = x_policy.probs(s)
                Qb = np.zeros(nA)
                for j, b in enumerate(self.actions):
                    val = 0.0
                    for i, a in enumerate(self.actions):
                        val += p_a[i] * self.expected_Q_one_step(s_idx, a, b, V)
                    Qb[j] = val
                V_new[s_idx] = Qb.min()
                delta = max(delta, abs(V_new[s_idx] - V[s_idx]))
            V = V_new
            if delta < tol:
                break
        return V

    def value_of_pair(self, x_policy, f_policy):
        """
        Exact V_{x,f} by solving (I - γP)V = r for the 4-state Markov chain.
        """
        nS = self.n_states
        nA = self.n_actions
        env = self.env

        R = np.zeros(nS, dtype=np.float64)
        P = np.zeros((nS, nS), dtype=np.float64)

        for s_idx, s in enumerate(self.states):
            p_a = x_policy.probs(s)
            p_b = f_policy.probs(s)
            for i, a in enumerate(self.actions):
                for j, b in enumerate(self.actions):
                    weight = p_a[i] * p_b[j]
                    for prob, next_idx, reward in self.expected_transition(s_idx, a, b):
                        R[s_idx] += weight * prob * reward
                        P[s_idx, next_idx] += weight * prob

        I = np.eye(nS)
        V = np.linalg.solve(I - env.gamma * P, R)
        return V


# ============================================================
# 6. Algorithm 4 – Online Greedy Step with Function Approximation
# ============================================================

def greedy_step(
    env,
    v_prev,
    init_max_theta=None,
    init_min_theta=None,
    T_greedy=10,
    N_greedy=50,
    eta=0.05,
    alpha=0.05,
    W_radius=10.0,
    momentum_max=0.0,   # <<< NEW: momentum for hunter
    momentum_min=0.0,   # <<< NEW: momentum for prey (optional)
):
    """
    Algorithm 4 with Warm Start capability.
    Returns the FINAL policies (max_pol, min_pol) to allow continuous learning.

    momentum_max: momentum coefficient for hunter (e.g. 0.9)
    momentum_min: momentum coefficient for prey  (usually 0.0 here)
    """
    f_dim = len(state_features(env.reset(), grid_size=env.grid_size))
    
    # Initialize policies with PASSED parameters (Warm Start)
    min_pol = LogLinearPolicy(env.n_actions, f_dim, grid_size=env.grid_size)
    if init_min_theta is not None:
        min_pol.theta = init_min_theta.copy()
    
    max_pol = LogLinearPolicy(env.n_actions, f_dim, grid_size=env.grid_size)
    if init_max_theta is not None:
        max_pol.theta = init_max_theta.copy()

    # <<< NEW: momentum buffers
    v_min_mom = np.zeros_like(min_pol.theta)
    v_max_mom = np.zeros_like(max_pol.theta)

    for t in range(T_greedy):
        # ---------- Min-player subproblem ----------
        w_min = np.zeros_like(min_pol.theta)
        w_min_acc = np.zeros_like(min_pol.theta)

        for n in range(N_greedy):
            s = sample_state_uniform(env.grid_size)
            a = max_pol.sample(s)
            b = min_pol.sample(s)
            s_next, r, done = env.step(s, a, b)
            y_n = r + env.gamma * V_value(v_prev, s_next, grid_size=env.grid_size)

            grad_b = min_pol.grad_log(s, b)
            b_prime = min_pol.sample(s)
            grad_bprime = min_pol.grad_log(s, b_prime)
            g_n = y_n * (grad_b - grad_bprime)

            inner = np.sum(w_min * grad_b)
            Fw = inner * grad_b
            w_min = w_min - 2.0 * alpha * (Fw - g_n)

            norm = np.linalg.norm(w_min)
            if norm > W_radius:
                w_min *= W_radius / norm

            w_min_acc += w_min

        w_hat_min = w_min_acc / N_greedy

        # <<< NEW: apply momentum to prey (often zero momentum_min)
        v_min_mom = momentum_min * v_min_mom + (1.0 - momentum_min) * w_hat_min
        min_pol.theta -= eta * v_min_mom

        # ---------- Max-player subproblem ----------
        w_max = np.zeros_like(max_pol.theta)
        w_max_acc = np.zeros_like(max_pol.theta)

        for n in range(N_greedy):
            s = sample_state_uniform(env.grid_size)
            a = max_pol.sample(s)
            b = min_pol.sample(s)
            s_next, r, done = env.step(s, a, b)
            y_n = r + env.gamma * V_value(v_prev, s_next, grid_size=env.grid_size)

            grad_a = max_pol.grad_log(s, a)
            a_prime = max_pol.sample(s)
            grad_aprime = max_pol.grad_log(s, a_prime)
            g_n = y_n * (grad_a - grad_aprime)

            inner = np.sum(w_max * grad_a)
            Fw = inner * grad_a
            w_max = w_max - 2.0 * alpha * (Fw - g_n)

            norm = np.linalg.norm(w_max)
            if norm > W_radius:
                w_max *= W_radius / norm

            w_max_acc += w_max

        w_hat_max = w_max_acc / N_greedy

        # <<< NEW: apply momentum to hunter
        v_max_mom = momentum_max * v_max_mom + (1.0 - momentum_max) * w_hat_max
        max_pol.theta += eta * v_max_mom

    return max_pol, min_pol


# ============================================================
# 7. Algorithm 3 – Iteration step and outer loop
# ============================================================

def iteration_step(
    env,
    x_policy,          # fixed max-player policy x_k
    T_iter=10,
    N_iter=50,
    eta=0.05,
    alpha=0.05,
    W_radius=10.0,
    v_init=None,
    value_episodes=50,
    value_alpha=0.002,
):
    """
    Algorithm 3 Iteration Step (with function approximation).
    """
    f_dim = len(state_features(env.reset(), grid_size=env.grid_size))
    min_policy = LogLinearPolicy(env.n_actions, f_dim, grid_size=env.grid_size)
    theta_hist = [min_policy.theta.copy()]

    # ----- NPG updates for min player -----
    for t in range(T_iter):
        w = np.zeros_like(min_policy.theta)
        w_acc = np.zeros_like(min_policy.theta)

        for n in range(N_iter):
            s, a, b = sample_from_nu_t(env, x_policy, min_policy)
            Q_hat = oracle_Q(env, x_policy, min_policy, s, a, b)

            grad_b = min_policy.grad_log(s, b)
            b_prime = min_policy.sample(s)
            grad_bprime = min_policy.grad_log(s, b_prime)
            grad_diff = grad_b - grad_bprime

            g_n = Q_hat * grad_diff

            inner = np.sum(w * grad_b)
            Fw = inner * grad_b

            w = w - 2.0 * alpha * (Fw - g_n)

            norm = np.linalg.norm(w)
            if norm > W_radius:
                w *= W_radius / norm

            w_acc += w

        w_hat = w_acc / N_iter
        min_policy.theta -= eta * w_hat
        theta_hist.append(min_policy.theta.copy())

    # choose f^{(k)} uniformly from {f_t} (t = 0 .. T-1)
    idx = np.random.randint(len(theta_hist) - 1)
    f_policy = LogLinearPolicy(env.n_actions, f_dim, grid_size=env.grid_size)
    f_policy.theta = theta_hist[idx].copy()

    # ----- Fit value function V_k under (x_policy, f_policy) -----
    f_dim_v = len(state_features(env.reset(), grid_size=env.grid_size))
    if v_init is None:
        v_param = np.zeros(f_dim_v, dtype=np.float32)
    else:
        v_param = v_init.copy()

    for _ in range(value_episodes):
        s = env.reset()
        states = []
        rewards = []

        # simulate one episode under (x_policy, f_policy)
        for _ in range(50):
            states.append(s)
            a = x_policy.sample(s)
            b = f_policy.sample(s)
            s_next, r, done = env.step(s, a, b)
            rewards.append(r)
            s = s_next  # already canonical

        # Monte Carlo returns
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + env.gamma * G
            returns.insert(0, G)

        # SGD for v_param
        for st, Gt in zip(states, returns):
            psi = state_features(st, grid_size=env.grid_size)
            pred_v = float(v_param @ psi)
            grad_v = 2.0 * (pred_v - Gt) * psi
            v_param -= value_alpha * grad_v

    return f_policy, v_param


def two_player_npg(
    env,
    analyzer: TabularAnalyzer,
    V_star_tabular,
    K=50,
    T_greedy=10,
    N_greedy=50,
    T_iter=10,
    N_iter=50,
    eta_g=0.03,
    alpha_g=0.03,
    eta_i=0.03,
    alpha_i=0.03,
    W_radius=10.0,
    polyak_tau=0.1,
    momentum_max=0.0,   # <<< NEW: pass momentum for hunter
    momentum_min=0.0,   # <<< NEW: pass momentum for prey
):
    f_dim_v = len(state_features(env.reset(), grid_size=env.grid_size))
    
    # === PERSISTENT STATE (Warm Start) ===
    v_param = np.zeros(f_dim_v, dtype=np.float32)
    theta_max = np.zeros((env.n_actions, f_dim_v), dtype=np.float32)
    theta_min = np.zeros((env.n_actions, f_dim_v), dtype=np.float32)

    history = {
        "V_start_hat": [],
        "x_probs": [],
        "BR_start": [],
        "nash_gap": [],
    }
    x_policy = None

    start_idx = analyzer.start_idx
    V_star_start = V_star_tabular[start_idx]
    start_state = env.reset()  # canonical (0,0,1,1)

    print(f"True Nash game value at start (canonical 4-state, LP) V* ≈ {V_star_start:.4f}\n")

    for k in range(1, K + 1):
        # 1. Greedy Step (Hunter)
        max_pol_new, min_pol_new = greedy_step(
            env,
            v_prev=v_param,
            init_max_theta=theta_max, 
            init_min_theta=theta_min,
            T_greedy=T_greedy,
            N_greedy=N_greedy,
            eta=eta_g,
            alpha=alpha_g,
            W_radius=W_radius,
            momentum_max=momentum_max,  # <<< NEW
            momentum_min=momentum_min,  # <<< NEW
        )
        
        theta_max = max_pol_new.theta.copy()
        theta_min = min_pol_new.theta.copy()
        x_k = max_pol_new 

        # 2. Iteration Step (Prey)
        f_k, v_param_new = iteration_step(
            env,
            x_k,
            T_iter=T_iter,
            N_iter=N_iter,
            eta=eta_i,
            alpha=alpha_i,
            W_radius=W_radius,
            v_init=v_param, 
            value_episodes=40,
            value_alpha=0.1,  # High alpha for one-hot
        )

        # 3. Polyak Averaging
        v_param = (1.0 - polyak_tau) * v_param + polyak_tau * v_param_new

        # --- Logging ---
        V_start_hat = V_value(v_param, start_state, grid_size=env.grid_size)
        V_br = analyzer.best_response_value(x_k)
        BR_start = V_br[start_idx]
        gap = V_star_start - BR_start

        history["V_start_hat"].append(V_start_hat)
        history["x_probs"].append(x_k.probs(start_state))
        history["BR_start"].append(BR_start)
        history["nash_gap"].append(gap)
        x_policy = x_k

        # --- Standard Metric Print ---
        print(
            f"Iter {k:3d}: "
            f"V_hat={V_start_hat:5.2f} | "
            f"Gap={gap:6.3f} | "
        )

        # Print policy details at start state
        h_p = x_k.probs(start_state)
        f_p = f_k.probs(start_state)

        print(f"   Hunter: S:{h_p[0]:.2f} U:{h_p[1]:.2f} D:{h_p[2]:.2f} L:{h_p[3]:.2f} R:{h_p[4]:.2f}")
        print(f"   Prey:   S:{f_p[0]:.2f} U:{f_p[1]:.2f} D:{f_p[2]:.2f} L:{f_p[3]:.2f} R:{f_p[4]:.2f}")
        print("-" * 60)

    # Return f_k as well (the prey policy from the last iteration)
    return x_policy, f_k, v_param, history


# ============================================================
# 8. Evaluation + main
# ============================================================

def evaluate(env, x_policy, f_policy, episodes=100, max_steps=50):
    returns = []
    for _ in range(episodes):
        s = env.reset()
        G = 0.0
        disc = 1.0
        for _ in range(max_steps):
            a = x_policy.sample(s)
            b = f_policy.sample(s)
            s, r, done = env.step(s, a, b)
            G += disc * r
            disc *= env.gamma
        returns.append(G)
    return np.mean(returns), np.std(returns)


if __name__ == "__main__":
    np.random.seed(1)  # Fresh seed for reproducibility

    # 1) Build environment
    env = HunterPreyGame(grid_size=2, slip_prob=0.25, gamma=0.9)

    # 2) Exact tabular Nash for the canonical 4-state game (value + policies)
    analyzer = TabularAnalyzer(env)
    V_star_tabular, pi_h_star, pi_p_star = analyzer.solve_nash_equilibrium(
        tol=1e-8,
        max_iter=10000,
        print_all=True,  # see all canonical states' NE policies
    )

    # ============================================================
    # 3) TRAINING RUN 1: NO MOMENTUM (baseline)
    # ============================================================
    print("\n================ RUN 1: NO MOMENTUM (hunter) ================\n")
    x_policy_nomom, f_policy_nomom, v_param_nomom, history_nomom = two_player_npg(
        env,
        analyzer,
        V_star_tabular,
        K=200,

        # --- HUNTER ---
        T_greedy=50,
        N_greedy=200,       
        eta_g=0.01,
        alpha_g=0.2,

        # --- PREY ---
        T_iter=100,
        N_iter=200,
        eta_i=0.1,
        alpha_i=0.2,

        # --- STABILITY ---
        W_radius=12.5,
        polyak_tau=0.05,

        # <<< NEW: no momentum
        momentum_max=0.0,
        momentum_min=0.0,
    )

    # ============================================================
    # 4) TRAINING RUN 2: MOMENTUM 0.9 FOR HUNTER
    # ============================================================
    print("\n================ RUN 2: MOMENTUM 0.9 (hunter) ================\n")
    x_policy_mom, f_policy_mom, v_param_mom, history_mom = two_player_npg(
        env,
        analyzer,
        V_star_tabular,
        K=200,

        # same hyperparams as run 1
        T_greedy=50,
        N_greedy=200,       
        eta_g=0.01,
        alpha_g=0.2,

        T_iter=100,
        N_iter=200,
        eta_i=0.1,
        alpha_i=0.2,

        W_radius=12.5,
        polyak_tau=0.05,

        # <<< NEW: momentum
        momentum_max=0.9,
        momentum_min=0.9,
    )

    # ============================================================
    # 5) Plot training stats (overlay no-momentum vs momentum)
    # ============================================================

    # Extract histories
    V_hat_nomom = np.array(history_nomom["V_start_hat"])
    BR_nomom    = np.array(history_nomom["BR_start"])
    gaps_nomom  = np.array(history_nomom["nash_gap"])
    x_probs_nomom = np.stack(history_nomom["x_probs"], axis=0)

    V_hat_mom = np.array(history_mom["V_start_hat"])
    BR_mom    = np.array(history_mom["BR_start"])
    gaps_mom  = np.array(history_mom["nash_gap"])
    x_probs_mom = np.stack(history_mom["x_probs"], axis=0)

    iters_nomom = np.arange(1, len(V_hat_nomom) + 1)
    iters_mom   = np.arange(1, len(V_hat_mom) + 1)

    # (a) Values – separate figure
    plt.figure(figsize=(6, 4))
    plt.plot(iters_nomom, V_hat_nomom, label="V_hat (no mom)", marker="o", markersize=3, alpha=0.7)
    plt.plot(iters_nomom, BR_nomom,    label="V_x,BR (no mom)", marker="x", markersize=3, alpha=0.7)
    plt.plot(iters_mom, V_hat_mom,     label="V_hat (mom 0.9)", marker="s", markersize=3, alpha=0.7)
    plt.plot(iters_mom, BR_mom,        label="V_x,BR (mom 0.9)", marker="^", markersize=3, alpha=0.7)

    plt.axhline(
        V_star_tabular[analyzer.start_idx],
        linestyle="--", linewidth=1, color='k', label="V* (tabular)"
    )
    plt.xlabel("Outer iteration k")
    plt.ylabel("Value at start")
    plt.title("Approx vs True vs Best-Response Value (no-mom vs mom)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("values_no_mom_vs_mom.pdf", bbox_inches="tight")   # <<< NEW
    plt.show()

    # (b) Nash gap – separate figure
    plt.figure(figsize=(6, 4))
    plt.plot(iters_nomom, gaps_nomom, marker="s", markersize=3, alpha=0.7, label="Gap (no mom)")
    plt.plot(iters_mom, gaps_mom, marker="d", markersize=3, alpha=0.7, label="Gap (mom 0.9)")
    plt.xlabel("Outer iteration k")
    plt.ylabel("Nash gap V* - V_x,BR")
    plt.title("Distance to Nash (no-mom vs mom)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("nash_gap_no_mom_vs_mom.pdf", bbox_inches="tight")   # <<< NEW
    plt.show()

    # (c) Hunter’s “Right” probability at start – separate figure
    plt.figure(figsize=(6, 4))
    plt.plot(iters_nomom, x_probs_nomom[:, 4], marker="d", markersize=3, alpha=0.7,
             label="P_right (no mom)")
    plt.plot(iters_mom, x_probs_mom[:, 4], marker="o", markersize=3, alpha=0.7,
             label="P_right (mom 0.9)")
    plt.xlabel("Outer iteration k")
    plt.ylabel("P_hunter(action=Right | start)")
    plt.title("Hunter policy at start over outer iterations")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("hunter_right_prob_no_mom_vs_mom.pdf", bbox_inches="tight")  # <<< NEW
    plt.show()

    # ============================================================
    # 6) Compare learned policies at start state
    # ============================================================

    start_state = env.reset()
    s0_idx = analyzer.start_idx
    learned_h_nomom = x_policy_nomom.probs(start_state)
    learned_h_mom   = x_policy_mom.probs(start_state)
    nash_h          = pi_h_star[s0_idx]

    print("\n--- Start-state policy comparison (canonical game) ---")
    print("Exact Nash hunter π_h(· | start):", nash_h)
    print("Learned hunter π_x (no momentum):", learned_h_nomom)
    print("Learned hunter π_x (mom = 0.9): ", learned_h_mom)
    print(f"Exact Nash value V*(start):       {V_star_tabular[s0_idx]:.6f}")

    # ============================================================
    # 7) Evaluation vs random prey for both runs
    # ============================================================

    random_prey = LogLinearPolicy(
        env.n_actions,
        len(state_features(env.reset(), grid_size=env.grid_size)),
        grid_size=env.grid_size
    )

    mean_nomom, std_nomom = evaluate(env, x_policy_nomom, random_prey, episodes=500)
    mean_mom,   std_mom   = evaluate(env, x_policy_mom,   random_prey, episodes=500)

    print(f"\nEvaluation vs random prey (MC) – NO MOMENTUM:  mean ≈ {mean_nomom:.3f} ± {std_nomom:.3f}")
    print(f"Evaluation vs random prey (MC) – MOMENTUM 0.9: mean ≈ {mean_mom:.3f} ± {std_mom:.3f}")

    # ============================================================
    # 8) Hunter strategy vs Nash in compressed (canonical) states
    #    comparing no-momentum vs momentum side-by-side
    # ============================================================

    print("\n--- Hunter strategy vs Nash in canonical coordinates (hunter at (0,0)) ---")

    canonical_states = [
        (0, 0, 0, 0),  # prey same cell as hunter
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 0, 1, 1),
    ]

    for s in canonical_states:
        idx = analyzer.state_to_idx[s]
        learned_nomom = x_policy_nomom.probs(s)
        learned_mom   = x_policy_mom.probs(s)
        nash_p        = pi_h_star[idx]

        print(f"Canonical state index {idx}, state {s}:")
        print(
            f"   NO MOM : S={learned_nomom[0]:.3f}, U={learned_nomom[1]:.3f}, "
            f"D={learned_nomom[2]:.3f}, L={learned_nomom[3]:.3f}, R={learned_nomom[4]:.3f}"
        )
        print(
            f"   MOM 0.9: S={learned_mom[0]:.3f}, U={learned_mom[1]:.3f}, "
            f"D={learned_mom[2]:.3f}, L={learned_mom[3]:.3f}, R={learned_mom[4]:.3f}"
        )
        print(
            f"   NASH   : S={nash_p[0]:.3f}, U={nash_p[1]:.3f}, "
            f"D={nash_p[2]:.3f}, L={nash_p[3]:.3f}, R={nash_p[4]:.3f}"
        )
        print("-" * 60)
