"""
Sutton and Barto from Scratch 1: Bandits and Dynamic Programming

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - create_bandit_testbed
def create_bandit_testbed(k, seed, mean=0.0, std=1.0):
    rng = np.random.RandomState(seed)
    return rng.normal(loc = mean, scale = std, size = k)

# Step 2 - pull_arm
def pull_arm(true_values, action, rng):
    """Pull one arm and return reward = true value + unit-normal noise.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        action (int): Index of the arm to pull.
        rng (np.random.Generator): Seeded random generator for the noise.

    Returns:
        float: Stochastic reward for this pull.
    """
    return true_values[action] + rng.normal()

# Step 3 - sample_average_update
def sample_average_update(q_values, action_counts, action, reward):
    q_values_c = q_values.copy()
    action_counts_c = action_counts.copy()
    action_counts_c[action] += 1
    q_values_c[action] += (reward - q_values_c[action]) / action_counts_c[action]
    return q_values_c, action_counts_c

# Step 4 - epsilon_greedy_action
def epsilon_greedy_action(q_values, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(len(q_values))
    return np.argmax(q_values)

# Step 5 - run_bandit_episode
def run_bandit_episode(true_values, n_steps, epsilon, rng):
    """Run one bandit episode with epsilon-greedy selection and sample-average updates.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, actions) with shapes (n_steps,) and (n_steps,) of ints.
    """
    k = len(true_values)
    q_values = np.zeros(k)
    action_counts = np.zeros(k)
    rewards = np.zeros(n_steps)
    actions = np.zeros(n_steps, dtype = int)
    for t in range(0, n_steps):
        action = epsilon_greedy_action(q_values, epsilon, rng)
        reward = pull_arm(true_values, action, rng)
        rewards[t] = reward
        actions[t] = action
        q_values, action_counts = sample_average_update(q_values, action_counts, action, reward)
    return rewards, actions

# Step 6 - track_rewards_and_optimal_actions
def track_rewards_and_optimal_actions(true_values, n_steps, epsilon, rng):
    """Run one episode tracking rewards and optimal-arm choices.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, optimal_flags) each shape (n_steps,).
            optimal_flags entries are 0.0 or 1.0 floats.
    """
    rewards, actions = run_bandit_episode(true_values, n_steps, epsilon, rng)
    oa_index = np.argmax(true_values)
    optimal_flags = (actions == oa_index).astype(float)
    return rewards, optimal_flags

# Step 7 - average_bandit_curves
def average_bandit_curves(k, n_runs, n_steps, epsilon, seed):
    rewards_all = np.zeros((n_runs, n_steps))
    optimal_all = np.zeros((n_runs, n_steps))
    for i in range(n_runs):
        true_values = create_bandit_testbed(k, seed + i)
        rng = np.random.default_rng(seed + i)
        rewards, optimal_flags = track_rewards_and_optimal_actions(true_values, n_steps, epsilon, rng)
        rewards_all[i] = rewards
        optimal_all[i] = optimal_flags
    mean_rewards = rewards_all.mean(axis = 0)
    mean_optimal_fraction = optimal_all.mean(axis = 0)
    return mean_rewards, mean_optimal_fraction

# Step 8 - apply_random_walk_drift
def apply_random_walk_drift(true_values, drift_std, rng):
    noise = rng.normal(loc = 0, scale = drift_std, size = true_values.shape)
    return true_values + noise

# Step 9 - constant_step_size_update
def constant_step_size_update(q_values, action, reward, alpha):
    q_values[action] += alpha * (reward - q_values[action])
    return q_values

# Step 10 - optimistic_initialization
def optimistic_initialization(k, initial_value):
    return np.full(k, initial_value)

# Step 11 - ucb_action_select
def ucb_action_select(q_values, action_counts, timestep, c):
    """Select an action by upper-confidence-bound scores.

    Args:
        q_values (np.ndarray): Action-value estimates, shape (k,).
        action_counts (np.ndarray): Visit counts per action, shape (k,).
        timestep (int): Current time step t (>= 1).
        c (float): Exploration constant.

    Returns:
        int: Index of the selected action.
    """
    k = len(q_values)
    scores = np.zeros(k)
    visited = action_counts > 0
    scores[visited] = (q_values[visited] + c * np.sqrt(np.log(timestep) / action_counts[visited]))
    scores[action_counts == 0] = np.inf
    return int(np.argmax(scores))

# Step 12 - gradient_bandit_update
def gradient_bandit_update(preferences, action, reward, average_reward, alpha):
    e = np.exp(preferences - np.max(preferences))
    pi = e / np.sum(e)
    adv = reward - average_reward
    preferences -= alpha * adv * pi
    preferences[action] += alpha * adv
    return preferences

# Step 13 - bandit_parameter_study
def bandit_parameter_study(n_runs, n_steps, seed, settings):
    results = {}
    for setting in settings:
        method = setting["method"]
        param = float(setting["param"])
        nonstationary = setting.get("nonstationary", False)
        label = f"{method}({param})"
        if nonstationary:
            label += ",ns"
        final_rewards = []
        for run in range(n_runs):
            ep_seed = seed + run
            rng = np.random.default_rng(ep_seed)
            if nonstationary:
                true_values = np.zeros(10)
            else:
                true_values = create_bandit_testbed(10, ep_seed)
            q = np.zeros(10)
            counts = np.zeros(10, dtype=int)
            preferences = np.zeros(10)
            average_reward = 0.0
            last_reward = 0.0
            for t in range(1, n_steps + 1):
                if method == "epsilon_greedy":
                    action = epsilon_greedy_action(q, param, rng)
                elif method == "constant_step":
                    action = epsilon_greedy_action(q, 0.1, rng)
                elif method == "optimistic":
                    if t == 1:
                        q = optimistic_initialization(10, param)
                    action = epsilon_greedy_action(q, 0.0, rng)
                elif method == "ucb":
                    action = ucb_action_select(q, counts, t, param)
                elif method == "gradient":
                    e = np.exp(preferences - np.max(preferences))
                    pi = e / np.sum(e)
                    action = rng.choice(10, p = pi)
                else:
                    raise ValueError(f"Unknown method: {method}")
                last_reward = pull_arm(true_values, action, rng)
                if method == "epsilon_greedy":
                    q, counts = sample_average_update(q, counts, action, last_reward)
                elif method == "constant_step":
                    q = constant_step_size_update(q, action, last_reward, param)
                elif method == "optimistic":
                    q = constant_step_size_update(q, action, last_reward, 0.1)
                elif method == "ucb":
                    q, counts = sample_average_update(q, counts, action, last_reward)
                elif method == "gradient":
                    average_reward += (last_reward - average_reward) / t
                    preferences = gradient_bandit_update(preferences, action, last_reward, average_reward, param)
                if nonstationary:
                    true_values = apply_random_walk_drift(true_values, 0.01, rng)
            final_rewards.append(float(last_reward))
        results[label] = float(np.mean(final_rewards))
    return results

# Step 14 - build_gridworld_mdp
def build_gridworld_mdp():
    n_states = 16
    n_actions = 4
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    P = []
    for s in range(n_states):
        state_transitions = []
        for a in range(n_actions):
            if s == 0 or s == 15:
                state_transitions.append([(1.0, s, 0.0)])
            else:
                row, col = divmod(s, 4)
                dr, dc = deltas[a]
                next_row = max(0, min(3, row + dr))
                next_col = max(0, min(3, col + dc))
                next_s = next_row * 4 + next_col
                state_transitions.append([(1.0, next_s, -1.0)])
        P.append(state_transitions)
    return {"n_states": n_states, "n_actions": n_actions, "P": P}

# Step 15 - iterative_policy_evaluation
def iterative_policy_evaluation(policy, mdp, gamma, theta):
    n_states = mdp["n_states"]
    n_actions = mdp["n_actions"]
    P = mdp["P"]
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            new_value = 0.0
            if policy.ndim == 1:
                actions = [(policy[s], 1.0)]
            else:
                actions = [(a, policy[s, a]) for a in range(n_actions)]
            for a, action_prob in actions:
                for prob, next_s, reward in P[s][a]:
                    new_value += (action_prob * prob * (reward + gamma * V[next_s]))
            V[s] = new_value
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

# Step 16 - greedy_policy_improvement
def greedy_policy_improvement(state_values, mdp, gamma):
    n_states = mdp["n_states"]
    n_actions = mdp["n_actions"]
    P = mdp["P"]
    policy = np.zeros(n_states, dtype = int)
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_s, reward in P[s][a]:
                action_values[a] += prob * (reward + gamma * state_values[next_s])
        policy[s] = np.argmax(action_values)
    return policy

# Step 17 - policy_iteration
def policy_iteration(mdp, gamma, theta):
    n_states = mdp["n_states"]
    policy = np.zeros(n_states, dtype = int)
    while True:
        state_values = iterative_policy_evaluation(policy, mdp, gamma, theta)
        new_policy = greedy_policy_improvement(state_values, mdp, gamma)
        if np.array_equal(policy, new_policy):
            return state_values, new_policy
        policy = new_policy

# Step 18 - value_iteration (not yet solved)
# TODO: implement

# Step 19 - build_gambler_mdp (not yet solved)
# TODO: implement

# Step 20 - gambler_value_iteration (not yet solved)
# TODO: implement

# Step 21 - extract_optimal_stakes (not yet solved)
# TODO: implement

