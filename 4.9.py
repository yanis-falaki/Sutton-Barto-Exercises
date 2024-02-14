import matplotlib.pyplot as plt
import matplotlib

# Probability of heads
ph = 0.4
# Discount factor
gamma = 1


def get_actions(state):
    actions = [i for i in range(min(state, 100 - state) + 1)]
    return actions


def update_value_for_state(state, state_values):
    if state == 100 or state == 0:
        return 0
    
    actions = get_actions(state)
    action_values = []
    for action in actions:
        win_state = min(state + action, 100)
        lose_state = state - action
        win_reward = 1 if win_state == 100 else 0
        lose_reward = 0 # Always 0 unless the goal state is reached, which is handled in win_reward

        # Expected value calculation for this action
        expected_value = ph * (win_reward + gamma * state_values[win_state]) + (1 - ph) * (lose_reward + gamma * state_values[lose_state])
        action_values.append(expected_value)

    # Update the state value to the maximum of the expected values over all actions
    return max(action_values) if action_values else 0

def define_policy(state_set, state_values):
    policy = {}
    for state in state_set[1:100]:
        actions = get_actions(state)
        action_returns = []

        for action in actions:
            win_state = min(state + action, 100)
            lose_state = state - action
            win_reward = 1 if win_state == 100 else 0
            lose_reward = 0 # Always 0 unless the goal state is reached, which is handled in win_reward

            expected_value = ph * (win_reward + gamma * state_values[win_state]) + (1 - ph) * (lose_reward + gamma * state_values[lose_state])
            action_returns.append((action, expected_value))

        best_action = max(action_returns, key=lambda x: x[1])[0]
        policy[state] = best_action

    return policy

state_set = [i for i in range(0, 101)]

state_values = [0 for i in range(0, 101)]

delta = 10
theta = 0.0000001
iteration = 0

while (delta > theta):
    iteration += 1
    delta = 0
    for state in state_set[1:100]:
        previous_value = state_values[state]
        state_values[state] = update_value_for_state(state, state_values)
        delta = max(delta, abs(state_values[state] - previous_value))

    plt.plot(state_set[1:100].copy(), state_values[1:100].copy(), label=f'iteration {iteration}') # Only displaying the value for non terminal states, the value for terminal state is always 0


plt.savefig('image')
