import random

class GridWorld:
    def __init__(self, n_states=5):
        self.n_states = n_states
        self.end = n_states - 1

    def step(self, state, action):
        if action == 1:
            next_state = min(self.end, state + 1)
        else:
            next_state = max(0, state - 1)
        reward = 1 if next_state == self.end else 0
        return next_state, reward

class QLearningAgent:
    def __init__(self, n_states, n_actions=2):
        self.q = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]
        self.n_actions = n_actions

    def choose_action(self, state, eps=0.1):
        if random.random() < eps:
            return random.randint(0, self.n_actions - 1)
        return max(range(self.n_actions), key=lambda a: self.q[state][a])

    def learn(self, env, episodes=100, alpha=0.1, gamma=0.9):
        for _ in range(episodes):
            state = 0
            while state != env.end:
                action = self.choose_action(state)
                next_state, reward = env.step(state, action)
                best_next = max(self.q[next_state])
                self.q[state][action] += alpha * (reward + gamma * best_next - self.q[state][action])
                state = next_state

if __name__ == "__main__":
    env = GridWorld(n_states=5)
    agent = QLearningAgent(n_states=5)
    agent.learn(env, episodes=200)

    state = 0
    steps = 0
    while state != env.end and steps < 10:
        action = agent.choose_action(state, eps=0)
        state, _ = env.step(state, action)
        print(f"State: {state}")
        steps += 1
