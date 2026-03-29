# general test for agent

import random
USE_RANDOM_AGENT = False  # Set to True to test random actions instead of choose_action

from src.lng_geoenv.env import LNGEnv
from src.lng_geoenv.agent import choose_action
from src.lng_geoenv.reward import RewardEngine

# Example config for RewardEngine (adjust as needed)
reward_config = {
    "w_cost": 1.0,
    "w_shortage": 10.0,
    "w_delay": 1.0,
    "w_risk": 1.0,
    "alpha": 2.0,
    "beta": 1.0,
    "gamma": 1.0
}

env_config = {
    "max_steps": 10,  # or any desired episode length
    "reward": reward_config
}

env = LNGEnv(env_config)

state = env.reset(seed=42)
done = False
total_reward = 0

print("\n--- START TEST ---\n")

while not done:
    print("\n==============================")
    print(f"Time Step: {state['time_step']}")
    print(f"Full State: {state}")
    print(f"Storage: {state['storage']['level']}")
    print(f"Blocked Routes: {state['blocked_routes']}")

    demand = state.get("demand", 0)
    if USE_RANDOM_AGENT:
        action = {"type": random.choice(["reroute", "store", "release", "hedge", "wait"]), "parameters": {}}
    else:
        action = choose_action(state, demand)

    print("Action:", action)

    next_state, reward, done, info = env.step(action)

    print("Reward:", reward)
    print("Reward Details:", info.get("metrics", {}))

    total_reward += reward
    state = next_state

print("\nTotal Reward:", total_reward)
print("\n--- EPISODE SUMMARY ---")
print(f"Final State: {state}")
print(f"Total Reward: {total_reward}")