import numpy as np
from src.lng_geoenv.env import LNGEnv
from src.lng_geoenv.demand import DemandGenerator
from src.lng_geoenv.reward import RewardEngine
from src.lng_geoenv.tasks import get_task_config


def generate_demand(seed=42, max_steps=10):
    np.random.seed(seed)
    demand_gen = DemandGenerator()
    demand_forecast = [demand_gen.step() for _ in range(max_steps)]
    return demand_forecast


def compute_reward(state, demand, reward_engine):
    storage_level = state.get("storage", {}).get("level", 0.0)
    ships = state.get("ships", [])
    blocked_routes = state.get("blocked_routes", [])

    incoming = sum(
        ship.get("capacity", 0)
        for ship in ships
        if ship.get("eta", 999) <= 1 and ship.get("status") == "moving"
    )
    supply = storage_level + incoming
    deficit = max(0, demand - supply)

    fuel_cost = sum(len(ship.get("route", "")) * 0.1 for ship in ships)
    storage_cost = 0.02 * storage_level
    hedge_cost = 0  

    delay = sum(max(0, ship.get("eta", 0)) for ship in ships)
    risk = 0.1 * len(blocked_routes)
    cargo_value = supply

    reward, components = reward_engine.compute(
        {
            "fuel_cost": fuel_cost,
            "storage_cost": storage_cost,
            "hedge_cost": hedge_cost,
            "deficit": deficit,
            "delay": delay,
            "risk": risk,
            "cargo_value": cargo_value,
        }
    )

    return reward, components


def choose_action(state, demand):
    storage_level = state.get("storage", {}).get("level", 0.0)
    storage_capacity = state.get("storage", {}).get("capacity", 200.0)
    price = state.get("price", 100.0)
    budget = state.get("budget", 500.0)
    ships = state.get("ships", [])
    blocked_routes = state.get("blocked_routes", [])

    if storage_level < demand:
        shortage_amount = demand - storage_level

        for ship in ships:
            if ship.get("route") in blocked_routes:
                available_routes = [
                    r
                    for r in ["Suez", "Panama", "Atlantic", "Hormuz"]
                    if r not in blocked_routes
                ]
                if available_routes:
                    new_route = available_routes[0]
                    return {
                        "type": "reroute",
                        "parameters": {
                            "ship_id": ship.get("id"),
                            "new_route": new_route,
                        },
                    }

        if storage_level > 0:
            release_amount = min(shortage_amount * 0.8, storage_level)
            return {"type": "release", "parameters": {"amount": release_amount}}

    for ship in ships:
        if ship.get("route") in blocked_routes and ship.get("status") == "moving":
            available_routes = [
                r
                for r in ["Suez", "Panama", "Atlantic", "Hormuz"]
                if r not in blocked_routes
            ]
            if available_routes:
                new_route = available_routes[0]
                return {
                    "type": "reroute",
                    "parameters": {"ship_id": ship.get("id"), "new_route": new_route},
                }

    if price > 120 and budget >= 10:
        return {"type": "hedge", "parameters": {}}

    storage_ratio = storage_level / max(storage_capacity, 1.0)
    if storage_ratio > 0.85:  
        release_amount = (storage_level - 0.7 * storage_capacity) * 0.5
        if release_amount > 0:
            return {"type": "release", "parameters": {"amount": release_amount}}

    return {"type": "wait", "parameters": {}}


def evaluate_episode(history):
    total_reward = sum(h.get("reward", 0.0) for h in history)
    episode_count = len(history)

    avg_reward = total_reward / max(episode_count, 1)

    final_score = min(1.0, max(0.0, (avg_reward + 10.0) / 20.0))

    return {
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "final_score": final_score,
        "steps": episode_count,
    }


def run_task(task_name, max_steps=10, seed=42):
    task_config = get_task_config(task_name)

    config = {
        "max_steps": max_steps,
        "reward": {
            "w_cost": 1.0,
            "w_shortage": 6.0,
            "w_delay": 1.0,
            "w_risk": 3.0,
            "alpha": 2.0,
            "beta": 1.0,
            "gamma": 2.0,
        },
    }

    env = LNGEnv(config)
    state = env.reset(seed=seed)

    demand_forecast = generate_demand(seed=seed, max_steps=max_steps)

    reward_engine = RewardEngine(config["reward"])

    history = []

    print(f"\n{'=' * 60}")
    print(f"Task: {task_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Initial State:")
    print(
        f"  Storage: {state.get('storage', {}).get('level', 0.0):.1f} / {state.get('storage', {}).get('capacity', 200.0):.1f}"
    )
    print(f"  Price: ${state.get('price', 100.0):.2f}")
    print(f"  Budget: ${state.get('budget', 500.0):.2f}")
    print()

    t = 0
    while t < max_steps:
        if t < len(demand_forecast):
            demand = demand_forecast[t]
        else:
            demand = 0.0

        action = choose_action(state, demand)

        state, env_reward, env_done, env_info = env.step(action)

        reward, components = compute_reward(state, demand, reward_engine)

        done = t >= max_steps - 1

        history.append({"state": state, "action": action, "reward": reward})

        if (t + 1) % 5 == 0 or done:
            storage_level = state.get("storage", {}).get("level", 0.0)
            print(f"Step {t + 1}:")
            print(f"  Action: {action['type']}")
            print(f"  Storage: {storage_level:.1f}")
            print(f"  Demand: {demand:.1f}")
            print(f"  Reward: {reward:.4f}")
            print()

        t += 1

    evaluation = evaluate_episode(history)

    print(f"{'=' * 60}")
    print(f"FINAL RESULTS - {task_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Steps completed: {evaluation['steps']}")
    print(f"Total reward: {evaluation['total_reward']:.4f}")
    print(f"Average reward: {evaluation['avg_reward']:.4f}")
    print(f"Final score: {evaluation['final_score']:.4f}")
    print(f"{'=' * 60}\n")

    return {
        "task": task_name,
        "final_score": evaluation["final_score"],
        "total_reward": evaluation["total_reward"],
        "avg_reward": evaluation["avg_reward"],
        "steps": evaluation["steps"],
    }


def run_all_tasks(max_steps=10, seed=42):
    tasks = ["stable", "volatile", "war"]
    results = []

    print("\n" + "=" * 60)
    print("LNG-GeoENV EXECUTION PIPELINE")
    print("=" * 60)

    for task in tasks:
        result = run_task(task, max_steps=max_steps, seed=seed)
        results.append(result)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    avg_score = (
        sum(r["final_score"] for r in results) / len(results) if results else 0.0
    )
    avg_reward = (
        sum(r["total_reward"] for r in results) / len(results) if results else 0.0
    )

    for result in results:
        print(
            f"{result['task'].upper():10} | Score: {result['final_score']:.4f} | "
            f"Reward: {result['total_reward']:.4f}"
        )

    print(f"\nAverage Score: {avg_score:.4f}")
    print(f"Average Reward: {avg_reward:.4f}")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    results = run_all_tasks(max_steps=10, seed=42)