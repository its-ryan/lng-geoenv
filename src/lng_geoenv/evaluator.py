import numpy as np


def evaluate_episode(history):
    total_reward = sum(h["reward"] for h in history)

    total_cost = sum(h["metrics"].get("cost", 0) for h in history)
    total_shortage = sum(h["metrics"].get("shortage", 0) for h in history)

    # Normalize penalties to avoid huge values
    normalized_penalty = total_shortage * 0.000001 + total_cost * 0.001

    score = 1 / (1 + normalized_penalty)

    return {
        "total_reward": total_reward,
        "avg_reward": total_reward / len(history),
        "final_score": max(0.0, min(1.0, score)),
        "risk_adjusted_score": score,
        "steps": len(history),
        "breakdown": {"cost": total_cost, "shortage": total_shortage},
        "explanation": "Score based on minimizing shortage and cost",
    }
