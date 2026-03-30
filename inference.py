from src.lng_geoenv.runner import run_task
import json


def run_all_tasks():
    tasks = ["stable", "volatile", "war"]

    results = []
    for task in tasks:
        result = run_task(task, seed=42)  # ensure reproducibility
        results.append(result)

    avg_score = sum(r["score"] for r in results) / len(results)

    return {
        "environment": "lng-geoenv",
        "tasks": results,
        "average_score": avg_score,
        "execution_status": "success"
    }


if __name__ == "__main__":
    output = run_all_tasks()
    print(json.dumps(output, indent=2))