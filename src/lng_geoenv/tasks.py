def get_task_config(task_name):
    if task_name == "stable":
        return {
            "risk_scale": 0.2,
            "price_volatility": 0.1,
            "shock_prob": 0.05,
            "seasonal_amp": 5
        }

    elif task_name == "volatile":
        return {
            "risk_scale": 0.5,
            "price_volatility": 0.4,
            "shock_prob": 0.15,
            "seasonal_amp": 10
        }

    elif task_name == "war":
        return {
            "risk_scale": 0.9,
            "price_volatility": 0.7,
            "shock_prob": 0.3,
            "seasonal_amp": 15
        }

    else:
        raise ValueError("Unknown task")