from typing import List
from pydantic import BaseModel, computed_field
import numpy as np


# -----------------------------
# Baseline Random Policy
# -----------------------------
class RandomPolicy:
    """
    Baseline random agent for testing.
    """

    def act(self, state):
        return {
            "action_type": np.random.choice(
                ["wait", "store", "release", "reroute", "hedge"]
            ),
            "amount": float(np.random.uniform(0, 30)),
            "ship_id": int(np.random.randint(1, 3)),
            "new_route": np.random.choice(["Suez", "Panama", "Atlantic", "Hormuz"]),
        }


# -----------------------------
# OpenEnv Typed Models
# -----------------------------


class Ship(BaseModel):
    id: int
    origin: str
    destination: str
    current_location: str
    eta: int
    capacity: float
    route: str
    status: str


class Storage(BaseModel):
    level: float
    capacity: float


class Observation(BaseModel):
    time_step: int
    ships: List[Ship]
    blocked_routes: List[str]
    storage: Storage
    demand_forecast: List[float]
    price: float
    budget: float

    @computed_field
    @property
    def demand(self) -> float:
        """Get current demand from forecast at current time_step."""
        if self.time_step < len(self.demand_forecast):
            return self.demand_forecast[self.time_step]
        return self.demand_forecast[-1] if self.demand_forecast else 0.0


class Action(BaseModel):
    action_type: str  # "wait" | "store" | "release" | "reroute" | "hedge"
    amount: float = 0.0
    ship_id: int | None = None
    new_route: str | None = None


class Reward(BaseModel):
    value: float
    breakdown: dict | None = None
