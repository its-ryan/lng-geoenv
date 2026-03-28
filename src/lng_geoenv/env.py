import random
from .world import update_ships, handle_arrivals, World
from .demand import DemandGenerator
from .reward import RewardEngine
from .grader import RewardNormalizer


ROUTE_DISTANCE = {
    "Suez": 100,
    "Panama": 140,
    "Atlantic": 160,
    "Hormuz": 80
}


class LNGEnv:
    def __init__(self, config):
        self.state = None
        self.max_steps = config["max_steps"]

        self.demand_gen = DemandGenerator()
        self.reward_engine = RewardEngine(config["reward"])
        self.normalizer = RewardNormalizer()
        self.world = World()

    def reset(self, seed=42):
        random.seed(seed)

        self.time_step = 0

        self.state = {
            "time_step": 0,
            "ships": [
                {
                    "id": 1,
                    "origin": "Qatar",
                    "destination": "Europe",
                    "eta": 5,
                    "capacity": 100,
                    "route": "Suez",
                    "status": "moving",
                },
                {
                    "id": 2,
                    "origin": "USA",
                    "destination": "Europe",
                    "eta": 7,
                    "capacity": 80,
                    "route": "Panama",
                    "status": "moving",
                }
            ],
            "blocked_routes": random.sample(["Suez", "Panama"], k=1),
            "storage": {
                "level": 50.0,
                "capacity": 200.0,
            },
            "price": random.uniform(50, 150),
            "budget": 500.0,
        }

        self.state["demand"] = self.demand_gen.step()

        self.total_cost = 0
        self.total_shortage = 0
        self.total_risk = 0

        return self.state

    def apply_action(self, action):
        if action["type"] == "release":
            amt = action["parameters"]["amount"]
            amt = min(amt, self.state["storage"]["level"])
            self.state["storage"]["level"] -= amt

        elif action["type"] == "store":
            amt = action["parameters"]["amount"]
            self.state["storage"]["level"] = min(
                self.state["storage"]["capacity"],
                self.state["storage"]["level"] + amt
            )

        elif action["type"] == "reroute":
            for ship in self.state["ships"]:
                if ship["id"] == action["parameters"]["ship_id"]:
                    ship["route"] = action["parameters"]["new_route"]
                    ship["eta"] += 2

        elif action["type"] == "hedge":
            cost = 10
            if self.state["budget"] >= cost:
                self.state["budget"] -= cost
                self.state["storage"]["level"] += 20  # hedge adds supply

    def step(self, action):
        self.apply_action(action)

        self.state["ships"] = update_ships(
            self.state["ships"],
            self.state["blocked_routes"]
        )

        self.state["ships"], self.state["storage"] = handle_arrivals(
            self.state["ships"], self.state["storage"]
        )

        # Demand evolves
        demand = self.demand_gen.step()

        # Price reacts to demand (simple coupling)
        self.state["price"] *= (1 + 0.01 * (demand / 100))

        self.state["demand"] = demand

        # Supply = storage + incoming shipments (ETA <= 1)
        incoming = sum(
            ship["capacity"]
            for ship in self.state["ships"]
            if ship["eta"] <= 1 and ship["status"] == "moving"
        )

        supply = self.state["storage"]["level"] + incoming

        deficit = max(0, demand - supply)

        # Cost model
        fuel_cost = 0
        risk = 0

        for ship in self.state["ships"]:
            distance = ROUTE_DISTANCE.get(ship["route"], 120)
            fuel_cost += self.world.fuel_cost(distance)

            risk += self.world.route_risk(
                ship["route"],
                self.state["blocked_routes"]
            )

        risk /= max(1, len(self.state["ships"]))

        storage_cost = 0.02 * self.state["storage"]["level"]
        hedge_cost = 10 if action["type"] == "hedge" else 0

        delay = sum(max(0, ship["eta"]) for ship in self.state["ships"])

        reward, components = self.reward_engine.compute({
            "fuel_cost": fuel_cost,
            "storage_cost": storage_cost,
            "hedge_cost": hedge_cost,
            "deficit": deficit,
            "delay": delay,
            "risk": risk,
            "cargo_value": supply
        })

        reward = self.normalizer.normalize(reward)

        self.total_cost += components["cost"]
        self.total_shortage += components["shortage"]
        self.total_risk += components["risk"]

        self.time_step += 1
        self.state["time_step"] = self.time_step

        done = self.time_step >= self.max_steps

        return self.state, reward, done, {
            "metrics": components
        }

    def get_state(self):
        return self.state