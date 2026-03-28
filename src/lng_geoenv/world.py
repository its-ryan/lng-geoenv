import numpy as np

def update_ships(ships, blocked_routes):
    for ship in ships:
        if ship["status"] == "moving":

            # if route blocked then delay
            if ship["route"] in blocked_routes:
                ship["eta"] += 1   # delay
            else:
                ship["eta"] -= 1

            if ship["eta"] <= 0:
                ship["status"] = "arrived"

    return ships


def handle_arrivals(ships, storage):
    for ship in ships:
        if ship["status"] == "arrived":
            storage["level"] = min(
                storage["capacity"],
                storage["level"] + ship["capacity"]
            )
            ship["status"] = "done"

    return ships, storage

class World:
    """
    Simulates routing risk and cost structure.
    """

    def __init__(self, config):
        self.risk_scale = config["risk_scale"]
        self.price_volatility = config["price_volatility"]

    def sample_route_risk(self):
        return np.random.uniform(0, 1) * self.risk_scale

    def fuel_cost(self, distance):
        base = 0.05 * distance
        fluctuation = np.random.normal(0, self.price_volatility * base)
        return max(0.0, base + fluctuation)

    def storage_cost(self, storage_level):
        return 0.02 * storage_level

    def hedge_cost(self, hedge_amount):
        return 0.1 * hedge_amount