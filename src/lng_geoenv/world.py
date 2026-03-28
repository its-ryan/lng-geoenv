import numpy as np

ROUTE_RISK = {
    "Suez": 0.6,
    "Panama": 0.3,
    "Atlantic": 0.2,
    "Hormuz": 0.9
}

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

    def __init__(self, config=None):
            self.price_volatility = 0.1

    def route_risk(self, route, blocked_routes):
        base = ROUTE_RISK.get(route, 0.5)

        if route in blocked_routes:
            base *= 1.5  # amplify risk

        return min(1.0, base)

    def fuel_cost(self, distance):
        base = 0.05 * distance
        noise = np.random.normal(0, self.price_volatility * base)
        return max(0, base + noise)