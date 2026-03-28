class RewardNormalizer:
    """
    Tracks reward range and normalizes to [0, 1].
    """

    def __init__(self):
        self.min_r = float("inf")
        self.max_r = float("-inf")

    def normalize(self, r):
        self.min_r = min(self.min_r, r)
        self.max_r = max(self.max_r, r)

        denom = (self.max_r - self.min_r)
        if denom < 1e-8:
            return 0.5

        return (r - self.min_r) / denom


class EpisodeGrader:
    """
    Converts raw metrics into a bounded score.
    """

    def __init__(self, weights):
        self.w_cost = weights["cost"]
        self.w_shortage = weights["shortage"]
        self.w_risk = weights["risk"]

    def grade(self, metrics):
        cost = metrics["total_cost"]
        shortage = metrics["total_shortage"]
        risk = metrics["total_risk"]

        score = (
            self.w_cost * (1 / (1 + cost)) +
            self.w_shortage * (1 / (1 + shortage)) +
            self.w_risk * (1 / (1 + risk))
        )

        return max(0.0, min(1.0, score))