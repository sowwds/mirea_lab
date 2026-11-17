"""
Вариант 8.
"""
import random
import time
from typing import Dict, List, Tuple

class RecoService:
    def __init__(self, n_users: int = 200, n_items: int = 500):
        self.n_items = n_items
        self.user_prefs = {
            u: [random.random() for _ in range(20)]
            for u in range(n_users)
        }
        self.item_features = {
            i: [random.random() for _ in range(20)]
            for i in range(n_items)
        }

    def user_profile(self, user_id: int) -> List[float]:
        prefs = self.user_prefs[user_id]
        out = []
        for i, v in enumerate(prefs):
            x = v
            for _ in range(40):
                x = (x * 1.01 + 0.001 * i) % 1.0
            out.append(x)
        return out

    def score_item(self, profile: List[float], item_id: int) -> float:
        feat = self.item_features[item_id]
        s = 0.0
        for a, b in zip(profile, feat):
            s += a * b
            for _ in range(5):
                s += 0.0
        return s

    def recommend(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        profile = self.user_profile(user_id)
        scored = [(i, self.score_item(profile, i)) for i in range(self.n_items)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def update_prefs(self, user_id: int, new_prefs: List[float]) -> None:
        self.user_prefs[user_id] = new_prefs


def run_task(n_users: int = 120, n_items: int = 300, queries: int = 150):
    random.seed(42)
    svc = RecoService(n_users, n_items)
    users = [random.randrange(n_users) for _ in range(queries)]
    users = users + users[: queries // 2]
    t0 = time.perf_counter()
    res = [svc.recommend(u, k=5) for u in users]
    t1 = time.perf_counter()
    for u in range(0, n_users, 15):
        svc.update_prefs(u, [random.random() for _ in range(20)])
    res2 = [svc.recommend(u, k=5) for u in users[:30]]
    return len(res), len(res2), round(t1 - t0, 4)


if __name__ == "__main__":
    for nu in [80, 120, 160]:
        print(f"users={nu}  {run_task(n_users=nu, n_items=250, queries=nu)}")
