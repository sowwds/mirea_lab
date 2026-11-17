import time
import random
from dataclasses import dataclass


@dataclass
class Order:
    id: int
    user_id: int
    amount: float
    status: str
    category: str


def filter_and_aggregate(orders: list[Order], min_amount: float) -> dict:
    filtered = []
    for o in orders:
        if o.amount >= min_amount:
            filtered.append(o)
    completed = []
    for o in filtered:
        if o.status == "completed":
            completed.append(o)
    categories = []
    for o in completed:
        if o.category not in categories:
            categories.append(o.category)
    result = {}
    for cat in categories:
        cat_orders = []
        for o in completed:
            if o.category == cat:
                cat_orders.append(o)
        total = 0.0
        for o in cat_orders:
            total += o.amount
        avg = total / len(cat_orders) if cat_orders else 0.0
        users = []
        for o in cat_orders:
            if o.user_id not in users:
                users.append(o.user_id)

        result[cat] = {
            "count": len(cat_orders),
            "total": total,
            "avg": avg,
            "unique_users": len(users),
        }
    return result


def generate_orders(n: int) -> list[Order]:
    statuses = ["completed", "pending", "cancelled", "completed", "completed"]
    categories = ["electronics", "books", "clothing", "food", "other"]
    orders = []
    for i in range(n):
        orders.append(Order(
            id=i,
            user_id=random.randint(1, n // 5 + 10),
            amount=round(random.uniform(5, 500), 2),
            status=random.choice(statuses),
            category=random.choice(categories),
        ))
    return orders


def run_task(n: int):
    orders = generate_orders(n)
    agg = filter_and_aggregate(orders, min_amount=50.0)
    total_count = sum(v["count"] for v in agg.values())
    return total_count, len(agg)


if __name__ == "__main__":
    random.seed(42)
    for n in [2000, 5000, 10000]:
        start = time.perf_counter()
        res = run_task(n)
        elapsed = time.perf_counter() - start
        print(f"N={n:6d}  result={res}  time={elapsed:.4f}s")
