import numpy as np
import matplotlib.pyplot as plt

def compute_directions(vector, tolerance):
    vector = np.array(vector, dtype=float)
    module = np.linalg.norm(vector)
    vectorN = vector / module

    # перпендикуляр
    u_perp = np.array([-vectorN[1], vectorN[0]])

    d1 = tolerance * vectorN + np.sqrt(1 - tolerance**2) * u_perp
    d2 = tolerance * vectorN - np.sqrt(1 - tolerance**2) * u_perp

    return d1, d2

def intersection_points(d, A):
    dx, dy = d
    s = dx + dy

    if abs(s) < 1e-8:
        return None, None

    p1 = (A / s) * d
    p2 = (0.7 * A / s) * d

    # отбрасываем дробную часть
    p1 = p1.astype(int)
    p2 = p2.astype(int)

    return p1, p2

def fix_point_2d(p, max_total):
    if p is None:
        return None

    x, y = p

    # если x отрицательный → (0, max_total)
    if x < 0:
        return np.array([0, max_total], dtype=int)

    # если y отрицательный → (max_total, 0)
    if y < 0:
        return np.array([max_total, 0], dtype=int)

    return p.astype(int)

# векторы состоящие из количества реплик каждого деплоймента
def compute_minmax_replicas_range(epoch_vectors, tolerance):
    epoch_vectors = [np.array(v, dtype=float) for v in epoch_vectors]

    if not epoch_vectors:
        return {}

    if any(len(v) != 2 for v in epoch_vectors):
        raise ValueError("Сейчас функция поддерживает только 2D-векторы")

    if not (0 <= tolerance <= 1):
        raise ValueError("tolerance должен быть в диапазоне [0, 1]")

    # максимум суммы реплик среди всех эпох
    max_total_replicas = max(np.sum(v) for v in epoch_vectors)

    result = {}

    for i, epoch in enumerate(epoch_vectors):
        d1, d2 = compute_directions(epoch, tolerance)

        max_point_1, min_point_1 = intersection_points(d1, max_total_replicas)
        max_point_2, min_point_2 = intersection_points(d2, max_total_replicas)

        max_point_1 = fix_point_2d(max_point_1, max_total_replicas)
        min_point_1 = fix_point_2d(min_point_1, max_total_replicas)
        max_point_2 = fix_point_2d(max_point_2, max_total_replicas)
        min_point_2 = fix_point_2d(min_point_2, max_total_replicas)

        valid_points = [p for p in [max_point_1, min_point_1, max_point_2, min_point_2] if p is not None]

        if not valid_points:
            result[f"epoch_{i}"] = {
                "epoch_vector": epoch.astype(int),
                "min_replicas": None,
                "max_replicas": None,
            }
            continue

        points = np.array(valid_points)

        min_replicas = np.min(points, axis=0).astype(int)
        max_replicas = np.max(points, axis=0).astype(int)

        result[f"epoch_{i}"] = {
            "epoch_vector": epoch.astype(int),
            "min_replicas": min_replicas,
            "max_replicas": max_replicas,
        }

    return {
        "max_total_replicas": int(max_total_replicas),
        "epochs": result,
    }


epoch_vectors = [
    # [2, 5],
    # [7, 3],
    # [3, 13],
    # [23,3],
    [3,16]
]

tolerance = 0.9

ranges = compute_minmax_replicas_range(epoch_vectors, tolerance)

for epoch_name, data in ranges["epochs"].items():
    print(epoch_name)
    print("  epoch_vector :", data["epoch_vector"])
    print("  min_replicas :", data["min_replicas"])
    print("  max_replicas :", data["max_replicas"])

print("max_total_replicas =", ranges["max_total_replicas"])