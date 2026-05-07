import numpy as np
from itertools import combinations


def compute_directions_2d(vector_2d, tolerance):
    vector_2d = np.array(vector_2d, dtype=float)

    if vector_2d.shape != (2,):
        raise ValueError("Ожидался 2D-вектор")

    norm = np.linalg.norm(vector_2d)
    if norm < 1e-12:
        return None, None

    u = vector_2d / norm
    u_perp = np.array([-u[1], u[0]])
    delta = np.sqrt(max(0.0, 1.0 - tolerance**2))

    d1 = tolerance * u + delta * u_perp
    d2 = tolerance * u - delta * u_perp
    return d1, d2


def intersection_with_sum_line(d, sum_value):
    if d is None:
        return None

    dx, dy = d
    s = dx + dy

    if abs(s) < 1e-12:
        return None

    return (sum_value / s) * d


def project_to_first_quadrant_edge(point, sum_value):
    """
    Если пересечение вышло в отрицательную координату,
    переносим его на соответствующую вершину отрезка x + y = sum_value
    в первом квадранте.
    """
    if point is None:
        return None

    x, y = point

    if x < 0:
        return np.array([0.0, sum_value], dtype=float)

    if y < 0:
        return np.array([sum_value, 0.0], dtype=float)

    return point


def floor_point(point):
    if point is None:
        return None
    return np.floor(point).astype(int)


def pair_minmax_2d(pair_epoch, A_pair, tolerance, low_ratio=0.7):
    """
    Для 2D-вектора pair_epoch считает min/max по двум прямым:
      x + y = A_pair
      x + y = low_ratio * A_pair
    и по двум граничным лучам конуса cos = tolerance.
    """
    d1, d2 = compute_directions_2d(pair_epoch, tolerance)
    if d1 is None:
        return None

    candidate_points = []

    for d in (d1, d2):
        for coeff in (1.0, low_ratio):
            sum_value = coeff * A_pair

            p = intersection_with_sum_line(d, sum_value)
            p = project_to_first_quadrant_edge(p, sum_value)
            p = floor_point(p)

            if p is not None:
                candidate_points.append(p)

    if not candidate_points:
        return None

    points = np.array(candidate_points, dtype=int)

    min_pair = np.min(points, axis=0)
    max_pair = np.max(points, axis=0)

    return {
        "pair_epoch": np.floor(pair_epoch).astype(int),
        "A_2d": float(A_pair),
        "min_2d": min_pair,
        "max_2d": max_pair,
        "points_2d": points,
    }


def compute_minmax_replicas_range_nd(epoch_vectors, tolerance, low_ratio=0.7):
    epoch_vectors = [np.array(v, dtype=float) for v in epoch_vectors]

    if not epoch_vectors:
        return {}

    n = len(epoch_vectors[0])

    if n < 2:
        raise ValueError("Нужно хотя бы 2 координаты")

    if any(len(v) != n for v in epoch_vectors):
        raise ValueError("Все эпохи должны иметь одинаковую размерность")

    if any(np.any(v < 0) for v in epoch_vectors):
        raise ValueError("Количество реплик не может быть отрицательным")

    if not (0 <= tolerance <= 1):
        raise ValueError("tolerance должен быть в диапазоне [0, 1]")

    if not (0 < low_ratio <= 1):
        raise ValueError("low_ratio должен быть в диапазоне (0, 1]")

    # Глобальный максимум суммы координат эпох
    max_total_replicas = max(np.sum(v) for v in epoch_vectors)

    result = {}

    for epoch_index, epoch in enumerate(epoch_vectors):
        total_epoch = np.sum(epoch)

        coord_min = np.full(n, np.inf)
        coord_max = np.full(n, -np.inf)

        pair_details = {}

        # Если эпоха нулевая, направление не определено
        if np.linalg.norm(epoch) < 1e-12:
            result[f"epoch_{epoch_index}"] = {
                "epoch_vector": epoch.astype(int),
                "min_replicas": np.zeros(n, dtype=int),
                "max_replicas": np.zeros(n, dtype=int),
                "pairs": {},
            }
            continue

        for i, j in combinations(range(n), 2):
            pair_epoch = epoch[[i, j]]

            # A для 2D-пространства:
            # max_sum_по_всем_эпохам - сумма отброшенных координат ТЕКУЩЕЙ эпохи
            dropped_sum = total_epoch - epoch[i] - epoch[j]
            A_pair = max_total_replicas - dropped_sum

            pair_result = pair_minmax_2d(pair_epoch, A_pair, tolerance, low_ratio)
            if pair_result is None:
                continue

            pair_details[f"({i},{j})"] = pair_result

            coord_min[i] = min(coord_min[i], pair_result["min_2d"][0])
            coord_max[i] = max(coord_max[i], pair_result["max_2d"][0])

            coord_min[j] = min(coord_min[j], pair_result["min_2d"][1])
            coord_max[j] = max(coord_max[j], pair_result["max_2d"][1])

        # Если для координаты не нашлось валидной пары,
        # оставляем исходное значение эпохи
        for k in range(n):
            if np.isinf(coord_min[k]):
                coord_min[k] = epoch[k]
                coord_max[k] = epoch[k]

        result[f"epoch_{epoch_index}"] = {
            "epoch_vector": epoch.astype(int),
            "min_replicas": coord_min.astype(int),
            "max_replicas": coord_max.astype(int),
            "pairs": pair_details,
        }

    return {
        "max_total_replicas": int(np.floor(max_total_replicas)),
        "epochs": result,
    }


# =========================
# Пример использования
# =========================

epoch_vectors = [
    [21, 25, 5, 6],
    [11, 19, 1, 26],
    [42, 3, 6, 9],
    [8, 34, 5, 11],
]

tolerance = 0.9

ranges = compute_minmax_replicas_range_nd(
    epoch_vectors=epoch_vectors,
    tolerance=tolerance,
    low_ratio=0.7
)

for epoch_name, data in ranges["epochs"].items():
    print(epoch_name)
    print("  epoch_vector :", data["epoch_vector"])
    print("  min_replicas :", data["min_replicas"])
    print("  max_replicas :", data["max_replicas"])
    # print("  pair_keys    :", list(data["pairs"].keys()))
    print()

print("max_total_replicas =", ranges["max_total_replicas"])