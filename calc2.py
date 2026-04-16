import math
import numpy as np


def solve_quadratic_interval_on_unit_segment(c2, c1, c0, eps=1e-12):
    """
    Решает неравенство c2*a^2 + c1*a + c0 >= 0 на отрезке [0, 1].
    Возвращает (a_min, a_max) допустимого интервала.
    Если допустимых значений нет, возвращает None.
    """
    def f(a):
        return c2 * a * a + c1 * a + c0

    # Почти линейный случай
    if abs(c2) < eps:
        if abs(c1) < eps:
            if c0 >= -eps:
                return (0.0, 1.0)
            return None

        # c1 * a + c0 >= 0
        root = -c0 / c1
        if c1 > 0:
            left = max(0.0, root)
            right = 1.0
        else:
            left = 0.0
            right = min(1.0, root)

        if left <= right + eps:
            return (max(0.0, left), min(1.0, right))
        return None

    # Квадратичный случай
    D = c1 * c1 - 4.0 * c2 * c0

    if D < -eps:
        # Нет корней: знак везде один
        if f(0.5) >= -eps:
            return (0.0, 1.0)
        return None

    D = max(D, 0.0)
    sqrtD = math.sqrt(D)
    r1 = (-c1 - sqrtD) / (2.0 * c2)
    r2 = (-c1 + sqrtD) / (2.0 * c2)
    if r1 > r2:
        r1, r2 = r2, r1

    # Проверяем интервалы [0,r1], [r1,r2], [r2,1]
    candidates = [
        (0.0, min(1.0, r1)),
        (max(0.0, r1), min(1.0, r2)),
        (max(0.0, r2), 1.0),
    ]

    feasible_parts = []
    for L, R in candidates:
        if L <= R + eps:
            mid = 0.5 * (L + R)
            if f(mid) >= -eps:
                feasible_parts.append((max(0.0, L), min(1.0, R)))

    if not feasible_parts:
        return None

    # В нашей задаче обычно получится один связный интервал
    left = min(p[0] for p in feasible_parts)
    right = max(p[1] for p in feasible_parts)
    return (left, right)


def coordinate_fraction_range(epoch, i, tolerance):
    """
    Для фиксированной эпохи epoch и координаты i возвращает
    точный диапазон u_i на симплексе:
        u >= 0, sum(u)=1, cos(u, epoch) >= tolerance
    """
    epoch = np.asarray(epoch, dtype=float)
    d = len(epoch)

    if d == 1:
        return (1.0, 1.0)

    E = np.linalg.norm(epoch)
    if E <= 1e-12:
        raise ValueError("Нулевая эпоха недопустима")

    ei = float(epoch[i])
    g = np.delete(epoch, i)

    G = float(np.sum(g))          # L1 нормы хвоста
    Q = float(np.linalg.norm(g))  # L2 нормы хвоста

    # Специальный случай: хвоста нет
    if G <= 1e-12:
        a_min = tolerance / (
            tolerance + math.sqrt(d - 1) * math.sqrt(max(0.0, 1.0 - tolerance**2))
        )
        return (a_min, 1.0)

    beta = Q / G
    H = (Q * Q) / G
    T = (tolerance * E) ** 2

    c2 = (ei - H) ** 2 - T * (1.0 + beta**2)
    c1 = 2.0 * H * (ei - H) + 2.0 * T * beta**2
    c0 = H**2 - T * beta**2

    interval = solve_quadratic_interval_on_unit_segment(c2, c1, c0)
    if interval is None:
        return None

    return interval


def compute_minmax_replicas_range_closed_form(epoch_vectors, tolerance, min_scale=0.7):
    """
    Точное решение без оптимизатора.
    Для каждой эпохи и каждой координаты возвращает min/max replicas.
    """
    epoch_vectors = [np.asarray(v, dtype=float) for v in epoch_vectors]

    if not epoch_vectors:
        return {}

    d = len(epoch_vectors[0])
    if any(len(v) != d for v in epoch_vectors):
        raise ValueError("Все эпохи должны иметь одинаковую размерность")

    if any(np.any(v < 0) for v in epoch_vectors):
        raise ValueError("Координаты эпох должны быть неотрицательны")

    if not (0.0 <= tolerance <= 1.0):
        raise ValueError("tolerance должен быть в [0,1]")

    S_max = max(np.sum(v) for v in epoch_vectors)
    S_min = min_scale * S_max

    result = {}

    for k, epoch in enumerate(epoch_vectors):
        min_rep = np.zeros(d, dtype=int)
        max_rep = np.zeros(d, dtype=int)
        min_frac = np.zeros(d, dtype=float)
        max_frac = np.zeros(d, dtype=float)

        for i in range(d):
            interval = coordinate_fraction_range(epoch, i, tolerance)
            if interval is None:
                raise ValueError(
                    f"Для эпохи {k} и координаты {i} допустимых значений не найдено"
                )

            a_min, a_max = interval
            min_frac[i] = a_min
            max_frac[i] = a_max

            min_rep[i] = int(math.floor(S_min * a_min))
            max_rep[i] = int(math.floor(S_max * a_max))

        result[f"epoch_{k}"] = {
            "epoch_vector": epoch.astype(int),
            "min_fraction": min_frac,
            "max_fraction": max_frac,
            "min_replicas": min_rep,
            "max_replicas": max_rep,
        }

    return {
        "max_total_replicas": int(math.floor(S_max)),
        "min_total_replicas": int(math.floor(S_min)),
        "epochs": result,
    }


# epoch = [[23, 3 ,13, 16]]
epoch = [[10,0,0,0]]
# s_max = sum(epoch)
# s_min = 0.8 * s_max

tolerance = 0.99

result = compute_minmax_replicas_range_closed_form(epoch, tolerance)

print(result)
