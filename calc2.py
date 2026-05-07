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
epoch = [
    [21, 25, 5, 6],
    [11, 19, 1, 26],
    # [42, 3, 6, 9],
    # [8, 34, 5, 11],
]
# s_max = sum(epoch)
# s_min = 0.8 * s_max

weights = [400, 400, 100, 1200]

tolerance = 0.99

result = compute_minmax_replicas_range_closed_form(epoch, tolerance)

# print(result)
def pretty_print_ranges(result):
    print("=" * 60)
    print("GLOBAL LIMITS")
    print("=" * 60)
    print(f"max_total_replicas = {result['max_total_replicas']}")
    print(f"min_total_replicas = {result['min_total_replicas']}")
    print()

    for epoch_name, data in result["epochs"].items():
        print("=" * 60)
        print(f"{epoch_name.upper()}")
        print("=" * 60)

        print("epoch_vector :", data["epoch_vector"])
        print()

        print("fractions (доли):")
        for i, (mn, mx) in enumerate(zip(data["min_fraction"], data["max_fraction"])):
            print(f"  coord[{i}] : [{mn:.4f}, {mx:.4f}]")

        print()
        print("replicas (целые):")
        for i, (mn, mx) in enumerate(zip(data["min_replicas"], data["max_replicas"])):
            print(f"  coord[{i}] : [{mn}, {mx}]")

        print()


pretty_print_ranges(result)
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import combinations


def pretty_print_minmax_closed_form(result):
    print(f"max_total_replicas = {result['max_total_replicas']}")
    print(f"min_total_replicas = {result['min_total_replicas']}")
    print()

    for epoch_name, data in result["epochs"].items():
        print(epoch_name)
        print(f"  epoch_vector   = {data['epoch_vector']}")
        print(f"  min_fraction   = {np.round(data['min_fraction'], 4)}")
        print(f"  max_fraction   = {np.round(data['max_fraction'], 4)}")
        print(f"  min_replicas   = {data['min_replicas']}")
        print(f"  max_replicas   = {data['max_replicas']}")
        print()


def plot_all_2d_projections_closed_form(result, tolerance, padding=3):
    epochs = result["epochs"]
    S_max = result["max_total_replicas"]
    S_min = result["min_total_replicas"]



    for epoch_name, data in epochs.items():
        epoch = np.asarray(data["epoch_vector"], dtype=float)
        mins = np.asarray(data["min_replicas"], dtype=float)
        maxs = np.asarray(data["max_replicas"], dtype=float)
        d = len(epoch)

        pairs = list(combinations(range(d), 2))
        n_plots = len(pairs)

        cols = math.ceil(math.sqrt(n_plots))
        rows = math.ceil(n_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows))
        axes = np.array(axes).reshape(-1)

        # общий верхний предел для осей
        max_axis_value = max(np.max(maxs), np.max(epoch), S_max) + padding


        for idx, (i, j) in enumerate(pairs):
            ax = axes[idx]

            x = epoch[i] * weights[i]
            y = epoch[j] * weights[j]

            xmin, xmax = mins[i] * weights[i], maxs[i] * weights[i]
            ymin, ymax = mins[j] * weights[j], maxs[j] * weights[j]

            # ax.set_xlim(0, min_va)
            # ax.set_ylim(0, max_axis_value)

            # прямоугольник диапазонов
            rect_x = [xmin, xmax, xmax, xmin, xmin]
            rect_y = [ymin, ymin, ymax, ymax, ymin]
            ax.plot(rect_x, rect_y, linewidth=2, label="ranges box")

            # эпоха
            ax.scatter([x], [y], s=80, zorder=5, label="epoch")

            # центральные линии через эпоху
            ax.hlines(y, xmin, xmax, linestyles=":", linewidth=1.8)
            ax.vlines(x, ymin, ymax, linestyles=":", linewidth=1.8)

            # крайние точки
            ax.scatter([xmin, xmax], [y, y], s=50, zorder=5)
            ax.scatter([x, x], [ymin, ymax], s=50, zorder=5)

            # подписи крайних точек
            ax.annotate(f"({int(xmin)}, {int(y)})", (xmin, y),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
            ax.annotate(f"({int(xmax)}, {int(y)})", (xmax, y),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
            ax.annotate(f"({int(x)}, {int(ymin)})", (x, ymin),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
            ax.annotate(f"({int(x)}, {int(ymax)})", (x, ymax),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)

            # --- две прямые по косинусному отклонению (без weights) ---
            vec = np.array([x, y], dtype=float)
            norm = np.linalg.norm(vec)

            if norm > 1e-12:
                u = vec / norm
                u_perp = np.array([-u[1], u[0]])
                sin_alpha = math.sqrt(max(0.0, 1.0 - tolerance ** 2))

                d1 = tolerance * u + sin_alpha * u_perp
                d2 = tolerance * u - sin_alpha * u_perp

                L = max(ax.get_xlim()[1], ax.get_ylim()[1])

                ax.plot([0, L * d1[0]], [0, L * d1[1]], linestyle="--", linewidth=1.5)
                ax.plot([0, L * d2[0]], [0, L * d2[1]], linestyle="--", linewidth=1.5)

            # точка эпохи подписью
            ax.annotate(f"E=({int(x)}, {int(y)})", (x, y),
                        textcoords="offset points", xytext=(7, -12), fontsize=10)

            ax.set_xlabel(f"coord[{i}]")
            ax.set_ylabel(f"coord[{j}]")
            ax.set_title(f"Projection ({i}, {j})")
            ax.grid(True, alpha=0.3)

        # удалить пустые subplot
        for idx in range(len(pairs), len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle(
            f"{epoch_name}\n"
            f"E = {data['epoch_vector']}, tolerance = {tolerance}\n"
            f"min_total = {S_min}, max_total = {S_max}",
            fontsize=16
        )
        plt.tight_layout()
        plt.show()




result = compute_minmax_replicas_range_closed_form(epoch, tolerance, min_scale=0.7)

pretty_print_minmax_closed_form(result)
plot_all_2d_projections_closed_form(result, tolerance)












