import math
from itertools import combinations

import numpy as np
from matplotlib import pyplot as plt


def compute_weighted_ranges_simple(epoch_vectors, weights, tolerance):
    epoch_vectors = [np.asarray(v, dtype=float) for v in epoch_vectors]
    weights = np.asarray(weights, dtype=float)

    sin_alpha = math.sqrt(max(0.0, 1.0 - tolerance**2))
    result = {}

    for k, epoch in enumerate(epoch_vectors):
        r = sin_alpha * np.sum(epoch * weights)
        delta = np.ceil (r / weights).astype(int)
        sum_limits = np.sum(epoch * weights)

        result[f"epoch_{k}"] = {
            "sum_limits" : sum_limits,
            "limits" : weights,
            "radius" : r,
            "epoch_vector": epoch.astype(int),
            "delta": delta,
            "min_replicas": np.maximum(epoch - delta, 0).astype(int),
            "max_replicas": (epoch + delta).astype(int),
        }

    return result

def pretty_print_weighted_ranges_simple(result, decimals=4):
    print("=" * 72)
    print("WEIGHTED RANGE RESULT")
    print("=" * 72)
    print()

    for epoch_name, data in result.items():
        print("-" * 72)
        print(epoch_name.upper())
        print("-" * 72)

        print(f"epoch_vector = {data['epoch_vector']}")
        print(f"sumLimit = {data['sum_limits']}")
        print(f"limits = {data['limits']}")
        print(f"delta        = {np.array2string(data['delta'], precision=decimals)}")
        print(f"radius       =  {data['radius']}")
        print()

        print("ranges:")
        for i, (mn, mx) in enumerate(zip(data["min_replicas"], data["max_replicas"])):
            width = mx - mn
            print(f"  coord[{i}] : [{mn}, {mx}]  (width={width})")
        print()


epoch_vectors = [
    [21, 25, 5, 6],
    [11, 19, 1, 26],
    [42, 3, 6, 9],
    [8, 34, 5, 11],
]

weights = [400, 700, 300, 1200]
tolerance = 0.99

result = compute_weighted_ranges_simple(
    epoch_vectors=epoch_vectors,
    weights=weights,
    tolerance=tolerance,
)


from itertools import combinations
import math
import numpy as np
import matplotlib.pyplot as plt


def plot_all_2d_projections(result):
    for epoch_name, data in result.items():
        epoch = np.asarray(data["epoch_vector"], dtype=float)
        delta = np.asarray(data["delta"], dtype=float)
        mins = np.asarray(data["min_replicas"], dtype=float)
        maxs = np.asarray(data["max_replicas"], dtype=float)
        d = len(epoch)

        pairs = list(combinations(range(d), 2))
        n_plots = len(pairs)

        # размер сетки
        cols = math.ceil(math.sqrt(n_plots))
        rows = math.ceil(n_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.array(axes).reshape(-1)

        for idx, (i, j) in enumerate(pairs):
            ax = axes[idx]
            ax.set_xlim(0, 60)
            ax.set_ylim(0, 60)
            x = epoch[i]
            y = epoch[j]
            dx = delta[i]
            dy = delta[j]

            xmin, xmax = mins[i], maxs[i]
            ymin, ymax = mins[j], maxs[j]

            # прямоугольник границ
            rect_x = [xmin, xmax, xmax, xmin, xmin]
            rect_y = [ymin, ymin, ymax, ymax, ymin]
            ax.plot(rect_x, rect_y)

            # точка эпохи
            ax.scatter([x], [y])

            # дельты
            ax.hlines(y, xmin, xmax)
            ax.vlines(x, ymin, ymax)

            # крайние точки
            ax.scatter([xmin, xmax], [y, y])
            ax.scatter([x, x], [ymin, ymax])

            # --- две прямые по косинусному отклонению ---
            vec = np.array([x, y], dtype=float)
            norm = np.linalg.norm(vec)

            if norm > 1e-12:
                u = vec / norm
                u_perp = np.array([-u[1], u[0]])
                sin_alpha = math.sqrt(max(0.0, 1.0 - tolerance ** 2))

                d1 = tolerance * u + sin_alpha * u_perp
                d2 = tolerance * u - sin_alpha * u_perp

                # длина прямых до края графика
                L = max(ax.get_xlim()[1], ax.get_ylim()[1])

                # рисуем лучи из начала координат
                ax.plot([0, L * d1[0]], [0, L * d1[1]], linestyle="--")
                ax.plot([0, L * d2[0]], [0, L * d2[1]], linestyle="--")

            # подписи
            # ax.set_title(f"({i}, {j})")
            ax.set_xlabel(f"coord[{i}]")
            ax.set_ylabel(f"coord[{j}]")

            ax.grid(True, alpha=0.3)

        # удалить лишние subplot если есть
        for idx in range(len(pairs), len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle(f"{epoch_name}\n"
    f"E = {data['epoch_vector']}, tolerance = {tolerance} \n limits = {weights}"
        , fontsize=16)
        plt.tight_layout()
        plt.show()

pretty_print_weighted_ranges_simple(result)
plot_all_2d_projections(result)