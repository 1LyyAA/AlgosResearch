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
    # p1 = p1.astype(int)
    # p2 = p2.astype(int)

    return p1, p2


def plot_all(ax, v, a, A, label_prefix=""):
    v = np.array(v, dtype=float)

    d1, d2 = compute_directions(v, a)

    p1_A, p1_07 = intersection_points(d1, A)
    p2_A, p2_07 = intersection_points(d2, A)

    # радиус-вектор
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1)

    # прямые
    t = np.linspace(-10, 20, 200)
    ax.plot(t * d1[0], t * d1[1])
    ax.plot(t * d2[0], t * d2[1])

    # линии x+y = const
    x = np.linspace(-10, 20, 200)
    ax.plot(x, A - x)
    ax.plot(x, 0.7 * A - x)

    # точки
    points = [p1_A, p1_07, p2_A, p2_07]
    for p in points:
        if p is not None:
            ax.scatter(p[0], p[1])


# ===== Пример =====
fig, ax = plt.subplots()

v = [9,12]
a = 0.9
A = 21

v2 = [2, 0]
a2 = 0.7
A2 = 11

plot_all(ax, v, a, A)
# plot_all(ax, v2, a2, A2)

# оформление один раз
ax.axhline(0)
ax.axvline(0)
ax.set_aspect('equal', adjustable='box')
ax.grid()
ax.set_title("Geometry visualization")



v1, v2 = compute_directions(v, a)
x1 = intersection_points(v1, 13)
x2 = intersection_points(v2, 13)

print(x1, x2)

plt.show()