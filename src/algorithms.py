import random
import numpy as np
from shapely.geometry import Polygon, Point


def area_shapely(polygon: Polygon) -> float:
    
    #метод обчислення площі через бібліотеку Shapely (GEOS).

    return polygon.area


def area_gauss(polygon: Polygon) -> float:

    #Обчислення площі полігону методом Гауса (Shoelace / Формула шнурків).

    #Формула: S = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    #де остання точка n збігається з нульовою 0.

    # Отримуємо координати вершин (без повторення першої точки в кінці)
    coords = list(polygon.exterior.coords[:-1])
    n = len(coords)

    area = 0.0
    for i in range(n):
        x_i, y_i = coords[i]
        x_next, y_next = coords[(i + 1) % n]  # наступна вершина (замикає полігон)
        area += (x_i * y_next) - (x_next * y_i)

    return abs(area) / 2.0


def area_monte_carlo(polygon: Polygon, num_points: int = 10000) -> float:
    """
    Обчислення площі полігону методом Монте-Карло.
    Алгоритм:
    1. Визначаємо bounding box полігону.
    2. Генеруємо M випадкових точок всередині прямокутника.
    3. Рахуємо кількість точок K, що потрапили всередину полігону.
    4. S_poly ≈ S_box * (K / M)
    """
    # Bounding box: (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = polygon.bounds
    box_area = (maxx - minx) * (maxy - miny)

    # Генеруємо випадкові точки всередині bounding box
    xs = np.random.uniform(minx, maxx, num_points)
    ys = np.random.uniform(miny, maxy, num_points)

    # Рахуємо кількість точок всередині полігону
    hits = 0
    for x, y in zip(xs, ys):
        if polygon.contains(Point(x, y)):
            hits += 1

    # Площа полігону
    return box_area * (hits / num_points)


def relative_error(estimated: float, reference: float) -> float:
    #Обчислює відносну похибку у відсотках.
    #δ = |S_estimated - S_reference| / S_reference * 100%

    if reference == 0:
        return float('inf')
    return abs(estimated - reference) / reference * 100.0