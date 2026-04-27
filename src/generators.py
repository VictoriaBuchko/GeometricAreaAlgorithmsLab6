import math
import random
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def generate_polygon(num_points: int, radius: float = 10.0, irregularity: float = 0.35) -> Polygon:
    """
    Генерує випадковий полігон без самоперетинів.

    Алгоритм:
    1. Генеруються випадкові кути (від 0 до 2pi).
    2. Кути сортуються.
    3. Для кожного кута генерується випадковий радіус.
    4. Полярні координати перетворюються в Декартові (x, y).

    Args:
        num_points (int): Кількість вершин.
        radius (float): Середній радіус описуючого кола.
        irregularity (float): Коефіцієнт "випадковості" радіуса (0.0 - ідеальне коло, 1.0 - сильний розкид).

    Returns:
        Polygon: Об'єкт полігону Shapely.
    """
    angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_points)])

    points = []
    for angle in angles:
        r = radius * (1 - irregularity * random.random())
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        points.append((x, y))

    poly = Polygon(points)

    if not poly.is_valid:
        return poly.buffer(0)

    return poly


def visualize_polygon(polygon: Polygon, title: str = None, filename: str = None):
    """
    Візуалізує полігон. Якщо вказано filename, зберігає у файл.
    """
    x, y = polygon.exterior.xy

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, color='blue', linewidth=2, zorder=1)
    plt.fill(x, y, color='skyblue', alpha=0.3)
    plt.scatter(x, y, color='red', s=20, zorder=2)

    n_vertices = len(polygon.exterior.coords) - 1
    plt.title(title or f"Polygon (Vertices: {n_vertices})")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Зображення збережено: {filename}")
        plt.close()
    else:
        plt.show()


def visualize_multiple_polygons(polygons: list, titles: list, filename: str = None):
    """
    Візуалізує декілька полігонів на одному рисунку.
    """
    n = len(polygons)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))

    for ax, poly, title in zip(axes, polygons, titles):
        x, y = poly.exterior.xy
        ax.plot(x, y, color='blue', linewidth=2, zorder=1)
        ax.fill(x, y, color='skyblue', alpha=0.3)
        ax.scatter(x, y, color='red', s=15, zorder=2)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Зображення збережено: {filename}")
        plt.close()
    else:
        plt.show()


# --- Тестовий запуск ---
if __name__ == "__main__":
    random.seed(42)
    try:
        my_poly = generate_polygon(num_points=15, radius=50)
        print(f"Test Polygon Area (Shapely): {my_poly.area}")
        visualize_polygon(my_poly)
    except Exception as e:
        print(f"Error: {e}")