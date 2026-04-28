import random
import time
import os

import numpy as np
import matplotlib.pyplot as plt

from generators import generate_polygon, visualize_multiple_polygons
from algorithms import area_shapely, area_gauss, area_monte_carlo, relative_error

#константи
SEED = 42
#шлях до папки images відносно файлу main.py
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


#генерація та візуалізація полігонів
def task1_generate_and_visualize():
    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 1: Генерація та візуалізація полігонів")
    print("=" * 60)

    vertex_counts = [10, 50, 100]
    polygons = []
    titles = []

    for n in vertex_counts:
        poly = generate_polygon(num_points=n, radius=10.0)
        polygons.append(poly)
        titles.append(f"N = {n} вершин\nПлоща (Shapely): {poly.area:.2f}")
        print(f"Полігон N={n}: площа (Shapely) = {poly.area:.4f}")

    filename = os.path.join(IMAGES_DIR, 'polygon_example.png')
    visualize_multiple_polygons(polygons, titles, filename=filename)

    return polygons


#реалізація алгоритмів та перевірка
def task2_algorithms(polygons: list):
    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 2: Порівняння методів обчислення площі")
    print("=" * 60)

    vertex_counts = [10, 50, 100]
    for poly, n in zip(polygons, vertex_counts):
        s_shapely = area_shapely(poly)
        s_gauss = area_gauss(poly)
        s_mc = area_monte_carlo(poly, num_points=10000)

        err_gauss = relative_error(s_gauss, s_shapely)
        err_mc = relative_error(s_mc, s_shapely)

        print(f"\nПолігон N={n}:")
        print(f"   Shapely  : {s_shapely:.6f}")
        print(f"   Гаус     : {s_gauss:.6f}  (похибка: {err_gauss:.6f}%)")
        print(f"   МонтеКарло(M=10000): {s_mc:.6f}  (похибка: {err_mc:.2f}%)")


#дослідження точності монте-Карло
def task3_monte_carlo_accuracy(polygon_n50):
    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 3: Дослідження точності методу Монте-Карло (N=50)")
    print("=" * 60)

    m_values = [100, 500, 1000, 5000, 10000, 50000, 100000]
    reference = area_shapely(polygon_n50)
    errors = []

    for m in m_values:
        s_mc = area_monte_carlo(polygon_n50, num_points=m)
        err = relative_error(s_mc, reference)
        errors.append(err)
        print(f"   M={m:>7}: S_MC = {s_mc:.4f}, похибка = {err:.4f}%")

    #побудова графіка залежності похибки від кількості точок
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(m_values, errors, marker='o', color="#AB412E", linewidth=2, markersize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Кількість точок M (log-шкала)', fontsize=12)
    ax.set_ylabel('Відносна похибка δ, %', fontsize=12)
    ax.set_title('Залежність похибки Монте-Карло від кількості точок', fontsize=13)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    plt.tight_layout()
    filename = os.path.join(IMAGES_DIR, 'error_plot.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nГрафік збережено: {filename}")
    plt.close()


#бенчмарк продуктивності з логарифмічною шкалою
def task4_benchmark():
    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 4: Бенчмарк продуктивності (час виконання)")
    print("=" * 60)

    vertex_counts = [10, 50, 100, 1000]
    mc_points = 100000
    repeats = 20

    results = []

    header = f"{'N вершин':>10} | {'Shapely (мс)':>14} | {'Гаус (мс)':>12} | {'М-Карло (мс)':>14}"
    print(header)
    print("-" * len(header))

    for n in vertex_counts:
        poly = generate_polygon(num_points=n, radius=10.0)

        #вимірювання часу для Shapely
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = poly.area
        t_shapely = (time.perf_counter() - t0) / repeats * 1000

        #вимірювання часу для методу Гауса
        t0 = time.perf_counter()
        for _ in range(repeats):
            area_gauss(poly)
        t_gauss = (time.perf_counter() - t0) / repeats * 1000

        #монте-Карло запускається один раз, бо він значно повільніший
        t0 = time.perf_counter()
        area_monte_carlo(poly, num_points=mc_points)
        t_mc = (time.perf_counter() - t0) * 1000

        results.append((n, t_shapely, t_gauss, t_mc))
        print(f"{n:>10} | {t_shapely:>14.6f} | {t_gauss:>12.6f} | {t_mc:>14.4f}")

    #побудова графіка порівняння продуктивності
    ns = [r[0] for r in results]
    t_sh = [r[1] for r in results]
    t_ga = [r[2] for r in results]
    t_mc = [r[3] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(ns, t_sh, marker='o', label='Shapely', linewidth=2)
    plt.plot(ns, t_ga, marker='s', label='Метод Гауса', linewidth=2)
    plt.plot(ns, t_mc, marker='^', label=f'Монте-Карло (M={mc_points})', linewidth=2)

    #логарифмічна шкала для осі Y для кращої читабельності
    plt.yscale('log')

    plt.xlabel('Кількість вершин', fontsize=12)
    plt.ylabel('Час виконання (мс) - лог. шкала', fontsize=12)
    plt.title('Порівняння продуктивності методів', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    plt.tight_layout()
    filename = os.path.join(IMAGES_DIR, 'time_benchmark.png')
    plt.savefig(filename, dpi=150)
    print(f"\nГрафік швидкодії збережено (лог-шкала): {filename}")
    plt.close()

    return results


#головна функція
def main():
    os.system('cls' if os.name == 'nt' else 'clear')  #очищення консолі перед запуском
    print("ЛАБОРАТОРНА РОБОТА: Обчислення площі полігонів")
    print("Методи: Shapely, Гаус, Монте-Карло\n")

    #генерація полігонів
    polygons = task1_generate_and_visualize()

    #перевірка алгоритмів
    task2_algorithms(polygons)

    #точність монте-Карло (беремо полігон на 50 вершин)
    task3_monte_carlo_accuracy(polygons[1])

    #бенчмарк продуктивності
    task4_benchmark()

    print("\n" + "=" * 60)
    print("УСІ ЗАВДАННЯ ВИКОНАНО УСПІШНО!")
    print("Результати в /images")
    print("=" * 60)


if __name__ == "__main__":
    main()
