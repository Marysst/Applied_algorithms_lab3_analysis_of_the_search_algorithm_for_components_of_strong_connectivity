# Бібліотека abc для реалізації абстрактного класу Graph
from abc import ABC
# Модуль потрібний для генерації випадкових графів у моделі Ердеша-Реньї та проведення експеременту
import random
# Бібліотеки для проведення експеременту та візуалізацій його результатів
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- Реалізація орієнтованого графа та алгоритму пошуку компонент сильної зв'язності --------------------
# ---------- Частина коду написана під час виконання лаби 2 ----------
# Абстрактний клас Graph, що містить базові операції для всіх типів графів
class Graph(ABC):
    def __init__(self, n, directed=False):
        """
        Ініціалізує граф з n вершинами.
        n: int - кількість вершин.
        directed: bool - флаг, що вказує, чи є граф орієнтованим.
        """
        self.n = n  # Кількість вершин
        self.directed = directed  # Чи орієнтований граф
        self.graph = {i: [] for i in range(n)}  # Словник списків суміжності для кожної вершини

    def add_vertex(self):
        """Додає нову вершину в граф."""
        self.graph[self.n] = []  # Додає нову вершину зі списком суміжності
        self.n += 1  # Збільшує кількість вершин на 1

    def add_edge(self, u, v, weight=1):
        """
        Додає ребро між вершинами u і v з вагою.
        u: int - початкова вершина.
        v: int - кінцева вершина.
        weight: int - вага ребра (за замовчуванням 1).
        """
        if u < self.n and v < self.n:
            self.graph[u].append((v, weight))  # Додає ребро u -> v
            if not self.directed:  # Якщо граф не орієнтований, додає також ребро v -> u
                self.graph[v].append((u, weight))

    def remove_vertex(self, v):
        """
        Видаляє вершину v і всі її зв'язки.
        v: int - вершина для видалення.
        """
        if v in self.graph:
            del self.graph[v]  # Видаляє вершину зі списку
            # Видаляє всі ребра, що ведуть до цієї вершини
            for u in self.graph:
                self.graph[u] = [edge for edge in self.graph[u] if edge[0] != v]
            self.n -= 1  # Зменшує кількість вершин

    def remove_edge(self, u, v):
        """
        Видаляє ребро між вершинами u і v.
        u: int - початкова вершина.
        v: int - кінцева вершина.
        """
        if u in self.graph:
            self.graph[u] = [edge for edge in self.graph[u] if edge[0] != v]  # Видаляє ребро u -> v
        if not self.directed and v in self.graph:
            self.graph[v] = [edge for edge in self.graph[v] if edge[0] != u]  # Видаляє симетричне ребро v -> u

    def convert_to_adj_matrix(self):
        """
        Конвертує граф у матрицю суміжності.
        """
        matrix = [[0] * self.n for _ in range(self.n)]  # Створює порожню матрицю
        for u in self.graph:
            for v, weight in self.graph[u]:
                matrix[u][v] = weight  # Встановлює вагу ребра в матриці
        return matrix

    def convert_to_adj_list(self):
        """
        Повертає список суміжності графа.
        """
        return self.graph  # Повертає список суміжності

# Клас для орієнтованого графа
class DirectedGraph(Graph):
    def __init__(self, n):
        """
        Ініціалізує орієнтований граф зі списком суміжності.
        """
        super().__init__(n, directed=True)  # Викликає конструктор базового класу з directed=True

# Клас для генерації випадкових графів за моделлю Ердеша-Реньї
class RandomGraph(Graph):
    def __init__(self, n, p, weighted=False, weight_range=(1, 10), directed=False):
        """
        Ініціалізує випадковий граф.
        n: int - кількість вершин.
        p: float - ймовірність існування ребра між вершинами.
        weighted: bool - чи зважений граф (за замовчуванням False).
        weight_range: tuple - межі для ваги ребра.
        directed: bool - чи орієнтований граф (за замовчуванням False).
        """
        super().__init__(n, directed)  # Викликає конструктор базового класу Graph
        self.p = p  # Ймовірність появи ребра
        self.weighted = weighted  # Чи зважений граф
        self.weight_range = weight_range  # Діапазон ваг ребер
        self.generate_random_graph()  # Генерує випадковий граф

    def generate_random_graph(self):
        """Генерує випадковий граф за ймовірністю p."""
        for u in range(self.n):  # Проходить по всім вершинам
            # Якщо граф неорієнтований, додає тільки половину ребер, уникаючи дублікатів
            for v in range(u + 1 if not self.directed else 0, self.n):
                if random.random() < self.p:  # Якщо випадкове число менше ймовірності p
                    weight = random.randint(*self.weight_range) if self.weighted else 1  # Визначає вагу
                    self.add_edge(u, v, weight)  # Додає ребро

# ---------- Частина коду написана саме для цієї лаби ----------
def complete_dfs_with_check(graph, v):
    """
    Виконує пошук в глибину (DFS) для графа з перевіркою всіх вершин при поверненні до стартової вершини.
    graph: об'єкт класу Graph.
    """
    visited = set()  # Множина відвіданих вершин
    pre_order = []  # Прямий порядок відвідування вершин
    post_order = []  # Зворотний порядок після завершення обходу

    def dfs(v):
        """Рекурсивна функція для виконання DFS з вершини v."""
        visited.add(v)
        pre_order.append(v)  # Обходимо в прямому порядку

        # Для кожного сусіда вершини v
        for (neighbour, _) in graph.graph[v]:
            if neighbour not in visited:
                dfs(neighbour)

        post_order.append(v)  # Обходимо в зворотному порядку

    # Запускаємо DFS із кожної вершини, якщо вона ще не відвідана
    for start_vertex in range(graph.n):
        if start_vertex not in visited:
            dfs(start_vertex)

    post_order.reverse()  # Зворотний порядок після завершення обходу

    return post_order

def transpose(graph):
    """
    Транспонує орієнтований граф, змінюючи напрямок усіх ребер.
    graph: об'єкт класу Graph.
    """
    transposed_graph = DirectedGraph(graph.n)  # Створюємо новий орієнтований граф

    # Проходимо по всіх вершинах і змінюємо напрямок ребер
    for u in range(graph.n):
        for (v, weight) in graph.graph[u]:
            transposed_graph.add_edge(v, u, weight)  # Додаємо ребро в зворотному напрямку

    return transposed_graph
  
def find_strongly_connected_components(graph):
    """
    Знаходить компоненти сильної зв'язності графа за допомогою алгоритму Косараджу.
    graph: об'єкт класу Graph.
    """
    # Крок 1: Отримуємо список вершин у зворотному порядку завершення обходу
    post_order = complete_dfs_with_check(graph, 0)
    
    # Крок 2: Транспонуємо граф
    transposed_graph = transpose(graph)
    
    # Крок 3: Виконуємо DFS на транспонованому графі в порядку post_order
    visited = set()  # Множина відвіданих вершин
    scc = []  # Список компонент сильної зв'язності

    def dfs(v, current_scc):
        """Рекурсивна функція для виконання DFS на транспонованому графі."""
        visited.add(v)
        current_scc.append(v)  # Додаємо вершину до поточної компоненти

        # Для кожного сусіда вершини v
        for (neighbour, _) in transposed_graph.graph[v]:
            if neighbour not in visited:
                dfs(neighbour, current_scc)

    # Обходимо вершини в порядку post_order
    for v in post_order:
        if v not in visited:
            current_scc = []  # Поточна компонента сильної зв'язності
            dfs(v, current_scc)
            scc.append(current_scc)  # Додаємо компоненту до списку

    return scc


# -------------------- Реалізація експерименту --------------------

# Функція аналізу компонент сильної зв'язності (проведення екперементу)
def analyze_scc(max_vertices, num_experiments):
    """
    Аналізує випадковий граф за допомогою пошуку компонент сильної зв'язності.
    num_experiments: кількість експериментів для кожної комбінації (n, p).
    """
    probabilities = [round(i * 0.1, 1) for i in range(1, 10)]  # Ймовірності від 0.1 до 0.9
    results = []

    # Визначаємо діапазон для кількості вершин
    vertex_range = range(5, max_vertices + 1)  # Від 5 до заданной кількості

    # Проходимо по кількості вершин
    for num_vertices in vertex_range:
        # Проходимо по ймовірності
        for p in probabilities:
            num_scc_list = []
            max_scc_size_list = []
            avg_scc_size_list = []
            scc_size_variance_list = []

            # Виконуємо експерименти
            for _ in range(num_experiments):
                # Генеруємо випадковий орієнтований граф
                random_graph = RandomGraph(num_vertices, p, directed=True)

                # Знаходимо компоненти сильної зв'язності
                scc = find_strongly_connected_components(random_graph)
                
                # Розрахунок необхідних характеристик
                num_scc = len(scc)  # Кількість компонент сильної зв'язності
                max_scc_size = max(len(c) for c in scc) if scc else 0  # Розмір найбільшої компоненти
                avg_scc_size = np.mean([len(c) for c in scc]) if scc else 0  # Середній розмір компонент
                scc_sizes = [len(c) for c in scc]
                scc_size_variance = np.var(scc_sizes) if scc_sizes else 0  # Дисперсія розмірів компонент

                # Додаємо результати до списків
                num_scc_list.append(num_scc)
                max_scc_size_list.append(max_scc_size)
                avg_scc_size_list.append(avg_scc_size)
                scc_size_variance_list.append(scc_size_variance)

            # Усереднюємо результати для поточної кількості вершин і ймовірності
            results.append({
                "n": num_vertices,
                "p": p,
                "num_scc": np.mean(num_scc_list),
                "max_scc_size": np.mean(max_scc_size_list),
                "avg_scc_size": np.mean(avg_scc_size_list),
                "scc_size_variance": np.mean(scc_size_variance_list)
            })

    return results

# Функція побудови графіків за результатами експерементів (візуалізація результатів експеременту)
def plot_relationships(results):
    """
    Створює накладені графіки для різних характеристик графа.
    results: список словників із результатами експериментів.
    """
    df = pd.DataFrame(results)

    # Унікальні значення ймовірності
    unique_probabilities = df['p'].unique()

    # Графік 1: Кількість компонент
    plt.figure(figsize=(14, 10))
    for fixed_probability in unique_probabilities:
        filtered_df = df[df['p'] == fixed_probability]
        plt.plot(filtered_df['n'], filtered_df['num_scc'], marker='o', label=f'p={fixed_probability}')

    plt.title('Кількість компонент зв\'язності в залежності від кількості вершин')
    plt.xlabel('Кількість вершин')
    plt.ylabel('Кількість компонент')
    plt.legend()
    plt.grid()
    plt.show()

    # Графік 2: Розмір найбільшої компоненти
    plt.figure(figsize=(14, 10))
    for fixed_probability in unique_probabilities:
        filtered_df = df[df['p'] == fixed_probability]
        plt.plot(filtered_df['n'], filtered_df['max_scc_size'], marker='o', label=f'p={fixed_probability}')

    plt.title('Розмір найбільшої компоненти в залежності від кількості вершин')
    plt.xlabel('Кількість вершин')
    plt.ylabel('Розмір найбільшої компоненти')
    plt.legend()
    plt.grid()
    plt.show()

    # Графік 3: Середній розмір компонент
    plt.figure(figsize=(14, 10))
    for fixed_probability in unique_probabilities:
        filtered_df = df[df['p'] == fixed_probability]
        plt.plot(filtered_df['n'], filtered_df['avg_scc_size'], marker='o', label=f'p={fixed_probability}')

    plt.title('Середній розмір компонент в залежності від кількості вершин')
    plt.xlabel('Кількість вершин')
    plt.ylabel('Середній розмір компонент')
    plt.legend()
    plt.grid()
    plt.show()

    # Графік 4: Дисперсія розмірів компонент
    plt.figure(figsize=(14, 10))
    for fixed_probability in unique_probabilities:
        filtered_df = df[df['p'] == fixed_probability]
        plt.plot(filtered_df['n'], filtered_df['scc_size_variance'], marker='o', label=f'p={fixed_probability}')

    plt.title('Дисперсія розмірів компонент в залежності від кількості вершин')
    plt.xlabel('Кількість вершин')
    plt.ylabel('Дисперсія розмірів компонент')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # --- Загальний аналіз від 5 до 50 вершин з кроком 1 ---
    # Виконуємо аналіз
    overall_result = analyze_scc(50, 1000)

    # Створюємо графіки
    plot_relationships(overall_result)

    # --- Деталізованний аналіз від 5 до 20 вершин з кроком 1 ---
    # Виконуємо аналіз
    detailed_result = analyze_scc(20, 1000)

    # Створюємо графіки
    plot_relationships(detailed_result)

if __name__ == "__main__":
    main()
