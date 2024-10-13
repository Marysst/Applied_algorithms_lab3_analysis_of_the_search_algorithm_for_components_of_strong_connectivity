# Бібліотека abc для реалізації абстрактного класу Graph
from abc import ABC
# Модуль потрібний для генерації випадкових графів у моделі Ердеша-Реньї
import random

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
