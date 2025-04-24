from pyomo.environ import *

from pyomo.opt import SolverFactory

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

# Datos del problema
ciudades = 10
spread = 1.5
vendedores = 2
nodos = [_ for _ in range(ciudades)]

# Cargar datos del csv
filename = "cost_matrix_" + str(ciudades) + "_nodes_" + str(spread) + "_spread.csv"

data = np.loadtxt(filename, delimiter=",", dtype=int)
nodos = list(map(int, data[0]))

graph = {
    (i, j): data[r + 1, c] for r, i in enumerate(nodos) for c, j in enumerate(nodos)
}


# Modelo en Pyomo
Model = ConcreteModel()

V = {_ for _ in nodos}
A = {(i, j) for i in nodos for j in nodos if i != j}
K = {_ for _ in range(vendedores)}

# Variable de decision
Model.x = Var(A, K, within=Binary)
Model.u = Var(V, K, within=NonNegativeIntegers, bounds=(1, ciudades - 1))

# Función objetivo: minimizar la distancia total
Model.obj = Objective(
    expr=sum(graph[i, j] * Model.x[i, j, k] for i, j in A for k in K), sense=minimize
)


# Cada ciudad de los tours es visitada una vez
Model.res1 = ConstraintList()
for j in nodos[1:]:  # Excluir la ciudad 0
    Model.res1.add(sum(Model.x[i, j, k] for i in nodos if i != j for k in K) == 1)

# Se sale de 0 una vez
Model.res2 = ConstraintList()
for k in K:
    Model.res2.add(sum(Model.x[0, j, k] for j in nodos[1:]) == 1)

# Se retorna a 0 una vez
Model.res3 = ConstraintList()
for k in K:
    Model.res3.add(sum(Model.x[i, 0, k] for i in nodos[1:]) == 1)

# Cada vendedor que llega a una ciudad debe salir de ella
Model.res4 = ConstraintList()
for j in nodos[1:]:  # Excluir la ciudad 0
    for k in K:
        Model.res4.add(
            sum(Model.x[i, j, k] for i in nodos if i != j)
            == sum(Model.x[j, i, k] for i in nodos if i != j)
        )

# Eliminación de subtours mtz
Model.res5 = ConstraintList()
for k in K:
    for i in nodos[1:]:  # Excluir la ciudad 0
        for j in nodos[1:]:  # Excluir la ciudad 0
            if i != j:
                Model.res5.add(
                    Model.u[i, k] - Model.u[j, k] + ciudades * Model.x[i, j, k]
                    <= ciudades - 1
                )


# Especificacion del solver
SolverFactory("glpk").solve(Model)
Model.display()

# Imprimir rutas
for k in K:
    print("Vendedor " + str(k) + ": ")
    route = [0]
    current_city = 0
    visited = set()
    while len(visited) < ciudades:
        for i, j in A:
            if Model.x[i, j, k].value > 0.5 and i == current_city and j not in visited:
                route.append(j)
                visited.add(j)
                current_city = j
                break
        if current_city == 0:
            break
    print(" -> ".join(map(str, route)))

print("Distancia minima: " + str(value(Model.obj)))

# Crear el grafo
G = nx.DiGraph()
G.add_nodes_from(nodos)

# Obtener posiciones aleatorias para las ciudades (solo para visualización)
random.seed(42)
pos = {node: (random.random(), random.random()) for node in nodos}

# Añadir aristas según la solución del modelo
colors = ["red", "blue", "green", "purple", "orange"]  # Asignar colores a vendedores
edges = []
edge_colors = []

for k in K:
    for i, j in A:
        if Model.x[i, j, k].value > 0.5:
            edges.append((i, j))
            edge_colors.append(colors[k % len(colors)])

# Dibujar nodos y etiquetas
plt.figure(figsize=(8, 6))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightgray",
    node_size=700,
    edge_color="black",
    width=0.5,
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=edges,
    edge_color=edge_colors,
    width=2,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=15,
)

plt.title("Rutas asignadas a los vendedores")
plt.show()