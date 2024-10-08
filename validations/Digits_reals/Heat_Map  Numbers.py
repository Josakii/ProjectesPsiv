import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Crear una matriz de 10x10 inicializada en 0 (para números del 0 al 9)
heatmap_data = np.zeros((10, 10))

# Diccionario de éxito y fallos para números del 0 al 9
success_data = {
    1: (29, [7]),
    2: (17, [0, 1]),
    3: (27, [1,1]),
    5: (25, [6,6,6,8]),
    6: (31, [0,0,8,8,]),
    7: (34, [1]),
    9: (30, [1,7,7]),
    0: (38,[]),
    4: (33, [0]),
    8: (34, [0,0,0,0]),
}

# Añadir los valores de éxito a la diagonal y repartir fallos
for num, (success, fails) in success_data.items():
    heatmap_data[num, num] = success  # Éxito en la diagonal
    for fail_pos in fails:
        heatmap_data[num, fail_pos] += 1  # Colocar fallos en las posiciones correspondientes

# Crear el heatmap con la paleta de colores 'Blues'
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap='Blues',  # Cambiamos a 'Blues' para el gradiente blanco-azul
            xticklabels=[str(i) for i in range(10)],  # Etiquetas para el eje X (0-9)
            yticklabels=[str(i) for i in range(10)])  # Etiquetas para el eje Y (0-9)

# Mostrar el heatmap
plt.title("Heatmap de Errores Distribuidos entre Números")
plt.show()