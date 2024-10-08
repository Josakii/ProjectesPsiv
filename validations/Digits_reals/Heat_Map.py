import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Crear una matriz de 26x26 inicializada en 0
heatmap_data = np.zeros((26, 26))

# Diccionario para convertir letras a índices (A=0, B=1, ..., Z=25)
letter_to_idx = {chr(i): i - 65 for i in range(ord('A'), ord('Z') + 1)}

# Diccionario de éxito y fallos
success_data = {
    'B': (10, [12, 15, 14, 12]),
    'C': (10, [12, 16, 12]),
    'D': (9, []),
    'F': (17, [7]),
    'G': (14, [2, 14]),
    'H': (16, [13, 8]),
    'J': (17, [8, 20]),
    'K': (21, [8, 22, 8, 13, 12, 12]),
    'L': (19, [20]),
    'M': (5, [16, 16]),
    'N': (10, []),
    'P': (10, [1, 8]),
    'R': (10, []),
    'S': (5, [6, 9]),
    'T': (8, [25, 25, 16]),
    'V': (4, []),
    'W': (2, []),
    'X': (6, [12]),
    'Y': (5, []),
    'Z': (5, [])
}

# Añadir los valores de éxito a la diagonal y repartir fallos
for letter, (success, fails) in success_data.items():
    idx = letter_to_idx[letter]
    heatmap_data[idx, idx] = success  # Éxito en la diagonal
    for fail_pos in fails:
        heatmap_data[idx, fail_pos - 1] += 1  # Colocar fallos en las posiciones correspondientes

# Crear el heatmap con la paleta de colores 'Blues'
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='Blues',  # Cambiamos a 'Blues' para el gradiente blanco-azul
            xticklabels=[chr(i) for i in range(ord('A'), ord('Z') + 1)], 
            yticklabels=[chr(i) for i in range(ord('A'), ord('Z') + 1)])

# Mostrar el heatmap
plt.title("Heatmap de Errores Distribuidos entre Letras")
plt.show()
