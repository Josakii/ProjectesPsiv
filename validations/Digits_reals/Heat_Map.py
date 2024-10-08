import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Crear una matriz de 26x26 inicializada en 0
heatmap_data = np.zeros((26, 26))

# Diccionario para convertir letras a índices (A=0, B=1, ..., Z=25)
letter_to_idx = {chr(i): i for i in range(ord('A'), ord('Z') + 1)}

# Añadir los valores de éxito a la diagonal principal
success_data = {
    'B': (10, [12, 15, 14, 12]),   # Todas las posiciones están bien (1-26)
    'C': (10, [12, 16, 12]),        # Todas las posiciones están bien (1-26)
    'D': (9, []),                   # Sin fallas
    'F': (17, [7]),                 # Correcto
    'G': (14, [2, 14]),             # Correcto
    'H': (16, [13, 8]),             # Correcto
    'J': (17, [8, 20]),             # Correcto
    'K': (21, [8, 22, 8, 13, 12, 12]),  # Correcto
    'L': (19, [20]),                # Correcto
    'M': (5, [16, 16]),             # Correcto
    'P': (10, [1, 8]),              # Correcto
    'S': (5, [6, 9]),               # Correcto
    'T': (8, [25, 25, 16]),         # Correcto
    'X': (6, [12]),                 # Correcto
    'Z': (6, [24, 24, 24, 8, 24, 24, 21, 19])  # Correcto
}

# Añadir los valores de éxito a la diagonal y repartir fallos
for letter, (success, fails) in success_data.items():
    print(success)
    idx = letter_to_idx[letter]
    print(letter_to_idx[letter])
    print(idx)
    heatmap_data[idx, idx] = success  # Éxito en la diagonal
    for fail_pos in fails:
        if fail_pos <= 26:  # Verificar que la posición esté en el rango de 1-26
            heatmap_data[idx, fail_pos - 1] += 1  # Ajustamos la posición (fail_pos - 1)
        else:
            print(f"Error: la posición {fail_pos} para la letra {letter} está fuera del rango permitido.")

# Crear el heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', xticklabels=[chr(i) for i in range(ord('A'), ord('Z') + 1)], 
            yticklabels=[chr(i) for i in range(ord('A'), ord('Z') + 1)])

# Mostrar el heatmap
plt.title("Heatmap de Errores Distribuidos entre Letras")
plt.show()
