import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Directorio que contiene las carpetas con archivos Excel
directorio = 'validations/'  # Cambia esto a tu ruta

# Lista para almacenar los DataFrames y sus nombres de archivo
dataframes = []

# Iterar sobre cada carpeta en el directorio
for carpeta in os.listdir(directorio):
    ruta_carpeta = os.path.join(directorio, carpeta)

    # Verificar que sea un directorio
    if os.path.isdir(ruta_carpeta):
        # Lista para almacenar los archivos Excel en la carpeta actual
        archivos_excel = [f for f in os.listdir(ruta_carpeta) if f.endswith('.xlsx') and not f.startswith('~$')]

        # Iterar sobre cada archivo Excel en la carpeta
        for archivo in archivos_excel:
            # Cargar el archivo Excel
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            try:
                df = pd.read_excel(ruta_archivo)
                print(df)

                # Verificar que la columna 'accuracy' existe
                if 'accuracy' in df.columns:
                    # Agregar el DataFrame y su nombre de archivo a la lista
                    dataframes.append((df, archivo, carpeta))
                else:
                    print(f'El archivo {archivo} en carpeta {carpeta} no contiene la columna "accuracy".')
            except (PermissionError, FileNotFoundError, ValueError) as e:
                print(f'Error al abrir el archivo {archivo}: {e}')

# Crear un gráfico de dispersión para cada DataFrame que tenga la columna 'accuracy'
for df, archivo, carpeta in dataframes:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=df.index, y='accuracy')  # Utiliza el índice del DataFrame para el eje x
    plt.title(f'Gráfico de precisión para {archivo} en carpeta {carpeta}')
    plt.xlabel('Índice del DataFrame')
    plt.ylabel('Accuracy')

    # Guardar el gráfico como un archivo PNG
    nombre_grafico = os.path.join(ruta_carpeta, f'grafico_precision_{archivo}.png')
    plt.savefig(nombre_grafico)
    print(f'Gráfico guardado como {nombre_grafico}')
    
    # Mostrar el gráfico
    plt.show()
