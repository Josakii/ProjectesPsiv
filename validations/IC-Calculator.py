import os
import zipfile
import pandas as pd
from scipy import stats
import numpy as np

# Función para calcular el accuracy y el intervalo de confianza
def calculate_metrics(tp, fp, fn, tn):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return accuracy

def read_txt_files(directory):
    data=[]
        # Leer todos los archivos .txt en el directorio especificado
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                
                # Verificar que el archivo tiene al menos 4 líneas
                if len(lines) < 4:
                    print(f"El archivo {filename} no tiene suficientes líneas. Se omitirá.")
                    continue
                
                try:
                    # Extraer los valores de TP, FP, FN, TN
                    tp = int(lines[1].split(': ')[1])  # Cambia el índice a 1 para TP
                    fp = int(lines[2].split(': ')[1])  # Cambia el índice a 2 para FP
                    fn = int(lines[3].split(': ')[1])  # Cambia el índice a 3 para FN
                    tn = int(lines[4].split(': ')[1])  # Cambia el índice a 4 para TN
                    
                    # Calcular accuracy
                    accuracy = calculate_metrics(tp, fp, fn, tn)
                    data.append((filename, tp, fp, fn, tn, accuracy))
                    print(data)
                except (ValueError, IndexError) as e:
                    print(f"Error al procesar el archivo {filename}: {e}. Se omitirá.")
    
    return data

def calculate_confidence_interval(data):
    accuracies = [entry[5] for entry in data]
    mean_accuracy = np.mean(accuracies)
    sem = stats.sem(accuracies)  # Error estándar de la media
    confidence_interval = stats.t.interval(0.95, len(accuracies)-1, loc=mean_accuracy, scale=sem)
    
    return mean_accuracy, confidence_interval

def save_to_excel(data, output_file):
    # Crear un DataFrame y guardar en Excel
    df = pd.DataFrame(data, columns=['Archivo', 'TP', 'FP', 'FN', 'TN', 'Accuracy'])
    df.to_excel(output_file, index=False)

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main():
    zip_path = 'C:/Users/Josep/Desktop/Nueva carpeta/ProjectesPsiv-1/validations/Letters_SVM/results_CFM_SVM_Letters.zip'
    extract_to = 'C:/Users/Josep/Desktop/Nueva carpeta/ProjectesPsiv-1/validations/Letters_SVM'  
    output_file = 'resultats.xlsx'
    
    # Descomprimir el archivo ZIP
    unzip_file(zip_path, extract_to)
    
    # Leer archivos de texto descomprimidos
    data = read_txt_files(extract_to)
    mean_accuracy, confidence_interval = calculate_confidence_interval(data)
    
    # Guardar resultados en Excel
    save_to_excel(data, output_file)
    
    print(f'Accuracy medio: {mean_accuracy:.4f}')
    print(f'Intervalo de confianza (95%): {confidence_interval[0]:.4f} - {confidence_interval[1]:.4f}')

if __name__ == '__main__':
    main()
