import zipfile
import pandas as pd
from scipy import stats
import numpy as np
import os

# Función para calcular accuracy, precision y f1-score
def calculate_metrics(tp, fp, fn, tn):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return accuracy, precision, f1_score

def read_txt_files(directory):
    data = []
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
                    
                    # Calcular accuracy, precision y f1-score
                    accuracy, precision, f1_score = calculate_metrics(tp, fp, fn, tn)
                    data.append((filename, tp, fp, fn, tn, accuracy, precision, f1_score))
                    print(data)
                except (ValueError, IndexError) as e:
                    print(f"Error al procesar el archivo {filename}: {e}. Se omitirá.")
    
    return data

# Función para calcular los intervalos de confianza para cada métrica
def calculate_confidence_interval(data):
    accuracies = [entry[5] for entry in data]
    precisions = [entry[6] for entry in data]
    f1_scores = [entry[7] for entry in data]
    
    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_f1_score = np.mean(f1_scores)
    
    sem_accuracy = stats.sem(accuracies)
    sem_precision = stats.sem(precisions)
    sem_f1_score = stats.sem(f1_scores)
    
    confidence_interval_accuracy = stats.t.interval(0.95, len(accuracies) - 1, loc=mean_accuracy, scale=sem_accuracy)
    confidence_interval_precision = stats.t.interval(0.95, len(precisions) - 1, loc=mean_precision, scale=sem_precision)
    confidence_interval_f1_score = stats.t.interval(0.95, len(f1_scores) - 1, loc=mean_f1_score, scale=sem_f1_score)
    
    return {
        'mean_accuracy': mean_accuracy,
        'confidence_interval_accuracy': confidence_interval_accuracy,
        'mean_precision': mean_precision,
        'confidence_interval_precision': confidence_interval_precision,
        'mean_f1_score': mean_f1_score,
        'confidence_interval_f1_score': confidence_interval_f1_score
    }

# Función para guardar los resultados en un archivo Excel
def save_to_excel(data, output_file):
    # Crear un DataFrame y guardar en Excel
    df = pd.DataFrame(data, columns=['Archivo', 'TP', 'FP', 'FN', 'TN', 'Accuracy', 'Precision', 'F1-Score'])
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
    
    # Calcular intervalos de confianza para accuracy, precision y f1-score
    metrics = calculate_confidence_interval(data)
    
    # Guardar resultados en Excel
    save_to_excel(data, output_file)
    
    print(f'Accuracy medio: {metrics["mean_accuracy"]:.4f}')
    print(f'Intervalo de confianza (95% - Accuracy): {metrics["confidence_interval_accuracy"][0]:.4f} - {metrics["confidence_interval_accuracy"][1]:.4f}')
    print(f'Precision media: {metrics["mean_precision"]:.4f}')
    print(f'Intervalo de confianza (95% - Precision): {metrics["confidence_interval_precision"][0]:.4f} - {metrics["confidence_interval_precision"][1]:.4f}')
    print(f'F1-Score medio: {metrics["mean_f1_score"]:.4f}')
    print(f'Intervalo de confianza (95% - F1-Score): {metrics["confidence_interval_f1_score"][0]:.4f} - {metrics["confidence_interval_f1_score"][1]:.4f}')

if __name__ == '__main__':
    main()
