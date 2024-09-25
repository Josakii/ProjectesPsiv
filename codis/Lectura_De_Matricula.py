import sys
from pathlib import Path

# Agregar la ruta al repositorio YOLOv5 al path
yolo_repo_path = 'C:/Users/Josep/Desktop/Matriculas Detection/yolov5'
sys.path.append(yolo_repo_path)

import torch
from models.common import DetectMultiBackend

# Asegúrate de usar la ruta como cadena para compatibilidad con Windows
pt_model_path = 'C:/Users/Josep/Desktop/Matriculas Detection/weights/best.pt'

# Cargar el modelo YOLOv5
model = DetectMultiBackend(pt_model_path)  # Pasar la ruta directamente como cadena
model.eval()  # Poner el modelo en modo evaluación
print("Modelo YOLOv5 cargado:", model)

# Preprocesamiento de imágenes
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

# Función para hacer predicciones con YOLOv5
def predict_image_yolo(image_path):
    image = Image.open(image_path)
    input_tensor = preprocess(image).unsqueeze(0)  # Añadir la dimensión de batch

    # Realizar la predicción
    with torch.no_grad():
        outputs = model(input_tensor)

    return outputs

# Ejemplo de uso
image_path = 'C:/Users/Josep/Desktop/Matriculas Detection/images/train/sample.jpg'
outputs = predict_image_yolo(image_path)
print("Predicciones:", outputs)
