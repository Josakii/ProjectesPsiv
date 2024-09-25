import sys
from pathlib import Path

# Agregar la ruta al repositorio YOLOv5 al path
yolo_repo_path = 'C:/Users/Josep/Desktop/Matriculas Detection/yolov5'
sys.path.append(yolo_repo_path)

import torch
from models.common import DetectMultiBackend


pt_model_path = 'C:/Users/Josep/Desktop/Matriculas Detection/weights/best.pt'

# Cargar el modelo YOLOv5
model = DetectMultiBackend(pt_model_path)  # Pasar la ruta directamente como cadena
model.eval()  
print("Modelo YOLOv5 cargado:", model)


from PIL import Image
import numpy as np
import torchvision.transforms as transforms

preprocess = transforms.Compose([
    transforms.ToTensor(),
])


def predict_image_yolo(image_path):
    image = Image.open(image_path)
    input_tensor = preprocess(image).unsqueeze(0)  # Añadir la dimensión de batch


    with torch.no_grad():
        outputs = model(input_tensor)

    return outputs


image_path = 'C:/Users/Josep/Desktop/Matriculas Detection/images/train/sample.jpg'
outputs = predict_image_yolo(image_path)
print("Predicciones:", outputs)
