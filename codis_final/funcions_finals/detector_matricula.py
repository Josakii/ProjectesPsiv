# IMPORT FUNCIONS INTERMITJES
from funcions_finals.matricula_to_digits_2 import matricula_to_digits
from funcions_finals.digits_to_predicts_3 import predict_digits, CNNModel_a, CNNModel_n
from funcions_finals.image_to_matricula_1 import crop_matricula

# IMPORTS MODULS
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


def init_models():
    # Inicializar los modelos
    model_alfa = CNNModel_a()
    model_num = CNNModel_n()

    # Cargar los estados de los modelos
    model_alfa.load_state_dict(torch.load("models/CNN1-alfa.pt"))
    model_num.load_state_dict(torch.load("models/CNN1-numeros.pt"))

    # Cambiar a modo de evaluación
    model_alfa.eval()
    model_num.eval()

    return model_alfa, model_num

def detect_matricula(img_path):

    # 1. CROP MATRICULA
    cropped_m = crop_matricula(img_path)

    # 2. MATRICULA TO DIGITS
    digits = matricula_to_digits(cropped_m)
    num_digits = digits[0:4]
    alfa_digits = digits[4:7]

    # 3. PREDICT DIGITS
    model_alfa, model_num = init_models()
    num_preds, alfa_preds = predict_digits(num_digits, alfa_digits,model_num,model_alfa)

    return num_preds, alfa_preds

