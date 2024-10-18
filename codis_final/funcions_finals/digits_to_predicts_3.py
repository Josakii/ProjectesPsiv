import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2

# Model numeric
class CNNModel_n(nn.Module):
    def __init__(self):
        super(CNNModel_n, self).__init__()

        # Definir las capas convolucionales y de agrupamiento
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # Capa convolucional 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling 1

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Capa convolucional 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling 2

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Capa convolucional 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling 3

        self.fc1 = nn.Linear(128 * 8 * 5, 128)  # Capa densa
        self.dropout = nn.Dropout(0.5)  # Dropout para regularización
        self.fc2 = nn.Linear(128, 10)  # Capa de salida para 10 clases

    def forward(self, x):
        # Definir el paso hacia adelante
        x = self.pool1(F.relu(self.conv1(x)))  # Capa 1
        x = self.pool2(F.relu(self.conv2(x)))  # Capa 2
        x = self.pool3(F.relu(self.conv3(x)))  # Capa 3

        x = x.view(-1, 128 * 8 * 5)  # Aplanar la salida para la capa densa
        x = F.relu(self.fc1(x))  # Capa densa
        x = self.dropout(x)  # Aplicar dropout
        x = self.fc2(x)  # Capa de salida

        return x
    

# Model alfabètic
class CNNModel_a(nn.Module):
    def __init__(self,p=26):
        super(CNNModel_a, self).__init__()

        # Definir las capas convolucionales y de agrupamiento
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # Capa convolucional 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling 1

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Capa convolucional 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling 2

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Capa convolucional 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling 3

        # Ajustar la capa fully connected para 26 clases
        self.fc1 = nn.Linear(128 * 8 * 5, 128)  # Capa densa
        self.dropout = nn.Dropout(0.5)  # Dropout para regularización
        self.fc2 = nn.Linear(128, p)  # Capa de salida ajustada a 26 clases (puede ser para p=21 en algunos modelos)

    def forward(self, x):
        # Definir el paso hacia adelante
        x = self.pool1(F.relu(self.conv1(x)))  # Capa 1
        x = self.pool2(F.relu(self.conv2(x)))  # Capa 2
        x = self.pool3(F.relu(self.conv3(x)))  # Capa 3

        x = x.view(-1, 128 * 8 * 5)  # Aplanar la salida para la capa densa
        x = F.relu(self.fc1(x))  # Capa densa
        x = self.dropout(x)  # Aplicar dropout
        x = self.fc2(x)  # Capa de salida

        return x
    

def predecir_imagen(model, image):
    # Cargar la imagen desde el path usando OpenCV
    img = image

    # Convertir la imagen de OpenCV a escala de grises si es necesario
    if len(img.shape) == 3:  # Verificar si la imagen tiene 3 canales (RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    
    # Aplicar las transformaciones

    resized_img = cv2.resize(img, (34, 80))  # (ancho, alto)
    img_tensor = torch.tensor(resized_img, dtype=torch.float32)  # Crear tensor
    img_tensor = img_tensor.unsqueeze(0)  # Añadir dimensión de canal
    img_tensor = img_tensor / 255.0  # Normalizar entre 0 y 1
    img_tensor = (img_tensor - 0.5) / 0.5  # Normalizar entre -1 y 1
    
    # Añadir una dimensión adicional para representar el batch (batch_size=1)
    img_tensor = img_tensor.unsqueeze(0)  # Cambia el tamaño a [1, 1, 64, 40]
    
    # Deshabilitar gradientes para hacer predicción (no entrenar)
    with torch.no_grad():
        # Pasar la imagen por el modelo
        outputs = model(img_tensor)
        
        # Obtener la predicción con la mayor probabilidad
        _, predicted = torch.max(outputs, 1)
    
    # Devolver la predicción (como un número de clase)
    return predicted.item()
    

def predict_digits(digits,model_num,model_alfa):

    # separate numbers and letters
    digits_alfa = digits[:3]
    digits_num = digits[3:]

    # Predict numeric
    numeric_preds = []
    for d in digits_num:
        pred = predecir_imagen(model_num, d)
        numeric_preds.append(pred)

    # Predict alfa
    alfa_preds = []
    for d in digits_alfa:
        pred = predecir_imagen(model_alfa, d)
        predicted_letter = chr(pred + ord('A'))
        alfa_preds.append(predicted_letter)

    # Get full string
    full_str = ''
    for i in range(4):
        full_str += str(numeric_preds[3-i])

    for i in range(3):
        full_str += alfa_preds[2-i]


    return numeric_preds, alfa_preds, full_str