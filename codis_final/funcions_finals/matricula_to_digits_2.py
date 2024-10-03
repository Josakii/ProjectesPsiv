import cv2  # Per al processament d'imatges i contorns
import os   # Per a la gestió d'arxius i directoris
from PIL import Image, ImageDraw, ImageFont  # Per a la manipulació d'imatges amb Pillow

def crop_plate_numbers(image):
    if type(image) == str:
        # Carregar imatge des del camí
        image = cv2.imread(image)

    # Convertir imatge a espai HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Obtenir el canal de valor (V) del HSV
    v_channel = hsv[:, :, 2]

    # Binaritzar la imatge utilitzant el mètode d'Otsu
    _, binary_image = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Copiar la imatge original per a no modificar-la
    gray_image = image.copy()

    # Convertir la imatge a escala de grisos
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    # Trobar els contorns a la imatge binaritzada
    contours, _ = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar els contorns d'esquerra a dreta
    contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])

    # Establir els llindars per a l'àrea dels contorns
    area_min = 400
    area_max = 900

    # Llista per a emmagatzemar els retalls dels números/lletres
    digits = []

    # Dibuixar un rectangle sobre cada contorn i guardar els retalls
    for contour in contours:
        # Obtenir les coordenades del rectangle que envolta el contorn
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Filtrar per àrea (massa petit o massa gran)
        if area_min < area < area_max:
            # Filtrar contorns que toquen les vores (posició en els 0)
            if x > 0 and y > 0 and (x + w) < gray_image.shape[1] and (y + h) < gray_image.shape[0]:
                # Retallar el contorn (número o lletra)
                digit = gray_image[y:y + h, x:x + w]
                digits.append(digit)

    return digits

import cv2
import numpy as np

import cv2
import numpy as np

def resize_with_padding(image, size=(40, 64)):
    """Redimensionar la imatge en format OpenCV a una mida específica amb farcit."""
    # Obtenir dimensions originals
    h_orig, w_orig = image.shape[:2]
    aspect_ratio = w_orig / h_orig
    
    # Calcular la mida nova mantenint la relació d'aspecte
    if aspect_ratio > (size[0] / size[1]):  # Imatge més ampla que la mida desitjada
        new_w = size[0]
        new_h = int(new_w / aspect_ratio)
    else:  # Imatge més alta que la mida desitjada
        new_h = size[1]
        new_w = int(new_h * aspect_ratio)

    # Redimensionar la imatge
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Crear una nova imatge en blanc (farcit) en format color
    final_img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255  # Fons blanc

    # Calcular la posició per centrar la imatge redimensionada
    x_offset = (size[0] - new_w) // 2
    y_offset = (size[1] - new_h) // 2

    # Si resized_img és una imatge en escala de grisos (2D), la convertim a 3 canals (RGB)
    if len(resized_img.shape) == 2:  # Si és 2D, és una imatge en escala de grisos
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)  # Convertir a RGB

    # Enganxar la imatge redimensionada en la nova imatge
    final_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
    
    return final_img



def save_cropped_resized_images(input_folder, output_folder):
    """Retalla i redimensiona imatges dins d'una carpeta."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for file in os.listdir(input_folder):
        if file.endswith(('.png', '.jpg', '.jpeg')):  # Filtrar imatges
            # Carregar la imatge
            img_path = os.path.join(input_folder, file)
            img = Image.open(img_path)

            # Redimensionar la imatge amb farcit
            resized_img = resize_with_padding(img, size=(40, 64))
            
            # Guardar la imatge redimensionada
            resized_img.save(os.path.join(output_folder, file))


def matricula_to_digits(image, size=(40, 64)):
    """
    Procesa una imagen de matrícula: recorta los números/letras, los redimensiona con relleno
    y devuelve una lista de imágenes en formato cv2
    """
    # Recortar los números de la matrícula usando la primera función
    digits = crop_plate_numbers(image)
    digits_final = []
    for d in digits:
        # Redimensionar la imagen con relleno usando la segunda función
        digit_resized = resize_with_padding(d, size)

        # Añadir a la lista final
        digits_final.append(digit_resized)

    return digits_final
