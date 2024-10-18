import cv2  # Per al processament d'imatges i contorns
import os   # Per a la gestió d'arxius i directoris
from PIL import Image, ImageDraw, ImageFont  # Per a la manipulació d'imatges amb Pillow
import numpy as np  # Per a la manipulació d'imatges amb NumPy

## funcio auxiliar per sobreposicions ##
def is_inside(contour1, contour2):
    """ Verifica si contour1 está completamente dentro de contour2 """
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    
    return (x1 > x2 and y1 > y2 and (x1 + w1) < (x2 + w2) and (y1 + h1) < (y2 + h2))

def find_contours_mat(binary_image,image = None, w_min = 13, w_max = 79, h_min = 26, h_max = 119):
    # Encontrar contornos en la imagen binarizada
    contours_letters, _ = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    filt_contours = []
    for contour in contours_letters:
            x, y, w, h = cv2.boundingRect(contour)
            # filtrem per alçada i amplada
            if w_min < w < w_max and h_min < h < h_max:
                    filt_contours.append(contour)

    # Crear una lista de contornos a eliminar
    contours_to_remove = set()

    # Eliminar contornos sobrepuestos y duplicados
    for i, cnt in enumerate(filt_contours):
        # Si el índice de cnt ya está marcado para eliminar, se salta
        if i in contours_to_remove:
            continue
        
        for j, cnt2 in enumerate(filt_contours):
            if i != j:
                # Si el índice de cnt2 ya está marcado para eliminar, se salta
                if j in contours_to_remove:
                    continue
                
                b = is_inside(cnt, cnt2)
                x1, y1, w1, h1 = cv2.boundingRect(cnt)
                x2, y2, w2, h2 = cv2.boundingRect(cnt2)

                # Verificar si son iguales o si uno está dentro del otro
                if (x1 == x2 and y1 == y2 and w1 == w2 and h1 == h2) or b:
                    contours_to_remove.add(i)  # Añadir índice de cnt a eliminar
                    break  # No es necesario seguir iterando sobre cnt2

    # Filtrar contornos que no están en contours_to_remove
    filt_contours = [cnt for i, cnt in enumerate(filt_contours) if i not in contours_to_remove]

    return filt_contours

def crop_plate_numbers(image):
    if type(image) == str:
        # Carregar imatge des del camí
        image = cv2.imread(image)

    # Resize mntenint aspect ratio
    ar1 = image.shape[1] / image.shape[0]
    new_width = 400
    new_height = int(new_width / ar1)
    image = cv2.resize(image, (new_width, new_height))

    # Convertir imatge a espai HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Obtenir el canal de valor (V) del HSV
    v_channel = hsv[:, :, 2]

    # Binaritzar la imatge utilitzant el mètode d'Otsu
    _, binary_image = cv2.threshold(v_channel, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Copiar la imatge original per a no modificar-la
    gray_image = image.copy()

    # Convertir la imatge a escala de grisos
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    n_intents = 6
    for i in range(n_intents):
        
        # 1. Si es el primer intent, fem trobar contorns
        if i == 0:
            flt_contours = find_contours_mat(binary_image)

        # 2. Si es el 2n intent, fem dilate + erode
        if i == 1:
            bin_cp = binary_image.copy()
            kernel = np.ones((2, 2), np.uint8)
            bin_cp = cv2.dilate(bin_cp, kernel, iterations=1)
            bin_cp = cv2.erode(bin_cp, kernel, iterations=1)

            # # save to tmp folder
            # cv2.imwrite('tmp/bin_ed1.jpg', bin_cp)

            flt_contours = find_contours_mat(bin_cp)

        # 3. Si es el 3r intent, fem canny edge
        if i == 2:
            bin_cp = binary_image.copy()

            # Aplicar un filtro de mediana para eliminar ruido
            gray = cv2.medianBlur(bin_cp, 3)

            # Aplicar el detector de bordes Canny
            edges = cv2.Canny(gray, 50, 150)

            # Encontrar contornos en la imagen binarizada
            flt_contours = find_contours_mat(edges)

        # 4. Si es el 6è intent, augmentem contrast per evitar detectar la nacionaalitat
        if i == 3:
            img_cp = image.copy()

            # augmentar contrast
            img_contr = cv2.convertScaleAbs(img_cp, alpha=3, beta=0)

            # Obtenim v de hsv i binaritzem
            hsv = cv2.cvtColor(img_contr, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            _, bin_cp = cv2.threshold(v_channel, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            flt_contours = find_contours_mat(bin_cp)

        # 5. Si es el 4t intent, ampliamos el rang de h
        if i == 4:
            bin_cp = binary_image.copy()

            # ampliem rang h
            flt_contours = find_contours_mat(bin_cp, image, w_min = 8, w_max = 95, h_min = 26, h_max = 119)

        # 6. Si es el 5è intent, ampliamos el rang de w
        if i == 5:
            bin_cp = binary_image.copy()

            # ampliem rang w
            flt_contours = find_contours_mat(bin_cp, image, w_min=5, w_max=80, h_min=7, h_max=70)


        if len(flt_contours) == 7:
            break

    digits = []

    # Post-processar els digits
    for contour in flt_contours:
        # Retallar el contorn (número o lletra)
        x, y, w, h = cv2.boundingRect(contour)
        digit = gray_image[y:y + h, x:x + w]
        # Normalizar los pixeles de los dígitos aumentando el contrast
        final_dig = cv2.normalize(digit, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        digits.append(final_dig)

    return digits


def matricula_to_digits(image, size=(34, 80)):
    """
    Funció final per obtenir i processar els digits de la matrícula.
    """
    # Obtenim els digits
    digits = crop_plate_numbers(image)
    digits_final = []

    # Redimensionem els digits
    for d in digits:
        # size = image.shape
        # # Añadir a la lista final
        # aspect_ratio = size[1] / size[0]
        # new_width = 15
        # new_height = int(new_width / aspect_ratio)
        # d = cv2.resize(d, (new_width, new_height))
        digits_final.append(d)
    return digits_final
