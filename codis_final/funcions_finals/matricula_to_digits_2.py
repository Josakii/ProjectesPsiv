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

def find_contours_mat(binary_image, w_min = 5, w_max = 32, h_min = 10, h_max = 45):
    # Encontrar contornos en la imagen binarizada
    contours_letters, _ = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Limites Width y Height
    # w_min = 5
    # w_max = 32
    # h_min = 12
    # h_max = 45

    filt_contours = []
    # Dibujar un rectángulo sobre cada contorno
    for contour in contours_letters:
        # # Obtener las coordenadas del rectángulo que encierra el contorno
        x, y, w, h = cv2.boundingRect(contour)
        # print(x,y,w,h)
        # cv2.rectangle(image_cp2, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde, grosor 2
        # plt.imshow(cv2.cvtColor(image_cp2, cv2.COLOR_BGR2RGB))
        # plt.axis('off')  # Opcional: para ocultar los ejes
        # plt.show()
        if w_min < w < w_max and h_min < h < h_max:
            # Filtrar contornos que tocan los bordes (posición en los 0)
            # if x > 0 and y > 0 and (x + w) < binary_image.shape[1] and (y + h) < binary_image.shape[0]:
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

def delete_shadows(image_param):
    imagecpy = image_param.copy()
    # Convertir la imagen a escala de grises (opcional)
    gray_image = cv2.cvtColor(imagecpy, cv2.COLOR_BGR2GRAY)

    # Crear una máscara de los píxeles superiores a 230
    # La máscara tendrá valores 255 donde la condición es verdadera y 0 donde es falsa
    mask = gray_image < 70

    # Convertir la máscara a tipo uint8
    mask = mask.astype(np.uint8) * 255  # Multiplicamos por 255 para tener valores de 0 o 255

    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(imagecpy, imagecpy, mask=mask)

    return result

def crop_plate_numbers(image):
    if type(image) == str:
        # Carregar imatge des del camí
        image = cv2.imread(image)

    # Resize mntenint aspect ratio
    ar1 = image.shape[1] / image.shape[0]
    new_width = 150
    new_height = int(new_width / ar1)
    image = cv2.resize(image, (new_width, new_height))

    # # delete shadows
    # image = delete_shadows(image)

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

    n_intents = 5
    for i in range(n_intents):
        
        # 1. Si es el primer intent, fem trobar contorns
        if i == 0:
            print('Intent 1')
            flt_contours = find_contours_mat(binary_image)

        # 2. Si es el 2n intent, fem dilate + erode
        if i == 1:
            print('Intent 2')
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
            print('Intent 3')
            # # Convertir imagen a escala de grises
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Aplicar un filtro de mediana para eliminar ruido
            gray = cv2.medianBlur(bin_cp, 3)

            # Aplicar el detector de bordes Canny
            edges = cv2.Canny(gray, 50, 150)

            # Encontrar contornos en la imagen binarizada
            flt_contours = find_contours_mat(edges)

        # 4. Si es el 4t intent, ampliamos el rang de h
        if i == 3:
            bin_cp = binary_image.copy()
            print('Intent 4')

            # ampliem rang h
            flt_contours = find_contours_mat(bin_cp, w_min = 5, w_max = 30, h_min = 40, h_max = 80)
            print(len(flt_contours))

        # 5. Si es el 5è intent, ampliamos el rang de w
        if i == 4:
            bin_cp = binary_image.copy()
            print('Intent 5')

            # ampliem rang w
            flt_contours = find_contours_mat(bin_cp, w_min=5, w_max=80, h_min=15, h_max=45)



        if len(flt_contours) == 7:
            break

    digits = []
    # Dibujar un rectángulo sobre cada contorno
    for contour in flt_contours:
        # Retallar el contorn (número o lletra)
        x, y, w, h = cv2.boundingRect(contour)
        digit = gray_image[y:y + h, x:x + w]
        # Normalizar los pixeles de los dígitos aumentando el contrast
        final_dig = cv2.normalize(digit, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        digits.append(final_dig)

    return digits

def resize_with_padding(image, size=(15, 31)):
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
