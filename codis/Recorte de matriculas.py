import cv2
import numpy as np
import os

# Ruta de la carpeta con las imágenes
input_folder = 'C:/Users/Josep/Desktop/Matriculas Detection/prueba'

# Crear una carpeta para guardar las imágenes recortadas
output_folder = 'C:/Users/Josep/Desktop/Matriculas Detection/output_recortes'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Tamaño deseado para las imágenes recortadas
desired_size = (300, 100)  # (ancho, alto)

# Obtener una lista de archivos en la carpeta de entrada
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.PNG', '.jpg', '.jpeg'))]

# Procesar cada imagen en la carpeta
plate_count = 0
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir rangos para el color rojo en HSV (bordes rojos)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Crear máscaras para el color rojo
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # Aplicar la máscara a la imagen
    result = cv2.bitwise_and(image, image, mask=red_mask)

    # Encontrar contornos en las áreas rojas
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Procesar los contornos detectados
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filtrar por tamaño mínimo de los rectángulos
        if w > 30 and h > 10:
            # Recortar la región de la imagen original (área dentro del borde rojo)
            plate_image = image[y:y+h, x:x+w]
            
            # Convertir la región recortada a espacio de color HSV
            hsv_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)

            # Extraer el canal V (Intensidad)
            v_channel = hsv_plate[:, :, 2]

            # Aplicar la binarización de Otsu para quedarnos con los objetos destacados
            _, otsu_mask = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Invertir la máscara de Otsu para quedarnos con las letras (partes no blancas)
            inverted_mask = cv2.bitwise_not(otsu_mask)

            # Aplicar la máscara invertida a la imagen recortada original
            filtered_plate = cv2.bitwise_and(plate_image, plate_image, mask=inverted_mask)

            # Redimensionar la imagen filtrada
            resized_plate = cv2.resize(filtered_plate, desired_size)

            # Encontrar contornos de las letras dentro de la matrícula binarizada
            contours_letters, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Dibujar los contornos de las letras en la imagen original
            for letter_contour in contours_letters:
                x_letter, y_letter, w_letter, h_letter = cv2.boundingRect(letter_contour)
                
                # Filtrar contornos pequeños para evitar ruido
                if w_letter > 5 and h_letter > 10:  # Filtrar letras por tamaño
                    cv2.rectangle(filtered_plate, (x_letter, y_letter), (x_letter + w_letter, y_letter + h_letter), (0, 255, 0), 2)

            # Guardar la imagen recortada con los contornos de las letras
            plate_filename = f'{output_folder}/license_plate_{plate_count}.jpg'
            cv2.imwrite(plate_filename, resized_plate)
            plate_count += 1

            # Dibujar rectángulo en la imagen original para referencia
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar la imagen procesada con los rectángulos (opcional)
    cv2.imshow(f'Detected License Plates - {image_file}', image)
    cv2.waitKey(500)  # Mostrar cada imagen durante 500ms (medio segundo)
    cv2.destroyAllWindows()

# Mensaje final
print(f"Se han guardado {plate_count} imágenes recortadas de matrículas en '{output_folder}'.")
