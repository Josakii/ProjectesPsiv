import pygame
import sys
import os

# Inicializar Pygame
pygame.init()

# Configuración de la ventana
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Dibuja cuadrados sobre las imágenes')

# Ruta de la carpeta con las imágenes
folder_path = 'C:/Users/Josep/Downloads/tmp_cropped/C5/'  # Cambia esto a la ruta de tu carpeta
image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'png', 'jpeg'))]

# Bucle para procesar cada imagen
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = pygame.image.load(image_path)
    image = pygame.transform.scale(image, (width, height))  # Escalar la imagen al tamaño de la ventana

    # Variables para almacenar coordenadas
    squares = []
    bounding_boxes = []  # Para almacenar los cuadros delimitadores

    # Bucle principal para cada imagen
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Click izquierdo
                    pos = pygame.mouse.get_pos()
                    size = 5  # Tamaño del cuadrado
                    squares.append((pos[0] - size // 2, pos[1] - size // 2, size, size))
        
        # Dibujar la imagen y los cuadrados
        screen.blit(image, (0, 0))
        for square in squares:
            pygame.draw.rect(screen, (255, 0, 0), square)  # Dibuja cuadrados en rojo

        pygame.display.flip()

        # Guardar coordenadas cada 4 puntos
        if len(squares) >= 4 and len(squares) % 4 == 0:
            # Extraer las coordenadas de los 4 puntos
            x1, y1 = squares[-4][0], squares[-4][1]
            x2, y2 = squares[-3][0], squares[-3][1]
            x3, y3 = squares[-2][0], squares[-2][1]
            x4, y4 = squares[-1][0], squares[-1][1]

            # Calcular el cuadro delimitador
            xmin = min(x1, x2, x3, x4)
            xmax = max(x1 + 5, x2 + 5, x3 + 5, x4 + 5)  # Sumando el tamaño del cuadrado
            ymin = min(y1, y2, y3, y4)
            ymax = max(y1 + 5, y2 + 5, y3 + 5, y4 + 5)

            # Calcular las coordenadas normalizadas para YOLO
            center_x = (xmin + xmax) / 2 / width
            center_y = (ymin + ymax) / 2 / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            # Guardar la clase y las coordenadas en formato YOLO
            bounding_boxes.append(f"0 {center_x} {center_y} {box_width} {box_height}")

            # Limpiar los puntos después de guardar
            squares = []  # Reiniciar la lista de cuadrados

    # Guardar todas las coordenadas al final de la imagen
    coordinates_filename = os.path.splitext(image_file)[0] + '_coordenadas.txt'
    with open(os.path.join(folder_path, coordinates_filename), 'w') as f:
        for box in bounding_boxes:
            f.write(box + '\n')

# Salir
pygame.quit()
sys.exit()
