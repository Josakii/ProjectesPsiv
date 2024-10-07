from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import torch


def crop_matricula(img_path, model):
    img = cv2.imread(img_path)

    res = model(img_path)

    predictions = res.pandas().xyxy[0]  # df con columnas: xmin, ymin, xmax, ymax, confidence, class, name

    for index, row in predictions.iterrows():
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])
        confidence = row['confidence']
        class_name = row['name']  # nombre de la clase detectada
        
        # Get the biggest rectangle
        if index == 0:
            biggest_rectangle = (xmin, ymin, xmax, ymax)
        else:
            if (xmax - xmin) * (ymax - ymin) > (biggest_rectangle[2] - biggest_rectangle[0]) * (biggest_rectangle[3] - biggest_rectangle[1]):
                biggest_rectangle = (xmin, ymin, xmax, ymax)

    # Crop the biggest rectangle
    xmin, ymin, xmax, ymax = biggest_rectangle
    cropped_img = img[ymin:ymax, xmin:xmax]

    return cropped_img