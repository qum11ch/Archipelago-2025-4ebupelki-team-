import numpy as np
from typing import List, Union

from ultralytics import YOLO
import torch

# Загружаем модель YOLOv8
model = YOLO("model.pt")


def infer_image_bbox(image: np.ndarray) -> List[dict]:
    """Функция для получения ограничивающих рамок объектов на изображении.

    Args:
        image (np.ndarray): Изображение, на котором будет производиться инференс.

    Returns:
        List[dict]: Список словарей с координатами ограничивающих рамок и оценками.
        Пример выходных данных:
        [
            {
                'xc': 0.5,
                'yc': 0.5,
                'w': 0.2,
                'h': 0.3,
                'label': 0,
                'score': 0.95
            },
            ...
        ]
    """
    res_list = []

    result = model.predict(source=image, imgsz=1280, device=0)
    
    result_numpy = []

    # Преобразуем результаты в numpy массивы
    for res in result:
        result_numpy.append(res.cpu().numpy())
    
    # Если есть результаты, обрабатываем их
    if len(result_numpy) > 0:
        for res in result_numpy:
            for box in res.boxes:
                xc = box.xywhn[0][0] 
                yc = box.xywhn[0][1]
                w = box.xywhn[0][2]
                h = box.xywhn[0][3]
                conf = box.conf[0].item()

                formatted = {
                    'xc': xc,
                    'yc': yc,
                    'w': w,
                    'h': h,
                    'label': 0,
                    'score': conf
                }
                res_list.append(formatted)

    return res_list


def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    """Функция производит инференс модели на одном или нескольких изображениях.

    Args:
        images (Union[List[np.ndarray], np.ndarray]): Список изображений или одно изображение.

    Returns:
        List[List[dict]]: Список списков словарей с результатами предикта 
        на найденных изображениях.
        Пример выходных данных:
        [
            [
                {
                    'xc': 0.5,
                    'yc': 0.5,
                    'w': 0.2,
                    'h': 0.3,
                    'label': 0,
                    'score': 0.95
                },
                ...
            ],
            ...
        ]
    """    
    results = []
    if isinstance(images, np.ndarray):
        images = [images]

    # Обрабатываем каждое изображение из полученного списка
    for image in images:        
        image_results = infer_image_bbox(image)
        results.append(image_results)
    
    return results
