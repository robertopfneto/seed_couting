import cv2 as cv
import numpy as np
import os
import pandas as pd


"""
Transformações realizadas:

- Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization), rotação e ajuste de brilho
- Redimensiona proporcionalmente com padding (garante a padronização das imagens em 512x512)

"""
def preprocess_image(image_path, annotation_path, output_dir, target_size=(512,512)):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    original_filename = os.path.basename(image_path)

    # convertendo as imagens para escala cinza
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # aplicando CLAHE (melhora de contraste)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    out_img = clahe.apply(gray)

    # redimensionando
    height, widht = out_img.shape #pegando dimensoes
    scale = min(target_size[0] / height, target_size[1] / widht) #definindo scala
    new_wid = int(widht * scale)
    new_heig = int(height * scale)
    resized_img = cv.resize(out_img, (new_wid,new_heig), interpolation=cv.INTER_AREA) #interpolation torna a redução de resolução mais suave

    # preenchendo as bordas com padding
    delta_heig = target_size[0] - new_heig
    delta_wid = target_size[1] - new_wid
    
    top = delta_heig // 2
    bottom = delta_heig - top

    left = delta_wid // 2
    right = delta_wid - left

    padded = cv.copyMakeBorder(resized_img, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)

    # ajustando anotacoes
    annotations = pd.read_csv(annotation_path, header=None, names=["label", "x", "y", "filename", "width", "height"])
    img_ann = annotations[annotations["filename"] == original_filename].copy()
    img_ann["x"] = img_ann["x"] * scale + left