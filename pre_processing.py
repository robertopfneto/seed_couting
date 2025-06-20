import cv2 as cv
import numpy as np
import os
import pandas as pd

"""
Transformações realizadas:

- Aplica CLAHE por canal mantendo cor (Contrast Limited Adaptive Histogram Equalization)
- Rotação e ajuste de brilho
- Redimensiona proporcionalmente com padding (padroniza as imagens em 512x512)
"""

def preprocess_image(image_path, annotation_path, output_dir, target_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    original_filename = os.path.basename(image_path)

    # Leitura da imagem colorida
    img = cv.imread(image_path)

    # Aplicar CLAHE por canal (mantém cor)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv.split(img)
    clahe_channels = [clahe.apply(c) for c in channels]
    out_img = cv.merge(clahe_channels)

    # Redimensionamento proporcional
    height, width = out_img.shape[:2]
    scale = min(target_size[0] / height, target_size[1] / width)
    new_wid = int(width * scale)
    new_heig = int(height * scale)
    resized_img = cv.resize(out_img, (new_wid, new_heig), interpolation=cv.INTER_AREA)

    # Padding para centralizar
    delta_heig = target_size[0] - new_heig
    delta_wid = target_size[1] - new_wid
    top = delta_heig // 2
    bottom = delta_heig - top
    left = delta_wid // 2
    right = delta_wid - left
    padded = cv.copyMakeBorder(resized_img, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)

    # Ajustar anotações
    annotations = pd.read_csv(annotation_path, header=None, names=["label", "x", "y", "filename", "width", "height"])
    img_ann = annotations[annotations["filename"] == original_filename].copy()
    img_ann["x"] = img_ann["x"] * scale + left
    img_ann["y"] = img_ann["y"] * scale + top

    # Rotacionar imagem
    center_x, center_y = target_size[1] // 2, target_size[0] // 2
    M = cv.getRotationMatrix2D((center_x, center_y), angle=5, scale=1.0)
    rotated = cv.warpAffine(padded, M, target_size, borderMode=cv.BORDER_CONSTANT, borderValue=0)

    # Ajustar pontos anotados para rotação
    coords = img_ann[["x", "y"]].values
    ones = np.ones((coords.shape[0], 1))
    coords_homog = np.hstack([coords, ones])
    rotated_coords = coords_homog @ M.T

    rotated_ann = img_ann.copy()
    rotated_ann["x"] = rotated_coords[:, 0]
    rotated_ann["y"] = rotated_coords[:, 1]

    # Criar imagem com brilho aumentado
    brighted_img = cv.convertScaleAbs(padded, alpha=1.1, beta=15)

    # Função para salvar imagem + anotações
    def save_variation(image, annotations_dataframe, suffix):
        filename = f"{basename}_{suffix}.png"
        csv_name = f"{basename}_{suffix}.csv"
        image_path = os.path.join(output_dir, filename)
        csv_path = os.path.join(output_dir, csv_name)

        annotations_dataframe = annotations_dataframe.copy()
        annotations_dataframe["filename"] = filename
        annotations_dataframe["height"] = target_size[0]
        annotations_dataframe["width"] = target_size[1]

        cv.imwrite(image_path, image)
        annotations_dataframe.to_csv(csv_path, header=False, index=False)

    # Salvar versões
    save_variation(padded, img_ann, "1_clahe")
    save_variation(rotated, rotated_ann, "2_rotated")
    save_variation(brighted_img, img_ann, "3_bright")
