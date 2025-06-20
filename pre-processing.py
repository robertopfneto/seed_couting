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
    img_ann["y"] = img_ann["y"] * scale + top    

    # rotaciona imagem e salva
    top_rot, bot_rot = target_size[1] // 2, target_size[0] // 2
    M = cv.getRotationMatrix2D((top_rot, bot_rot), angle=5, scale=1.0)
    rotated = cv.warpAffine(padded, target_size, borderMode=cv.BORDER_CONSTANT, borderValue=0)

    coords = img_ann[["x","y"]].values #pego as coordenadas nas dimensoes (x,y) -> um array de shape (N,2)
    ones = np.ones((coords.shape[0], 1)) # adiciono mais uma dimensao (x, y, 1) pra conseguir rotacionar
    coords_homog = np.hstack([coords, ones])
    rotated_coords = coords_homog @ M.T #multiplico a matriz dos pontos pela matriz de rotacao
    
    rotated_ann = img_ann.copy()
    rotated_ann["x"] = rotated_coords[:, 0]
    rotated_ann["y"] = rotated_coords[:, 1]

    # Criar e salvar imagem com brilho aumentado
    contrast = 1.1 
    bright = 15
    brighted_img = cv.convertScaleAbs(padded, alpha=contrast, beta=bright) #aumenta contraste em 10% e aumenta 15 de brilho por pixel

     # salvando imagem e anotacoes
    def save_variation(image, annotations_dataframe, suffix):
        filename = f"{basename}_{suffix}.png"
        csv_name = f"{basename}_{suffix}.png"

        image_path = os.path.join(output_dir, filename)
        csv_path = os.path.join(output_dir, csv_name)

        #atualizar dataframe
        annotations_dataframe = annotations_dataframe.copy()
        annotations_dataframe["filename"] = filename
        annotations_dataframe["height"] = target_size[0]
        annotations_dataframe["width"] = target_size[1]

        # salvando
        cv.imwrite(image_path, image)
        annotations_dataframe.to_csv(csv_path, header=False, index=False)
    
    #salvar versao clahe
    save_variation(padded, img_ann, f"1_clahe")
    #salvar versao rotacionada
    save_variation(rotated, rotated_ann, "2_rotated")
    #salvar versao com aumento de constraste e brilho
    save_variation(brighted_img, img_ann, "3_bright")