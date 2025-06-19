import cv2
import numpy as np
import os
import pandas as pd


"""
Transformações realizadas:

- Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization), rotação e ajuste de brilho
- Redimensiona proporcionalmente com padding (garante a padronização das imagens em 512x512)

"""
def preprocess_image(image_path, annotation_path, output_dir, target_size=(512,512)):
    return