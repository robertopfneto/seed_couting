import cv2 as cv
import numpy as np
import os
import json
from glob import glob
import random

from caminho import caminho 

# === CONFIGURAÇÃO DE CAMINHOS ===
base_path = caminho
train_path = os.path.join(base_path, "train")
annotation_file = os.path.join(train_path, "_annotations.coco.json")

# === FUNÇÕES AUXILIARES ===

# Carrega o arquivo de anotações COCO existente ou cria um novo se não existir
def load_or_init_coco(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"images": [], "annotations": [], "categories": [{"id": 1, "name": "semente"}]}

# Retorna os próximos IDs disponíveis para imagem e anotação (evita duplicidade)
def get_next_ids(coco):
    img_id = max((img["id"] for img in coco["images"]), default=0) + 1
    ann_id = max((ann["id"] for ann in coco["annotations"]), default=0) + 1
    return img_id, ann_id

# Redimensiona a imagem proporcionalmente e adiciona bordas (padding)
def resize_with_padding(img, target_size=(512, 512)):
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    resized = cv.resize(img, (int(w * scale), int(h * scale)))
    top = (target_size[0] - resized.shape[0]) // 2
    bottom = target_size[0] - resized.shape[0] - top
    left = (target_size[1] - resized.shape[1]) // 2
    right = target_size[1] - resized.shape[1] - left
    return cv.copyMakeBorder(resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=[255, 255, 255])

# Pré-processamento base com CLAHE + brilho
def preprocess_clahe(img):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_channels = [clahe.apply(c) for c in cv.split(img)]
    img_clahe = cv.merge(clahe_channels)
    img_bright = cv.convertScaleAbs(img_clahe, alpha=1.2, beta=20)
    return resize_with_padding(img_bright)

# Roda imagem em ângulos aleatórios
def random_rotate(img):
    angle = random.choice([-30, -15, 0, 15, 30])
    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

# Aplica brilho aleatório
def random_brightness(img, delta=40):
    value = random.randint(-delta, delta)
    return cv.convertScaleAbs(img, alpha=1, beta=value)

# Aplica contraste aleatório
def random_contrast(img, range_=(0.8, 1.2)):
    factor = random.uniform(*range_)
    return cv.convertScaleAbs(img, alpha=factor, beta=0)

# Correção gamma aleatória
def random_gamma(img, gamma_range=(0.5, 1.8)):
    gamma = random.uniform(*gamma_range)
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv.LUT(img, table)

# Flip horizontal
def horizontal_flip(img):
    return cv.flip(img, 1)

# Ruído Gaussiano
def add_gaussian_noise(img, mean=0, sigma=10):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.int16)
    noisy = img.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Gera anotações (bounding boxes) com base nos contornos
def generate_annotations(img, image_id, annotation_id):
    annots = []
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        area = cv.contourArea(cnt)
        if area > 10:
            annots.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": float(area),
                "iscrowd": 0
            })
            annotation_id += 1
    return annots, annotation_id

# === PROCESSAMENTO PRINCIPAL ===
coco_data = load_or_init_coco(annotation_file)
image_id, annotation_id = get_next_ids(coco_data)
existing_files = set(os.path.basename(img["file_name"]) for img in coco_data["images"])

img_paths = sorted(glob(os.path.join(train_path, "*.jpg")))
num_geradas = 0

# Transformações disponíveis
augmentations = [
    ("clahe", preprocess_clahe),
    ("rotated", lambda img: resize_with_padding(random_rotate(img))),
    ("bright", lambda img: resize_with_padding(random_brightness(img))),
    ("contrast", lambda img: resize_with_padding(random_contrast(img))),
    ("gamma", lambda img: resize_with_padding(random_gamma(img))),
    ("flip", lambda img: resize_with_padding(horizontal_flip(img))),
    ("noise", lambda img: resize_with_padding(add_gaussian_noise(img)))
]

for img_path in img_paths:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    if base_name.endswith("_bright"):
        continue  # pula imagens já processadas

    img = cv.imread(img_path)
    if img is None:
        continue

    for tag, func in augmentations:
        aug_filename = f"{base_name}_{tag}.jpg"
        aug_path = os.path.join(train_path, aug_filename)

        if aug_filename in existing_files:
            continue  # pula se essa variação já existe no COCO

        # Aplica transformação
        aug_img = func(img)

        # Salva imagem processada
        cv.imwrite(aug_path, aug_img)

        # Atualiza estrutura COCO
        coco_data["images"].append({
            "id": image_id,
            "file_name": aug_filename,
            "width": 512,
            "height": 512
        })

        # Gera anotações
        new_annots, annotation_id = generate_annotations(aug_img, image_id, annotation_id)
        coco_data["annotations"].extend(new_annots)

        image_id += 1
        num_geradas += 1

# === SALVAMENTO DO JSON FINAL ===
with open(annotation_file, "w") as f:
    json.dump(coco_data, f, indent=4)

# === LOG DE RESULTADO ===
print(f"[✓] {num_geradas} novas imagens processadas com data augmentation.")
print(f"[✓] Anotações salvas em: {annotation_file}")
