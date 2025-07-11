import os
import json
import pandas as pd
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# === Caminhos ===
dataset_path = os.path.join("C:", os.sep, "Users", "astma", "Documents", "seed_project", "seed_couting", "dataset", "train")
annotation_file = os.path.join(dataset_path, "_annotations.coco.json")

# === Parâmetros do mapa de densidade ===
GAUSSIAN_STD = 15  # Espalhamento da densidade

# === Função para gerar mapa de densidade ===
def generate_density_map(image_shape, points, sigma=GAUSSIAN_STD):
    h, w = image_shape
    density = np.zeros((h, w), dtype=np.float32)

    for x, y in points:
        x = min(w - 1, max(0, int(x)))
        y = min(h - 1, max(0, int(y)))
        density[y, x] += 1

    return gaussian_filter(density, sigma=sigma, mode='constant')

# === Carregar COCO ===
with open(annotation_file, 'r') as f:
    coco = json.load(f)

# === Indexar imagens por ID ===
image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

# === Processar imagens e anotações ===
csv_gerados = 0
npy_gerados = 0

points_per_image = {}

for ann in coco['annotations']:
    image_id = ann['image_id']
    x, y, w, h = ann['bbox']
    cx = x + w / 2
    cy = y + h / 2

    filename = image_id_to_filename[image_id]
    if filename not in points_per_image:
        points_per_image[filename] = []

    points_per_image[filename].append((cx, cy))

# === Gerar arquivos CSV e NPY ===
for filename, points in points_per_image.items():
    base_name = os.path.splitext(filename)[0]
    csv_path = os.path.join(dataset_path, f"{base_name}.csv")
    npy_path = os.path.join(dataset_path, f"{base_name}.npy")
    image_path = os.path.join(dataset_path, filename)

    # Salvar CSV de pontos
    df = pd.DataFrame(points, columns=["x", "y"])
    df.to_csv(csv_path, index=False, header=False)
    csv_gerados += 1

    # Carregar imagem correspondente para obter shape
    if not os.path.exists(image_path):
        print(f"[!] Imagem não encontrada: {filename}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Erro ao carregar imagem: {filename}")
        continue

    h, w = img.shape[:2]
    density_map = generate_density_map((h, w), points)

    # Salvar .npy com o mapa de densidade
    np.save(npy_path, density_map)
    npy_gerados += 1

# === LOG DE RESULTADO ===
print(f"[✓] {csv_gerados} arquivos CSV gerados com pontos.")
print(f"[✓] {npy_gerados} mapas de densidade .npy gerados (sigma={GAUSSIAN_STD})")


""" feat: adiciona conversão de anotações COCO para pontos e geração de mapas de densidade

- Converte bounding boxes do _annotations.coco.json em coordenadas de centro (x, y)
- Gera arquivos .csv com os pontos para cada imagem, no formato esperado por CSRNet
- Calcula e salva mapas de densidade (.npy) usando filtro gaussiano (sigma=15)
- Utiliza a resolução original das imagens para alinhar os mapas
- Salva os arquivos .csv e .npy na mesma pasta do dataset, evitando duplicação
- Imprime logs claros de sucesso e de eventuais erros de carregamento de imagem

 permitir o treinamento de modelos CSRNet com o mesmo dataset usado em detecção,
sem alterar a estrutura original dos arquivos de imagem. """