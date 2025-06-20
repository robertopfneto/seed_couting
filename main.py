from pre_processing import preprocess_image

#caminhos 
imagem = "./imagens/sementes_04.jpg"
csv = "./anotacoes/seed4.csv"
saida = "output_images"

preprocess_image(image_path=imagem, annotation_path=csv, output_dir=saida)