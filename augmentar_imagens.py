from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os
from natsort import natsorted

input_dir = r"C:/Users/gusta/Desktop/dataset"
output_dir = r"C:/Users/gusta/Desktop/dataset2"
os.makedirs(output_dir, exist_ok=True)

prompt = ""

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os
from natsort import natsorted

input_dir = r"C:/Users/gusta/Desktop/dataset"
output_dir = r"C:/Users/gusta/Desktop/dataset2"
os.makedirs(output_dir, exist_ok=True)

prompt = ""

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
)
pipe = pipe.to("cpu")

arquivos = [
    f for f in os.listdir(input_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

arquivos = natsorted(arquivos)

for nome_arquivo in arquivos:
    caminho_imagem = os.path.join(input_dir, nome_arquivo)

    image = Image.open(caminho_imagem).convert("RGB")
    image = image.resize((512, 512))

    output = pipe(prompt=prompt, image=image, strength=0.32, guidance_scale=3.0)
    fake_image = output.images[0]

    nome_saida = f"clone_fake_{os.path.splitext(nome_arquivo)[0]}.jpg"
    caminho_saida = os.path.join(output_dir, nome_saida)

    fake_image.save(caminho_saida)
    print(f"Imagem gerada: {nome_saida}")


)
pipe = pipe.to("cpu")

arquivos = [
    f for f in os.listdir(input_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

arquivos = natsorted(arquivos)

for nome_arquivo in arquivos:
    caminho_imagem = os.path.join(input_dir, nome_arquivo)

    image = Image.open(caminho_imagem).convert("RGB")
    image = image.resize((512, 512))

    output = pipe(prompt=prompt, image=image, strength=0.32, guidance_scale=3.0)
    fake_image = output.images[0]

    nome_saida = f"clone_fake_{os.path.splitext(nome_arquivo)[0]}.jpg"
    caminho_saida = os.path.join(output_dir, nome_saida)

    fake_image.save(caminho_saida)
    print(f"Imagem gerada: {nome_saida}")

