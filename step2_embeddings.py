# step2_embeddings.py
import json
from pathlib import Path

import open_clip
import torch
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Chargement du modèle ----------
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model = model.to(device).eval()


# ---------- Embeddings ----------
@torch.no_grad()
def embed_texts(texts, batch_size=32):
    """Retourne un tableau (n, 512) de vecteurs normalisés (norme L2 = 1)."""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch).to(device)   # tronque à 77 tokens
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)   # normalisation L2
        all_vecs.append(feats.cpu())
    return torch.cat(all_vecs).numpy()


@torch.no_grad()
def embed_images(paths, batch_size=16):
    """Retourne un tableau (n, 512) de vecteurs normalisés (norme L2 = 1)."""
    all_vecs = []
    for i in range(0, len(paths), batch_size):
        imgs = [preprocess(Image.open(p).convert("RGB"))
                for p in paths[i:i + batch_size]]
        batch = torch.stack(imgs).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)   # normalisation L2
        all_vecs.append(feats.cpu())
    return torch.cat(all_vecs).numpy()


# ---------- Test ----------
if __name__ == "__main__":
    records = json.loads(Path("output/chunks.json").read_text(encoding="utf-8"))
    texts = [r["content"] for r in records if r["type"] == "text"]
    images = [r["image_path"] for r in records if r["type"] == "image"]

    text_vecs = embed_texts(texts)
    image_vecs = embed_images(images)

    print("Texte  :", text_vecs.shape)
    print("Images :", image_vecs.shape)
    print("Norme d'un vecteur texte  :", (text_vecs[0] ** 2).sum() ** 0.5)
    print("Norme d'un vecteur image  :", (image_vecs[0] ** 2).sum() ** 0.5)