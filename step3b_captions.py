# step3b_captions.py
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import chromadb
import numpy as np

from step2_embeddings import embed_texts, embed_images

CAPTIONS = {
    "page3_img0": "Figure 1: The Transformer model architecture, with the encoder on the left "
                  "and the decoder on the right, using multi-head attention, feed forward, "
                  "add and norm layers, and positional encoding.",
    "page4_img0": "Multi-Head Attention: several scaled dot-product attention layers running "
                  "in parallel, with linear projections of V, K, Q, then concat and linear.",
    "page4_img1": "Scaled Dot-Product Attention: MatMul of Q and K, scale, mask, softmax, "
                  "then MatMul with V.",
    "page9_fig0": "Table 3: variations on the Transformer architecture, number of layers, "
                  "heads, dimensions, dropout, with perplexity and BLEU on the development set.",
}

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("attention_multimodal")

res = collection.get(where={"type": "image"})

ids, vecs = [], []
for id_, meta in zip(res["ids"], res["metadatas"]):
    stem = Path(meta["image_path"]).stem
    if stem not in CAPTIONS:
        continue
    img_vec = embed_images([meta["image_path"]])[0]
    txt_vec = embed_texts([CAPTIONS[stem]])[0]
    mix = img_vec + txt_vec                    # moyenne image + légende
    mix = mix / np.linalg.norm(mix)            # normalisation L2
    ids.append(id_)
    vecs.append(mix.tolist())
    print("mis à jour :", stem)

collection.update(ids=ids, embeddings=vecs)
print("Total dans la base :", collection.count())