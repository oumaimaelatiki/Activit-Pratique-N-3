# step1_extraction.py
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

PDF_PATH = Path("data/attention is all you need.pdf")
OUT_DIR = Path("output")
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 300      # caractères (~60-70 tokens, compatible CLIP : 77 max)
CHUNK_OVERLAP = 50


# ---------- TEXTE ----------
def clean_text(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)   # mots coupés en fin de ligne
    text = re.sub(r"\s*\n\s*", " ", text)    # retours à la ligne -> espace
    return re.sub(r"\s{2,}", " ", text).strip()


def split_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Découpage par fenêtre glissante, en coupant sur un espace."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + size // 2:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def extract_text_chunks(doc):
    records = []
    for page_idx, page in enumerate(doc, start=1):
        text = clean_text(page.get_text("text"))
        for j, chunk in enumerate(split_text(text)):
            records.append({
                "type": "text",
                "page": page_idx,
                "chunk_id": f"p{page_idx}_c{j}",
                "content": chunk,
            })
    return records


# ---------- IMAGES ----------
def extract_embedded_images(doc, min_size=100):
    """Images bitmap intégrées au PDF."""
    records = []
    for page_idx, page in enumerate(doc, start=1):
        for k, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            info = doc.extract_image(xref)
            if info["width"] < min_size or info["height"] < min_size:
                continue
            path = IMG_DIR / f"page{page_idx}_img{k}.{info['ext']}"
            path.write_bytes(info["image"])
            records.append({"type": "image", "page": page_idx,
                            "image_path": str(path), "source": "embedded"})
    return records


def extract_vector_figures(doc, dpi=200, min_size=100):
    """Figures vectorielles : on rend en PNG chaque zone de dessins."""
    records = []
    for page_idx, page in enumerate(doc, start=1):
        for k, rect in enumerate(page.cluster_drawings()):
            if rect.width < min_size or rect.height < min_size:
                continue
            rect = rect + (-5, -5, 5, 5)  # petite marge
            pix = page.get_pixmap(clip=rect, dpi=dpi)
            path = IMG_DIR / f"page{page_idx}_fig{k}.png"
            pix.save(path)
            records.append({"type": "image", "page": page_idx,
                            "image_path": str(path), "source": "vector"})
    return records


# ---------- MAIN ----------
if __name__ == "__main__":
    doc = fitz.open(PDF_PATH)
    print(f"Document : {doc.page_count} pages")

    text_records = extract_text_chunks(doc)
    img_records = extract_embedded_images(doc)
    img_records += extract_vector_figures(doc)

    records = text_records + img_records
    (OUT_DIR / "chunks.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(text_records)} chunks de texte")
    print(f"{len(img_records)} images/figures sauvegardées dans {IMG_DIR}/")
    print("Exemple :", text_records[0])