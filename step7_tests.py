# step7_tests.py
import sys
sys.stdout.reconfigure(encoding="utf-8")

from step6_agent import ask

TESTS = [
    # 1. Question texte simple
    "What is the BLEU score of the big Transformer model on English-to-German translation?",

    # 2. Question sur une figure (doit utiliser content_type='image')
    "Show me the figure of the Transformer architecture.",

    # 3. Requête hybride texte + image
    "Show me the figure of multi-head attention and explain what it computes.",

    # 4. Requête par image (le chemin de l'image est la requête)
    "Find passages similar to the image output/images/page4_img1.png",

    # 5. Question hors document (l'agent doit-il appeler l'outil ?)
    "What is the capital of France?",
]

if __name__ == "__main__":
    for q in TESTS:
        ask(q)