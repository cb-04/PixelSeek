import json
import numpy as np
import faiss
import torch
import open_clip
from PIL import Image

INDEX_PATH = "index.faiss"
PATHS_PATH = "paths.json"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

index = faiss.read_index(INDEX_PATH)
with open(PATHS_PATH) as f:
    paths = json.load(f)

def search_by_image(img_path, k=10):
    img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    scores, indices = index.search(feat.numpy().astype("float32"), k)
    return [(paths[i], float(scores[0][j])) for j, i in enumerate(indices[0])]

def search_by_text(query, k=10):
    tokens = tokenizer([query])
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    scores, indices = index.search(feat.numpy().astype("float32"), k)
    return [(paths[i], float(scores[0][j])) for j, i in enumerate(indices[0])]