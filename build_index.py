import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("MPS FALLBACK:", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))

import json, time, logging
import numpy as np
import faiss
import torch
import open_clip
from PIL import Image
from tqdm import tqdm


IMAGE_DIR = "images/"
INDEX_PATH = "index.faiss"
PATHS_PATH = "paths.json"
BATCH_SIZE = 8
REBUILD = False  # set True to force recompute

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

device = "cpu"

def load_model():
    logging.info("Loading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", 
        pretrained="openai",
        quick_gelu=True
    )
    model = model.to(device)
    model.eval()
    logging.info("Model loaded.")
    return model, preprocess

def embed_images(paths, model, preprocess):
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i:i + BATCH_SIZE]
        imgs = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
                valid_paths.append(p)
                img.close()
            except Exception as e:
                logging.warning(f"Failed to load {p}: {e}")
        if not imgs:
            continue
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        yield feats.cpu().numpy().astype("float32"), valid_paths


def gather_image_paths(image_dir):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(image_dir)
        for f in files if os.path.splitext(f)[1].lower() in exts
    ]
    logging.info(f"Found {len(paths)} images in '{image_dir}'")
    return paths


def main():
    # Cache check
    if not REBUILD and os.path.exists(INDEX_PATH) and os.path.exists(PATHS_PATH):
        logging.info("Index already exists. Set REBUILD=True to recompute.")
        return

    # Validate image directory
    if not os.path.exists(IMAGE_DIR):
        logging.error(f"IMAGE_DIR '{IMAGE_DIR}' does not exist.")
        raise FileNotFoundError(f"IMAGE_DIR '{IMAGE_DIR}' not found.")

    # Gather paths
    all_paths = gather_image_paths(IMAGE_DIR)
    if not all_paths:
        logging.error("No images found. Check IMAGE_DIR and supported extensions.")
        return

    # Load model
    model, preprocess = load_model()

    # Derive embedding dimension from model
    dim = model.visual.output_dim
    logging.info(f"Embedding dimension: {dim}")

    # Build index
    index = faiss.IndexFlatIP(dim)
    all_valid_paths = []
    start_time = time.time()
    total_batches = (len(all_paths) + BATCH_SIZE - 1) // BATCH_SIZE

    logging.info("Starting embedding and indexing...")
    for vecs, paths in tqdm(embed_images(all_paths, model, preprocess), total=total_batches, unit="batch"):
        index.add(vecs)
        all_valid_paths.extend(paths)

    # Save index and paths
    faiss.write_index(index, INDEX_PATH)
    with open(PATHS_PATH, "w") as f:
        json.dump(all_valid_paths, f)

    # Stats
    elapsed = time.time() - start_time
    logging.info(f"Indexed {index.ntotal} images")
    logging.info(f"Index saved to '{INDEX_PATH}'")
    logging.info(f"Paths saved to '{PATHS_PATH}'")
    logging.info(f"Total time: {elapsed:.2f}s ({index.ntotal / elapsed:.1f} images/sec)")

if __name__ == "__main__":
    main()