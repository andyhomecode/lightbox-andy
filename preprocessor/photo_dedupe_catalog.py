
# CNN-based duplicate/near-duplicate detection using cosine similarity

import os
import csv
import json
from collections import defaultdict
from typing import List
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import insightface
import cv2
import numpy as np

def list_images(directory: str, exts={'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp', '.heic', '.heif'}):
    files = []
    for root, _, filenames in os.walk(directory):
        for fname in filenames:
            if os.path.splitext(fname.lower())[1] in exts:
                files.append(os.path.join(root, fname))
    return files

def extract_embeddings(files: List[str], device='cpu'):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove classifier
    model.eval()
    model.to(device)
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    embs = {}
    with torch.no_grad():
        for f in files:
            try:
                img = Image.open(f).convert('RGB')
                x = preprocess(img).unsqueeze(0).to(device)
                feat = model(x).squeeze().cpu().numpy()
                embs[f] = feat / np.linalg.norm(feat)
            except Exception:
                continue
    return embs

def group_by_cnn(files: List[str], threshold: float = 0.92, device='cpu') -> List[List[str]]:
    embs = extract_embeddings(files, device=device)
    files_list = list(embs.keys())
    parent = {f: f for f in files_list}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(len(files_list)):
        for j in range(i+1, len(files_list)):
            f1, f2 = files_list[i], files_list[j]
            sim = float(np.dot(embs[f1], embs[f2]))
            if sim >= threshold:
                union(f1, f2)

    groups = defaultdict(list)
    for f in files_list:
        groups[find(f)].append(f)
    return [g for g in groups.values() if len(g) > 1]

def main(image_dir: str, out_json: str = 'duplicate_groups.json', out_csv: str = 'photo_catalog.csv', cnn_threshold: float = 0.92, device='cpu'):
    files = list_images(image_dir)
    print(f"Found {len(files)} images. Extracting CNN embeddings...")
    dup_groups = group_by_cnn(files, threshold=cnn_threshold, device=device)
    print(f"Found {len(dup_groups)} duplicate/near-duplicate groups.")

    # Face recognition: assign unique people labels across all images using insightface
    model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    model.prepare(ctx_id=0 if device == 'cuda' else -1)
    known_face_embs = []
    known_face_labels = []
    next_person_id = 1
    photo_people = {}
    for fpath in files:
        try:
            img = cv2.imread(fpath)
            faces = model.get(img)
            labels = []
            for face in faces:
                emb = face.embedding
                # Compare to known faces
                found = False
                for i, known_emb in enumerate(known_face_embs):
                    sim = np.dot(emb, known_emb) / (np.linalg.norm(emb) * np.linalg.norm(known_emb))
                    if sim > 0.5:  # threshold for same person, can be tuned
                        label = known_face_labels[i]
                        found = True
                        break
                if not found:
                    label = f"person{next_person_id}"
                    known_face_embs.append(emb)
                    known_face_labels.append(label)
                    next_person_id += 1
                labels.append(label)
            photo_people[fpath] = ','.join(labels) if labels else ''
        except Exception:
            photo_people[fpath] = ''

    # Write duplicate groups to JSON
    with open(out_json, 'w') as f:
        json.dump(dup_groups, f, indent=2)
    print(f"Wrote {out_json}")

    # Write catalog CSV with people labels
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'people'])
        for fpath in files:
            writer.writerow([fpath, photo_people.get(fpath, '')])
    print(f"Wrote {out_csv}")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Detect duplicate/near-duplicate images using CNN embeddings (ResNet50, cosine similarity)')
    ap.add_argument('--dir', type=str, required=True, help='Directory of images')
    ap.add_argument('--out-json', type=str, default='duplicate_groups.json', help='Output JSON path')
    ap.add_argument('--out-csv', type=str, default='photo_catalog.csv', help='Output CSV path')
    ap.add_argument('--cnn-threshold', type=float, default=0.92, help='Cosine similarity threshold for grouping (0.0-1.0, higher is stricter)')
    ap.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
    args = ap.parse_args()
    main(args.dir, args.out_json, args.out_csv, args.cnn_threshold, args.device)
