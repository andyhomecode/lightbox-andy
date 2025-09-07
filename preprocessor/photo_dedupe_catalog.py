import argparse
import csv
import json
import os
from pyexpat.errors import codes
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Image/EXIF
from PIL import Image, ExifTags
import numpy as np

# Robust imread for non-ASCII paths
import cv2

# Duplicate detection
from imagededup.methods import PHash, CNN

# Try HEIC if available
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# Optional CLIP tagging
try:
    import torch
    import open_clip
    _CLIP_AVAILABLE = True
except Exception:
    _CLIP_AVAILABLE = False


EXTS = {".jpg",".jpeg",".png",".webp",".bmp",".tiff",".tif",".gif",".heic",".heif"}


def list_images(root: str) -> List[str]:
    files = []
    for base, _, names in os.walk(root):
        for n in names:
            ext = os.path.splitext(n.lower())[1]
            if ext in EXTS:
                files.append(os.path.join(base, n))
    return files


def safe_imread(path: str):
    # Handles unicode paths; returns BGR numpy array or None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def score_image_bgr(img_bgr: np.ndarray,
                    w_sharp: float = 0.5,
                    w_res: float = 0.3,
                    w_exposure: float = 0.15,
                    w_contrast: float = 0.05) -> float:
    """Compute a simple quality score from a BGR image."""
    if img_bgr is None:
        return -1.0
    h, w = img_bgr.shape[:2]
    # Resolution (cap to 12MP so it doesn't dominate)
    res = min(w * h, 4000 * 3000) / float(4000 * 3000)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Sharpness via Laplacian variance
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharp_norm = min(sharp, 2000.0) / 2000.0

    mean = float(gray.mean())
    std = float(gray.std())
    mid_pref = 1.0 - (abs(mean - 128.0) / 128.0)  # peak at 128
    mid_pref = max(0.0, min(1.0, mid_pref))
    contrast = min(std, 64.0) / 64.0

    score = (w_sharp * sharp_norm +
             w_res * res +
             w_exposure * mid_pref +
             w_contrast * contrast)
    return float(score)


def exif_dict_from_pil(img: Image.Image) -> Dict:
    """Extract EXIF as a dict with human-readable keys when possible."""
    exif_raw = img.getexif()
    exif = {}
    if exif_raw:
        for k, v in exif_raw.items():
            tag = ExifTags.TAGS.get(k, k)
            exif[tag] = v
    return exif


def _ratio_to_float(r):
    # Convert PIL's rational to float safely
    try:
        num, den = r
        return float(num) / float(den)
    except Exception:
        try:
            return float(r)
        except Exception:
            return None


def gps_from_exif(exif: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Return (lat, lon) in decimal degrees if available."""
    gps_info = exif.get("GPSInfo")
    if not gps_info:
        return None, None

    # gps_info may have integer keys; map to names via GPSTAGS
    gps_tags = {}
    for key, val in gps_info.items():
        name = ExifTags.GPSTAGS.get(key, key)
        gps_tags[name] = val

    def convert_to_degrees(values):
        try:
            d = _ratio_to_float(values[0])
            m = _ratio_to_float(values[1])
            s = _ratio_to_float(values[2])
            if None in (d, m, s):
                return None
            return d + (m / 60.0) + (s / 3600.0)
        except Exception:
            return None

    lat = None
    lon = None
    if "GPSLatitude" in gps_tags and "GPSLatitudeRef" in gps_tags:
        lat = convert_to_degrees(gps_tags["GPSLatitude"])
        if lat is not None and gps_tags["GPSLatitudeRef"] in ["S", b"S"]:
            lat = -lat
    if "GPSLongitude" in gps_tags and "GPSLongitudeRef" in gps_tags:
        lon = convert_to_degrees(gps_tags["GPSLongitude"])
        if lon is not None and gps_tags["GPSLongitudeRef"] in ["W", b"W"]:
            lon = -lon

    return lat, lon


def extract_exif_fields(path: str) -> Dict:
    res = {"datetime": None, "camera_make": None, "camera_model": None,
           "lat": None, "lon": None}
    try:
        with Image.open(path) as im:
            exif = exif_dict_from_pil(im)
            res["datetime"] = exif.get("DateTimeOriginal") or exif.get("DateTime")
            res["camera_make"] = exif.get("Make")
            res["camera_model"] = exif.get("Model")
            lat, lon = gps_from_exif(exif)
            res["lat"], res["lon"] = lat, lon
    except Exception:
        pass
    return res


def build_dup_groups(image_dir: str,
                     use_cnn: bool = False,
                     phash_threshold: int = 15,
                     cnn_threshold: float = 0.92) -> List[List[str]]:
    """Find duplicate/near-duplicate groups as connected components."""
    # Get all image files in the directory
    files = list_images(image_dir)
    if use_cnn:
        enc = CNN()
        embs = enc.encode_images(image_dir)  # Pass directory path
        dups = enc.find_duplicates(encoding_map=embs, min_similarity_threshold=cnn_threshold)
    else:
        enc = PHash()
        codes = enc.encode_images(image_dir)  # Pass directory path
        dups = enc.find_duplicates(encoding_map=codes, max_distance_threshold=phash_threshold)
        
        # Group files with identical hashes
        from collections import defaultdict
        hash_to_files = defaultdict(list)
        for fname, phash in codes.items():
            hash_to_files[phash].append(fname)

        # Add to dups: if >1 file has same hash, group them
        for files_with_same_hash in hash_to_files.values():
            if len(files_with_same_hash) > 1:
                for f in files_with_same_hash:
                    dups.setdefault(f, [])
                    for g in files_with_same_hash:
                        if g != f and g not in dups[f]:
                            dups[f].append(g)
                
    idx = {f: i for i, f in enumerate(files)}
    parent = list(range(len(files)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for f, neigh in dups.items():
        for g in neigh:
            if f in idx and g in idx:
                unite(idx[f], idx[g])

    groups_map: Dict[int, List[str]] = {}
    for f, i in idx.items():
        r = find(i)
        groups_map.setdefault(r, []).append(f)

    groups = [sorted(v) for v in groups_map.values() if len(v) >= 2]
    return groups

    idx = {f: i for i, f in enumerate(files)}
    parent = list(range(len(files)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for f, neigh in dups.items():
        for g in neigh:
            if f in idx and g in idx:
                unite(idx[f], idx[g])

    groups_map: Dict[int, List[str]] = {}
    for f, i in idx.items():
        r = find(i)
        groups_map.setdefault(r, []).append(f)

    groups = [sorted(v) for v in groups_map.values() if len(v) >= 2]
    return groups


def choose_best_in_group(group: List[str],
                         weight_sharp: float,
                         weight_res: float,
                         weight_exposure: float,
                         weight_contrast: float) -> Tuple[str, List[Tuple[float, str]]]:
    scored = []
    for f in group:
        img = safe_imread(f)
        s = score_image_bgr(img,
                            w_sharp=weight_sharp,
                            w_res=weight_res,
                            w_exposure=weight_exposure,
                            w_contrast=weight_contrast)
        # fallback: size contributes a tiny bit if image unreadable
        if s < 0:
            try:
                s = 0.001 + (os.path.getsize(f) / (10 * 1024 * 1024))
            except Exception:
                s = 0.0
        scored.append((float(s), f))
    scored.sort(reverse=True)
    return scored[0][1], scored


def clip_tagger_setup(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str = "cpu"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device


def clip_tag_image(path: str,
                   labels: List[str],
                   clip_ctx) -> Optional[str]:
    """Return best-matching label using CLIP zero-shot; None if fails."""
    if not labels:
        return None
    try:
        model, preprocess, tokenizer, device = clip_ctx
        with Image.open(path) as im:
            im = im.convert("RGB")
            img_in = preprocess(im).unsqueeze(0).to(device)

        prompts = [f"a photo of {lab.strip()}" for lab in labels]
        text = tokenizer(prompts).to(device)

        with torch.no_grad():
            img_features = model.encode_image(img_in)
            txt_features = model.encode_text(text)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            txt_features /= txt_features.norm(dim=-1, keepdim=True)
            logits = img_features @ txt_features.T  # (1, N)
            idx = int(logits.argmax(dim=-1).item())
        return labels[idx].strip()
    except Exception:
        return None


@dataclass
class CatalogRow:
    file: str
    group_id: Optional[int]
    keep_flag: str  # "KEEP" or ""
    score: float
    datetime: Optional[str]
    camera_make: Optional[str]
    camera_model: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    tag: Optional[str]


def main():
    ap = argparse.ArgumentParser(description="Identify duplicate images, pick best, extract EXIF/GPS, optional CLIP tagging.")
    ap.add_argument("--dir", required=True, help="Root directory of images")
    ap.add_argument("--use-cnn", action="store_true", help="Use CNN near-dup detection (slower, more robust)")
    ap.add_argument("--phash-threshold", type=int, default=8, help="Max Hamming distance for PHash duplicates (lower=stricter)")
    ap.add_argument("--cnn-threshold", type=float, default=0.92, help="Min similarity for CNN duplicates (higher=stricter)")

    ap.add_argument("--w-sharp", type=float, default=0.5, help="Weight: sharpness")
    ap.add_argument("--w-res", type=float, default=0.3, help="Weight: resolution")
    ap.add_argument("--w-exposure", type=float, default=0.15, help="Weight: exposure sanity (mid-gray preference)")
    ap.add_argument("--w-contrast", type=float, default=0.05, help="Weight: contrast")

    ap.add_argument("--labels", type=str, default="", help="Comma-separated labels for CLIP tagging (e.g., 'Birthday party,Beach,Boating,Camping')")
    ap.add_argument("--clip-model", type=str, default="ViT-B-32", help="open-clip model name")
    ap.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k", help="open-clip pretrained set")
    ap.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda'")

    ap.add_argument("--out-csv", type=str, default="photo_catalog.csv", help="Output CSV path")
    ap.add_argument("--out-json", type=str, default="duplicate_groups.json", help="Output JSON path")
    ap.add_argument("--write-keep-delete", action="store_true", dest="write_keep_delete", help="Write keep_list.txt and delete_list.txt")
    ap.add_argument("--make-review-symlinks", action="store_true", help="Create review/keep_symlinks and review/dupe_symlinks")

    args = ap.parse_args()

    image_dir = args.dir
    files = list_images(image_dir)
    if not files:
        print("No images found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} images. Detecting duplicates...")

    groups = build_dup_groups(image_dir, use_cnn=args.use_cnn,
                              phash_threshold=args.phash_threshold,
                              cnn_threshold=args.cnn_threshold)
    print(f"Found {len(groups)} duplicate/near-duplicate groups.")

    # Map each file to group id (1-based)
    file_to_group: Dict[str, Optional[int]] = {f: None for f in files}
    for gi, grp in enumerate(groups, 1):
        for f in grp:
            file_to_group[f] = gi

    # Choose best in each group
    best_in_group: Dict[int, str] = {}
    for gi, grp in enumerate(groups, 1):
        best, _ = choose_best_in_group(grp,
                                       weight_sharp=args.w_sharp,
                                       weight_res=args.w_res,
                                       weight_exposure=args.w_exposure,
                                       weight_contrast=args.w_contrast)
        best_in_group[gi] = best

    # CLIP tagging setup
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    clip_ctx = None
    if labels and _CLIP_AVAILABLE:
        try:
            device = args.device
            if device == "cuda" and (not torch.cuda.is_available()):
                device = "cpu"
            clip_ctx = clip_tagger_setup(args.clip_model, args.clip_pretrained, device=device)
            print(f"CLIP tagging enabled with {len(labels)} labels on device={device}.")
        except Exception as e:
            print("CLIP setup failed; continuing without tags.", e)
            clip_ctx = None
    elif labels and not _CLIP_AVAILABLE:
        print("open-clip-torch/torch not installed; skipping tagging.")

    # Walk all files, compute EXIF and scores (if not in group, still score)
    rows: List[CatalogRow] = []
    keep_list: List[str] = []
    delete_list: List[str] = []

    for f in files:
        img = safe_imread(f)
        score = score_image_bgr(img,
                                w_sharp=args.w_sharp,
                                w_res=args.w_res,
                                w_exposure=args.w_exposure,
                                w_contrast=args.w_contrast)
        ex = extract_exif_fields(f)
        tag = None
        if clip_ctx and labels:
            tag = clip_tag_image(f, labels, clip_ctx)

        gid = file_to_group.get(f)
        keep_flag = ""
        if gid is None:
            keep_flag = ""  # not in a dup group; leave blank so you decide
        else:
            if best_in_group.get(gid) == f:
                keep_flag = "KEEP"
                keep_list.append(f)
            else:
                delete_list.append(f)

        rows.append(CatalogRow(
            file=f,
            group_id=gid,
            keep_flag=keep_flag,
            score=round(float(score), 6) if isinstance(score, (int, float)) else None,
            datetime=ex["datetime"],
            camera_make=ex["camera_make"],
            camera_model=ex["camera_model"],
            lat=ex["lat"],
            lon=ex["lon"],
            tag=tag
        ))

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "file","group_id","keep_flag","score",
            "datetime","camera_make","camera_model","lat","lon","tag"
        ])
        for r in rows:
            w.writerow([
                r.file, r.group_id, r.keep_flag, r.score,
                r.datetime, r.camera_make, r.camera_model, r.lat, r.lon, r.tag
            ])
    print(f"Wrote {args.out_csv}")

    # Write JSON
    out_groups = []
    for gi, grp in enumerate(groups, 1):
        out_groups.append({
            "group_id": gi,
            "keep": best_in_group.get(gi),
            "candidates": grp
        })
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(out_groups, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out_json}")

    # Optional keep/delete lists
    if args.write_keep_delete:
        with open("keep_list.txt", "w", encoding="utf-8") as fh:
            for p in keep_list:
                fh.write(p + "\n")
        with open("delete_list.txt", "w", encoding="utf-8") as fh:
            for p in delete_list:
                fh.write(p + "\n")
        print("Wrote keep_list.txt & delete_list.txt")

    # Optional symlinks for review (non-destructive)
    if args.make_review_symlinks:
        os.makedirs("review/keep_symlinks", exist_ok=True)
        os.makedirs("review/dupe_symlinks", exist_ok=True)

        def safe_symlink(src, dst):
            try:
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
            except AttributeError:
                # Windows no symlink perms? Fall back to copying
                try:
                    import shutil
                    shutil.copy2(src, dst)
                except Exception:
                    pass
            except Exception:
                pass

        for p in keep_list:
            name = os.path.basename(p)
            safe_symlink(p, os.path.join("review/keep_symlinks", name))
        for p in delete_list:
            name = os.path.basename(p)
            safe_symlink(p, os.path.join("review/dupe_symlinks", name))
        print("Created review symlink folders.")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        sys.exit(2)
