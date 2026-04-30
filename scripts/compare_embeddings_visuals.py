"""Compare two embedding archives (CLIP vs DINO): UMAP/PCA scatter, centroid heatmaps, retrieval contact sheets.

Usage example:
  python scripts/compare_embeddings_visuals.py \
    --clip data/combined_learning_reduced/embeddings/clip_embeddings_all.npz \
    --dino data/combined_learning_reduced/embeddings/dino_embeddings_all.npz \
    --out-dir data/combined_learning_reduced/visuals
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os


def load_npz(path: Path):
    with np.load(path, allow_pickle=True) as archive:
        data = {k: archive[k] for k in archive.files}
    return data


def reduce_embed(emb, n=2000):
    if emb.shape[0] <= n:
        return emb, np.arange(emb.shape[0])
    rng = np.random.RandomState(0)
    idx = rng.choice(np.arange(emb.shape[0]), size=n, replace=False)
    return emb[idx], idx


def compute_2d(embeddings):
    try:
        import umap as _umap

        reducer = _umap.UMAP(n_components=2, random_state=0)
        return reducer.fit_transform(embeddings)
    except Exception:
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(n_components=2, init="pca", random_state=0, perplexity=30)
            return tsne.fit_transform(embeddings)
        except Exception:
            # fallback to PCA
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            return pca.fit_transform(embeddings)


def plot_scatter(ax, xy, stages, title):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=stages, cmap="tab10", s=6, alpha=0.8)
    ax.set_title(title)


def centroid_matrix(embeddings, stages):
    unique = np.unique(stages)
    cents = []
    for s in unique:
        vecs = embeddings[stages == s]
        cent = vecs.mean(axis=0)
        norm = np.linalg.norm(cent)
        if norm > 0:
            cent = cent / norm
        cents.append(cent)
    C = np.stack(cents, axis=0)
    return unique, C @ C.T


def plot_heatmap(ax, mat, labels, title):
    im = ax.imshow(mat, vmin=0, vmax=1.0, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_contact_sheet(out_path: Path, query_img: Image.Image, clip_imgs, dino_imgs):
    # layout: 1 + k columns, 3 rows: header, CLIP row, DINO row
    k = len(clip_imgs)
    thumb_w = 224
    thumb_h = 224
    padding = 6
    rows = 3
    cols = 1 + k
    W = cols * thumb_w + (cols + 1) * padding
    H = rows * thumb_h + (rows + 1) * padding
    canvas = Image.new("RGB", (W, H), (255, 255, 255))

    def paste(img, r, c):
        x = padding + c * (thumb_w + padding)
        y = padding + r * (thumb_h + padding)
        canvas.paste(img.resize((thumb_w, thumb_h)), (x, y))

    # header label: query in top-left, rest blank
    paste(query_img, 0, 0)
    # CLIP row
    for i, im in enumerate(clip_imgs):
        paste(im, 1, i + 1)
    # DINO row
    for i, im in enumerate(dino_imgs):
        paste(im, 2, i + 1)

    canvas.save(out_path)


def build_retrievals(emb, paths, query_idx, topk=5):
    # cosine similarity
    q = emb[query_idx:query_idx+1]
    sims = emb @ q.T
    sims = sims.ravel()
    sims[query_idx] = -np.inf
    idx = np.argsort(-sims)[:topk]
    return [paths[i] for i in idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--dino", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    clip_npz = Path(args.clip)
    dino_npz = Path(args.dino)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip = load_npz(clip_npz)
    dino = load_npz(dino_npz)

    assert clip["embeddings"].shape[0] == dino["embeddings"].shape[0], "Mismatched sample counts"

    stages = clip["stage_indices"].astype(int)
    paths = clip["image_paths"].astype(str)

    # reduce for 2D
    emb_clip_sub, idx_clip = reduce_embed(clip["embeddings"], n=2000)
    emb_dino_sub, idx_dino = reduce_embed(dino["embeddings"], n=2000)
    # compute 2D embeddings
    xy_clip = compute_2d(emb_clip_sub)
    xy_dino = compute_2d(emb_dino_sub)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_scatter(axes[0, 0], xy_clip, stages[idx_clip], "CLIP (2D)")
    plot_scatter(axes[0, 1], xy_dino, stages[idx_dino], "DINO (2D)")

    labels_clip, mat_clip = centroid_matrix(clip["embeddings"], stages)
    labels_dino, mat_dino = centroid_matrix(dino["embeddings"], stages)

    plot_heatmap(axes[1, 0], mat_clip, labels_clip, "CLIP centroid cosine")
    plot_heatmap(axes[1, 1], mat_dino, labels_dino, "DINO centroid cosine")

    fig.tight_layout()
    fig.savefig(out_dir / "embeddings_umap_centroids.png", dpi=200)
    plt.close(fig)

    # retrieval sheets
    rng = random.Random(0)
    chosen = rng.sample(list(range(len(paths))), args.num_seeds)
    for i, qidx in enumerate(chosen):
        qpath = paths[qidx]
        try:
            qimg = Image.open(qpath).convert("RGB")
        except Exception:
            qimg = Image.new("RGB", (256, 256), (200, 200, 200))

        clip_ret = build_retrievals(clip["embeddings"], paths, qidx, topk=args.topk)
        dino_ret = build_retrievals(dino["embeddings"], paths, qidx, topk=args.topk)

        clip_imgs = []
        dino_imgs = []
        for pth in clip_ret:
            try:
                clip_imgs.append(Image.open(pth).convert("RGB"))
            except Exception:
                clip_imgs.append(Image.new("RGB", (256, 256), (240, 240, 240)))
        for pth in dino_ret:
            try:
                dino_imgs.append(Image.open(pth).convert("RGB"))
            except Exception:
                dino_imgs.append(Image.new("RGB", (256, 256), (240, 240, 240)))

        outp = out_dir / f"retrieval_seed_{i:02d}.png"
        save_contact_sheet(outp, qimg, clip_imgs, dino_imgs)

    print("Saved visuals to", out_dir)


if __name__ == "__main__":
    main()
