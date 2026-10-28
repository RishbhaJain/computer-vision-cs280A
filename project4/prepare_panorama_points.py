"""Optional, reproducible correspondence preparation for the supplied photos.

OpenCV is used ONLY for SIFT descriptors and descriptor matching. Robust fitting
is implemented below, and every homography is fitted with our own Ah=b solver.
The saved point file allows the notebook to run without OpenCV installed.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from panorama_extensions import load_photos, computeH_normalized, project_points


def robust_matches(p, q, seed=0, threshold=2.5):
    rng = np.random.default_rng(seed)
    best = np.zeros(len(p), dtype=bool)
    best_median = np.inf
    limit, iteration = 2500, 0
    while iteration < limit:
        iteration += 1
        ids = rng.choice(len(p), 4, replace=False)
        try:
            H = computeH_normalized(p[ids], q[ids])
            errors = np.linalg.norm(project_points(H, p) - q, axis=1)
        except (ValueError, np.linalg.LinAlgError):
            continue
        good = errors < threshold
        count = int(good.sum())
        median = np.median(errors[good]) if count else np.inf
        if (count, -median) > (int(best.sum()), -best_median):
            best, best_median = good, median
            chance = (count / len(p))**4
            if 0 < chance < 1:
                limit = min(limit, max(250, int(np.ceil(np.log(0.001)/np.log1p(-chance)))))
            elif chance == 1:
                limit = min(limit, 250)
    if best.sum() < 12:
        raise ValueError(f"Only {best.sum()} consistent point matches; inspect this pair.")
    for _ in range(5):
        H = computeH_normalized(p[best], q[best])
        errors = np.linalg.norm(project_points(H, p) - q, axis=1)
        updated = errors < threshold
        if updated.sum() < 12 or np.array_equal(updated, best):
            break
        best = updated
    H = computeH_normalized(p[best], q[best])
    return H, best


def spread_points(p, errors, width, height, maximum=64):
    """Select accurate matches across the overlap rather than one tiny patch."""
    cells = {}
    for i in np.argsort(errors):
        cell = (min(int(p[i, 0]/width*10), 9), min(int(p[i, 1]/height*10), 9))
        if cell not in cells:
            cells[cell] = int(i)
    ids = list(cells.values())[:maximum]
    for i in np.argsort(errors):
        if len(ids) >= maximum:
            break
        if int(i) not in ids and (not ids or np.min(np.linalg.norm(p[ids]-p[i], axis=1)) > 15):
            ids.append(int(i))
    return np.array(ids, dtype=int)


def draw_matches(a, b, p, q, filename, limit=32):
    a, b = Image.fromarray(a), Image.fromarray(b)
    canvas = Image.new("RGB", (a.width+b.width, max(a.height, b.height)), (248, 248, 248))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width, 0))
    draw = ImageDraw.Draw(canvas)
    rng = np.random.default_rng(42)
    for n, i in enumerate(np.linspace(0, len(p)-1, min(limit, len(p))).astype(int)):
        color = tuple(rng.integers(90, 255, 3).tolist())
        x, y = p[i]
        u, v = q[i]
        u += a.width
        draw.line((x, y, u, v), fill=color, width=2)
        for xx, yy in [(x, y), (u, v)]:
            draw.ellipse((xx-4, yy-4, xx+4, yy+4), fill=color, outline="black")
            draw.text((xx+5, yy+3), str(n+1), fill="white", stroke_width=1, stroke_fill="black")
    canvas.save(filename, quality=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opencv-path", default=None)
    parser.add_argument("--max-side", type=int, default=1200)
    args = parser.parse_args()
    if args.opencv_path:
        sys.path.insert(0, args.opencv_path)
    import cv2
    photos, sizes = load_photos(max_side=args.max_side)
    sift = cv2.SIFT_create(nfeatures=4500, contrastThreshold=0.025)
    features = {}
    for name, im in photos.items():
        gray = np.asarray(Image.fromarray(im).convert("L"))
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        features[name] = (keypoints, descriptors)
        print(name, "keypoints", len(keypoints), flush=True)
    names = list(photos)
    edges = list(zip(names[:-1], names[1:]))
    # Additional overlapping views reduce chained alignment drift.
    edges += [(names[4], names[8]), (names[5], names[8]), (names[6], names[8])]
    matcher = cv2.BFMatcher()
    out = Path("outputs/panoramas")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for edge_index, (a, b) in enumerate(edges):
        ka, da = features[a]
        kb, db = features[b]
        forward = matcher.knnMatch(da, db, k=2)
        backward = matcher.knnMatch(db, da, k=2)
        reverse = {m.queryIdx: m.trainIdx for m, n in backward if m.distance < 0.78*n.distance}
        matches = [m for m, n in forward if m.distance < 0.73*n.distance
                   and reverse.get(m.trainIdx) == m.queryIdx]
        # SIFT can assign several orientations to the same physical point.
        unique = {}
        for m in sorted(matches, key=lambda m: m.distance):
            loc = tuple(np.round(np.array(ka[m.queryIdx].pt)*2).astype(int))
            unique.setdefault(loc, m)
        matches = list(unique.values())
        p = np.array([ka[m.queryIdx].pt for m in matches])
        q = np.array([kb[m.trainIdx].pt for m in matches])
        if len(p) < 12:
            raise RuntimeError(f"Insufficient matches for {a} to {b}: {len(p)}")
        H, inliers = robust_matches(p, q, seed=edge_index)
        pgood, qgood = p[inliers], q[inliers]
        errors = np.linalg.norm(project_points(H, pgood)-qgood, axis=1)
        h, w = photos[a].shape[:2]
        chosen = spread_points(pgood, errors, w, h)
        pp, qq = pgood[chosen], qgood[chosen]
        fitted = computeH_normalized(pp, qq)
        err = np.linalg.norm(project_points(fitted, pp)-qq, axis=1)
        ah, aw = photos[a].shape[:2]
        bh, bw = photos[b].shape[:2]
        original_p = (pp+0.5)*np.asarray(sizes[a])/[aw, ah]-0.5
        original_q = (qq+0.5)*np.asarray(sizes[b])/[bw, bh]-0.5
        record = {"source": a, "target": b,
                  "source_points": original_p.tolist(), "target_points": original_q.tolist(),
                  "candidate_count": len(p), "ransac_inlier_count": int(inliers.sum()),
                  "saved_count": len(pp), "median_working_error_px": float(np.median(err)),
                  "max_working_error_px": float(err.max())}
        records.append(record)
        draw_matches(photos[a], photos[b], pp, qq, out/f"matches_{Path(a).stem}_{Path(b).stem}.jpg")
        print(a, "->", b, "candidates", len(p), "inliers", int(inliers.sum()),
              "saved", len(pp), "median", round(float(np.median(err)), 3), flush=True)
    payload = {"provenance": "Automatic SIFT descriptor matches, mutual ratio filtering, handwritten RANSAC, and normalized eight-unknown least-squares homographies. These are not manually clicked points.",
               "working_max_side": args.max_side, "original_sizes": sizes,
               "opencv_version_for_preparation": cv2.__version__, "pairs": records}
    Path("panorama_correspondences.json").write_text(json.dumps(payload, indent=2)+"\n")
    print("Wrote panorama_correspondences.json", flush=True)


if __name__ == "__main__":
    main()
