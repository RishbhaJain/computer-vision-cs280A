"""Part A: multiple-image feather blending and direct cylindrical resampling.

No library homography-estimation, perspective-warp, or interpolation routines
are used here. Coordinates are (x, y); image indexing is [y, x].
"""

from pathlib import Path
import json
import time

import numpy as np
from PIL import Image, ImageOps


def load_photos(folder="images", max_side=1200, mode="RGB"):
    """Read photos at a practical working resolution; retain original sizes."""
    from pillow_heif import register_heif_opener
    register_heif_opener()
    photos, sizes = {}, {}
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in {".heic", ".heif", ".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(path) as raw:
            im = ImageOps.exif_transpose(raw).convert(mode)
            sizes[path.name] = im.size
            if max_side is not None:
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            photos[path.name] = np.array(im)
    if not photos:
        raise ValueError(f"No photos found in {folder}")
    return photos, sizes


def project_points(H, points):
    p = np.column_stack((np.asarray(points, dtype=float), np.ones(len(points))))
    q = (np.asarray(H, dtype=float) @ p.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[:, :2] / q[:, 2:3]


def computeH_normalized(im1_pts, im2_pts):
    """The same eight-unknown Ah=b solve, with coordinate normalization."""
    src, dst = np.asarray(im1_pts, float), np.asarray(im2_pts, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise ValueError("Point arrays must have equal shape (n, 2), with n >= 4.")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Point coordinates must be finite.")

    def normalize(p):
        center = p.mean(axis=0)
        centered = p - center
        if np.linalg.matrix_rank(centered) < 2:
            raise ValueError("Correspondences must not be collinear.")
        scale = np.sqrt(2) / np.sqrt(np.mean(np.sum(centered**2, axis=1)))
        T = np.array([[scale, 0, -scale * center[0]],
                      [0, scale, -scale * center[1]], [0, 0, 1]])
        return centered * scale, T

    src, T1 = normalize(src)
    dst, T2 = normalize(dst)
    A = np.zeros((2 * len(src), 8))
    for i, ((x, y), (u, v)) in enumerate(zip(src, dst)):
        A[2*i] = [x, y, 1, 0, 0, 0, -u*x, -u*y]
        A[2*i+1] = [0, 0, 0, x, y, 1, -v*x, -v*y]
    h, _, rank, _ = np.linalg.lstsq(A, dst.reshape(-1), rcond=None)
    if rank < 8:
        raise ValueError("These correspondences do not determine a homography.")
    Hn = np.append(h, 1).reshape(3, 3)
    H = np.linalg.solve(T2, Hn @ T1)
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("Cannot normalize this homography with H[2,2] = 1.")
    return H / H[2, 2]


def pair_points(record, photos, original_sizes):
    """Scale stored original-resolution coordinates to each working image."""
    points = []
    for name_key, point_key in [("source", "source_points"),
                                ("target", "target_points")]:
        name = record[name_key]
        h, w = photos[name].shape[:2]
        ow, oh = original_sizes[name]
        # Pillow resize uses pixel-center coordinates.
        p = np.asarray(record[point_key], float)
        points.append((p + 0.5) * [w / ow, h / oh] - 0.5)
    return points


def align_photos(photos, original_sizes, point_file="panorama_correspondences.json",
                 reference="IMG_6209.HEIC", compute_h=None):
    """Compose pairwise homographies into one chosen reference frame.

    Pass compute_h=computeH to use the function in your notebook. The default
    uses the same least-squares equations with normalized coordinates.
    """
    compute_h = compute_h or computeH_normalized
    data = json.loads(Path(point_file).read_text())
    graph = {name: [] for name in photos}
    diagnostics = []
    for pair in data["pairs"]:
        a, b = pair["source"], pair["target"]
        if a not in photos or b not in photos:
            continue
        for name in (a, b):
            if tuple(data["original_sizes"][name]) != tuple(original_sizes[name]):
                raise ValueError(f"Stored correspondences do not match {name}'s size.")
        p, q = pair_points(pair, photos, original_sizes)
        H = compute_h(p, q)
        # Given a known transform for b, transform for a is H_b @ H_a_to_b.
        graph[b].append((a, H))
        graph[a].append((b, np.linalg.inv(H)))
        errors = np.linalg.norm(project_points(H, p) - q, axis=1)
        diagnostics.append({"source": a, "target": b, "pairs": len(p),
                            "median_error_px": float(np.median(errors)),
                            "max_error_px": float(errors.max())})
    if reference not in graph:
        raise ValueError(f"Missing reference photo: {reference}")
    transforms, queue = {reference: np.eye(3)}, [reference]
    for known in queue:
        for name, H_to_known in graph[known]:
            if name not in transforms:
                H = transforms[known] @ H_to_known
                transforms[name] = H / H[2, 2]
                queue.append(name)
    missing = set(photos) - set(transforms)
    if missing:
        raise ValueError(f"No chain of correspondences connects these photos: {missing}")
    return transforms, diagnostics


def _image_boundary(im, samples=256):
    h, w = im.shape[:2]
    x, y = np.linspace(0, w-1, samples), np.linspace(0, h-1, samples)
    return np.vstack((np.column_stack((x, np.zeros_like(x))),
                      np.column_stack((x, np.full_like(x, h-1))),
                      np.column_stack((np.zeros_like(y), y)),
                      np.column_stack((np.full_like(y, w-1), y))))


def sample_bilinear(im, x, y):
    """Handwritten bilinear interpolation plus a separate validity mask."""
    h, w = im.shape[:2]
    valid = (np.isfinite(x) & np.isfinite(y) &
             (x >= 0) & (x <= w-1) & (y >= 0) & (y <= h-1))
    x, y = np.where(valid, x, 0), np.where(valid, y, 0)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    dx, dy = x-x0, y-y0
    # Linear feather weights favor points farther from source boundaries.
    weight = np.minimum.reduce((x+1, w-x, y+1, h-y)) * valid
    if im.ndim == 3:
        dx, dy = dx[..., None], dy[..., None]
    values = ((1-dx)*(1-dy)*im[y0, x0] + dx*(1-dy)*im[y0, x1]
              + (1-dx)*dy*im[y1, x0] + dx*dy*im[y1, x1])
    return values.astype(np.float32), weight.astype(np.float32)


def make_panorama(photos, transforms, names=None, reference="IMG_6209.HEIC",
                  projection="planar", focal_length=None, rows_per_chunk=96,
                  max_pixels=30_000_000):
    """Blend any number of originals into a planar or cylindrical canvas.

    Cylindrical pixels are inverse-mapped directly into each original photo;
    an already-stitched panorama is never resampled. Focal length is in pixels
    at the working resolution and is a visual choice, not a calibration.
    """
    if projection not in {"planar", "cylindrical"}:
        raise ValueError("projection must be 'planar' or 'cylindrical'.")
    names = list(names) if names is not None else list(photos)
    if reference not in photos or reference not in transforms:
        raise ValueError("The reference image needs a known transform.")
    start = time.perf_counter()
    ref_h, ref_w = photos[reference].shape[:2]
    cx, cy = (ref_w-1)/2, (ref_h-1)/2
    f = float(focal_length if focal_length is not None else ref_w)
    if f <= 0:
        raise ValueError("focal_length must be positive.")
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    K_inv = np.linalg.inv(K)
    # Rebase all transforms into the requested reference's coordinates.
    rebase = np.linalg.inv(transforms[reference])
    matrices = {name: rebase @ transforms[name] for name in names}
    bounds = []
    for name in names:
        H = matrices[name]
        border = _image_boundary(photos[name], samples=256 if projection == "cylindrical" else 2)
        p = np.column_stack((border, np.ones(len(border))))
        q = (H @ p.T).T
        if projection == "planar":
            if q[:, 2].min() <= 0 <= q[:, 2].max():
                raise ValueError("Planar projection crosses infinity; use another reference.")
            bounds.append(q[:, :2] / q[:, 2:3])
        else:
            # Positive homogeneous orientation at the image center.
            h, w = photos[name].shape[:2]
            if (H @ [(w-1)/2, (h-1)/2, 1])[2] < 0:
                q = -q
                matrices[name] = -H
            rays = (K_inv @ q.T).T
            angle = np.arctan2(rays[:, 0], rays[:, 2])
            vertical = f*rays[:, 1]/np.hypot(rays[:, 0], rays[:, 2])
            bounds.append(np.column_stack((f*angle, vertical)))
    bounds = np.vstack(bounds)
    if not np.isfinite(bounds).all():
        raise ValueError("Invalid projected image bounds.")
    if projection == "cylindrical" and np.ptp(bounds[:, 0]) > np.pi*f:
        raise ValueError("This view crosses the cylinder seam; choose another reference.")
    # Two-pixel margin covers subpixel extrema between sampled boundary points.
    margin = 2 if projection == "cylindrical" else 0
    xmin, ymin = np.floor(bounds.min(axis=0)).astype(int) - margin
    xmax, ymax = np.ceil(bounds.max(axis=0)).astype(int) + margin
    width, height = int(xmax-xmin+1), int(ymax-ymin+1)
    if height*width > max_pixels:
        raise ValueError("Canvas too large; lower max_side or inspect the homographies.")
    channels = photos[names[0]].shape[2:]
    if any(photos[name].shape[2:] != channels for name in names):
        raise ValueError("All photos must have the same channel count.")
    output = np.zeros((height, width) + channels, dtype=np.uint8)
    coverage = np.zeros((height, width), dtype=bool)
    inverses = {name: np.linalg.inv(matrices[name]) for name in names}
    for row in range(0, height, rows_per_chunk):
        stop = min(row+rows_per_chunk, height)
        y, x = np.mgrid[row:stop, 0:width]
        x, y = x+xmin, y+ymin
        if projection == "cylindrical":
            angle = x/f
            c = np.cos(angle)
            # K @ [sin(theta), v/f, cos(theta)], without a planar warp.
            ref = np.stack((f*np.sin(angle)+cx*c, y+cy*c, c))
        else:
            ref = np.stack((x, y, np.ones_like(x)))
        total = np.zeros((stop-row, width) + channels, np.float32)
        weights = np.zeros((stop-row, width), np.float32)
        for name in names:
            p = np.einsum("ij,jhw->ihw", inverses[name], ref)
            with np.errstate(divide="ignore", invalid="ignore"):
                sx, sy = p[0]/p[2], p[1]/p[2]
            values, weight = sample_bilinear(photos[name], sx, sy)
            total += values * (weight[..., None] if channels else weight)
            weights += weight
        denominator = weights[..., None] if channels else weights
        np.divide(total, denominator, out=total, where=denominator > 0)
        output[row:stop] = np.clip(np.rint(total), 0, 255).astype(np.uint8)
        coverage[row:stop] = weights > 0
    info = {"projection": projection, "reference": reference, "images": names,
            "width": width, "height": height, "origin": [int(xmin), int(ymin)],
            "focal_length": f if projection == "cylindrical" else None,
            "seconds": time.perf_counter()-start,
            "covered_pixels": int(coverage.sum())}
    return output, coverage, info


def save_panorama(image, mask, path):
    """Save both the RGB/grayscale result and a separate coverage mask."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    Image.fromarray(mask.astype(np.uint8)*255).save(path.with_name(path.stem+"_mask.png"))


EXAMPLE_GROUPS = {
    "mosaic_1": (["IMG_6205.HEIC", "IMG_6206.HEIC", "IMG_6207.HEIC"], "IMG_6206.HEIC"),
    "mosaic_2": (["IMG_6208.HEIC", "IMG_6209.HEIC", "IMG_6210.HEIC"], "IMG_6209.HEIC"),
    "mosaic_3": (["IMG_6211.HEIC", "IMG_6212.HEIC", "IMG_6213.HEIC"], "IMG_6212.HEIC"),
}
