"""Generate the two A.3 rectifications, close-ups, and measured timings."""
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pillow_heif import register_heif_opener

from part_a import computeH, warpImageNearestNeighbor, warpImageBilinear


CASES = [
    {"name": "sliding_door", "title": "Sliding-door frame", "image": "IMG_6205.HEIC",
     "source_points": [[239, 654], [546, 658], [550, 853], [245, 869]],
     "output_shape": [480, 720], "detail_box": [20, 365, 250, 475]},
    {"name": "window_pane", "title": "Side window pane", "image": "IMG_6211.HEIC",
     "source_points": [[249, 574], [349, 587], [354, 719], [250, 712]],
     "output_shape": [480, 360], "detail_box": [55, 90, 195, 235]},
]


def captioned_panels(panels, labels, path, cell=(570, 530)):
    cw, ch = cell
    sheet = Image.new("RGB", (cw*len(panels), ch+40), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        image = Image.fromarray(panel).convert("RGB") if isinstance(panel, np.ndarray) else panel.convert("RGB")
        image.thumbnail((cw-20, ch-20), Image.Resampling.NEAREST)
        sheet.paste(image, (i*cw+(cw-image.width)//2, 45+(ch-20-image.height)//2))
        draw.text((i*cw+12, 10), label, font=font, fill="black")
    sheet.save(path)


def main():
    register_heif_opener()
    out = Path("outputs/rectification")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        photo = ImageOps.exif_transpose(Image.open(Path("images")/case["image"])).convert("RGB")
        original_size = photo.size
        photo.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        image = np.asarray(photo)
        points = np.asarray(case["source_points"], dtype=float)
        out_h, out_w = case["output_shape"]
        target = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype=float)
        H = computeH(points, target)
        nn = warpImageNearestNeighbor(image, H, (out_h, out_w))
        bil = warpImageBilinear(image, H, (out_h, out_w))
        Image.fromarray(nn).save(out/f'{case["name"]}_nearest.png')
        Image.fromarray(bil).save(out/f'{case["name"]}_bilinear.png')

        annotated = photo.copy()
        draw = ImageDraw.Draw(annotated)
        xy = [tuple(p) for p in points]
        draw.line(xy+[xy[0]], fill=(240, 55, 40), width=4)
        font = ImageFont.load_default(size=22)
        for i, (x, y) in enumerate(points):
            draw.ellipse((x-5, y-5, x+5, y+5), fill="yellow", outline="black", width=1)
            draw.text((x+7, y-27), str(i+1), fill="yellow", font=font,
                      stroke_width=2, stroke_fill="black")
        annotated.save(out/f'{case["name"]}_source.jpg', quality=92)
        xmin, ymin = np.floor(points.min(axis=0)-35).astype(int)
        xmax, ymax = np.ceil(points.max(axis=0)+35).astype(int)
        crop = annotated.crop((max(0,xmin), max(0,ymin), min(photo.width,xmax), min(photo.height,ymax)))
        captioned_panels([crop, nn, bil], ["Source crop and corners", "Nearest neighbor", "Bilinear"],
                         out/f'{case["name"]}_comparison.png')
        details = [Image.fromarray(a).crop(case['detail_box']) for a in (nn, bil)]
        details = [im.resize((im.width*4, im.height*4), Image.Resampling.NEAREST) for im in details]
        captioned_panels(details, ["Nearest neighbor: enlarged detail", "Bilinear: enlarged detail"],
                         out/f'{case["name"]}_detail.png', cell=(980, 610))

        # Alternate order to reduce systematic warm-cache advantages.
        timings = {"nearest": [], "bilinear": []}
        methods = [("nearest", warpImageNearestNeighbor), ("bilinear", warpImageBilinear)]
        for repeat in range(7):
            for name, function in methods if repeat%2 == 0 else methods[::-1]:
                start = time.perf_counter()
                function(image, H, (out_h, out_w))
                timings[name].append(time.perf_counter()-start)
        median = {name:float(np.median(times)*1000) for name,times in timings.items()}
        p = (H @ np.column_stack((points, np.ones(4))).T).T
        error = np.linalg.norm(p[:,:2]/p[:,2:3]-target, axis=1)
        record = {**case, "original_size": list(original_size), "working_size": list(photo.size),
                  "target_points": target.tolist(), "homography": H.tolist(),
                  "corner_errors_px": error.tolist(), "median_ms": median,
                  "timing_samples_seconds": timings, "bilinear_over_nearest": median['bilinear']/median['nearest']}
        results.append(record)
        print(case['name'], median, 'max corner error', error.max(), flush=True)
    payload = {"working_max_side": 1600, "repeats": 7,
               "timing_setup": "Median of seven timed runs per method after one warm-up, alternating order; excludes image loading and saving.",
               "machine": platform.machine(), "system": platform.system(),
               "selection": "Four frame corners selected by inspecting the supplied photographs at 1200 by 1600 resolution.",
               "aspect_ratio_note": "Target rectangle sizes are chosen for display; physical dimensions were not measured.",
               "cases": results}
    (out/'results.json').write_text(json.dumps(payload, indent=2)+'\n')


if __name__ == '__main__':
    main()
