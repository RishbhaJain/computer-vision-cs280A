"""Generate the three mosaics and the nine-image cylindrical extension."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from panorama_extensions import (
    load_photos, align_photos, make_panorama, save_panorama, EXAMPLE_GROUPS,
)


def comparison_tile(names, photos, result, title, destination):
    width = 1400
    source_height = 300
    sources = Image.new("RGB", (width, source_height+45), "white")
    d = ImageDraw.Draw(sources)
    d.text((15, 8), title, fill="black")
    for i, name in enumerate(names):
        tile = Image.fromarray(photos[name]).convert("RGB")
        cell_width = width//len(names)
        tile.thumbnail((cell_width-10, source_height-25))
        x = i*cell_width+(cell_width-tile.width)//2
        sources.paste(tile, (x, 45))
        d.text((i*cell_width+10, 29), name, fill="black")
    panorama = Image.fromarray(result).convert("RGB")
    panorama.thumbnail((width, 1050))
    canvas = Image.new("RGB", (width, sources.height+panorama.height+30), "white")
    canvas.paste(sources, (0, 0))
    canvas.paste(panorama, ((width-panorama.width)//2, sources.height+15))
    canvas.save(destination, quality=92)


def main():
    out = Path("outputs/panoramas")
    out.mkdir(parents=True, exist_ok=True)
    photos, sizes = load_photos(max_side=1200)
    transforms, diagnostics = align_photos(photos, sizes)
    outputs = {}
    for label, (names, reference) in EXAMPLE_GROUPS.items():
        result, mask, info = make_panorama(photos, transforms, names=names, reference=reference)
        save_panorama(result, mask, out/f"{label}.png")
        comparison_tile(names, photos, result, label.replace("_", " "), out/f"{label}_sources.jpg")
        outputs[label] = info
        print(label, info, flush=True)
    result, mask, info = make_panorama(photos, transforms)
    save_panorama(result, mask, out/"all_images_planar.png")
    outputs["all_images_planar"] = info
    print("all_images_planar", info, flush=True)
    for f in [600, 900, 1200]:
        result, mask, info = make_panorama(photos, transforms, projection="cylindrical", focal_length=f)
        label = f"all_images_cylindrical_f{f}"
        save_panorama(result, mask, out/f"{label}.png")
        outputs[label] = info
        print(label, info, flush=True)
    panels = []
    for label in ["all_images_planar", "all_images_cylindrical_f600",
                  "all_images_cylindrical_f900", "all_images_cylindrical_f1200"]:
        im = Image.open(out/f"{label}.png").convert("RGB")
        im.thumbnail((1200, 1000))
        panel = Image.new("RGB", (1200, im.height+45), "white")
        ImageDraw.Draw(panel).text((12, 12), label, fill="black")
        panel.paste(im, ((1200-im.width)//2, 40))
        panels.append(panel)
    sheet = Image.new("RGB", (2400, max(p.height for p in panels[:2])+max(p.height for p in panels[2:])), "white")
    offset = max(p.height for p in panels[:2])
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i%2)*1200, 0 if i<2 else offset))
    sheet.save(out/"projection_comparison.jpg", quality=92)
    payload = {"working_max_side": 1200,
               "point_provenance": "Automatic SIFT matches, handwritten RANSAC; not manually clicked.",
               "pair_diagnostics": diagnostics,
               "transforms_to_IMG_6209": {name: H.tolist() for name, H in transforms.items()},
               "outputs": outputs}
    (out/"results.json").write_text(json.dumps(payload, indent=2)+"\n")
    print("Saved outputs/panoramas/results.json", flush=True)


if __name__ == "__main__":
    main()
