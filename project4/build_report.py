"""Build the static report after running a2.ipynb."""
import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from part_a import computeH, homography_system
from panorama_extensions import project_points


def build(destination):
    destination = Path(destination)
    assets = destination / 'assets'
    results = destination / 'results'
    assets.mkdir(parents=True, exist_ok=True)
    results.mkdir(exist_ok=True)

    files = {
        'sources': 'panoramas/source_photo_gallery.png',
        'correspondences': 'panoramas/manual_correspondences.png',
        'additional_matches': 'panoramas/additional_correspondences.png',
        'mosaic_1': 'panoramas/mosaic_1_sources.jpg',
        'mosaic_2': 'panoramas/mosaic_2_sources.jpg',
        'mosaic_3': 'panoramas/mosaic_3_sources.jpg',
        'projections': 'panoramas/planar_vs_cylindrical.png',
        'focal_lengths': 'panoramas/cylindrical_focal_comparison.png',
    }
    for label, source in files.items():
        with Image.open(Path('outputs')/source) as im:
            im = im.convert('RGB')
            im.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
            im.save(assets/f'{label}.jpg', quality=92, optimize=True)
    for name in ['sliding_door', 'window_pane']:
        for suffix in ['comparison', 'detail', 'nearest', 'bilinear']:
            filename = f'{name}_{suffix}.png'
            shutil.copy2(Path('outputs/rectification')/filename, assets/filename)
        shutil.copy2(Path('outputs/rectification')/f'{name}_source.jpg', assets/f'{name}_source.jpg')

    data = json.loads(Path('img1_img2.json').read_text())
    source, target = np.array(data['im1Points']), np.array(data['im2Points'])
    A, b = homography_system(source, target)
    H = computeH(source, target)
    errors = np.linalg.norm(project_points(H, source)-target, axis=1)
    np.savetxt(results/'A.csv', A, delimiter=',', fmt='%.10g')
    np.savetxt(results/'b.csv', b, delimiter=',', fmt='%.10g')
    (results/'manual_homography.json').write_text(json.dumps({
        'source_image': 'IMG_6205.HEIC', 'target_image': 'IMG_6206.HEIC',
        'coordinate_size': [4284, 5712], 'source_points': source.tolist(),
        'target_points': target.tolist(), 'A': A.tolist(), 'b': b.tolist(),
        'H': H.tolist(), 'fitting_errors_px': errors.tolist(),
    }, indent=2)+'\n')
    for name in ['panoramas', 'rectification']:
        shutil.copy2(Path('outputs')/name/'results.json', results/f'{name}.json')
    rectification = json.loads((results/'rectification.json').read_text())

    def figure(filename, caption, alt=None):
        path = assets/filename
        with Image.open(path) as im:
            width, height = im.size
        return (f'<figure><a href="assets/{filename}"><img src="assets/{filename}" '
                f'width="{width}" height="{height}" loading="lazy" '
                f'alt="{html.escape(alt or caption, quote=True)}"></a>'
                f'<figcaption>{caption}</figcaption></figure>')

    h_text = '\n'.join('[ ' + '  '.join(f'{v: .8f}' for v in row) + ' ]' for row in H)
    rect_html = ''
    timing_rows = ''
    for case in rectification['cases']:
        name = case['name']
        height, width = case['output_shape']
        rect_html += f'<h3>{case["title"]}</h3>'
        rect_html += figure(f'{name}_comparison.png',
            f'{case["image"]}: selected corners, nearest neighbor, and bilinear. Output: {width} × {height}.')
        rect_html += figure(f'{name}_detail.png', 'Enlarged details show the difference along edges.')
        rect_html += (f'<p class="small"><a href="assets/{name}_source.jpg">Full source</a> · '
                      f'<a href="assets/{name}_nearest.png">Nearest-neighbor result</a> · '
                      f'<a href="assets/{name}_bilinear.png">Bilinear result</a></p>')
        timings = case['median_ms']
        timing_rows += (f'<tr><td>{case["title"]}</td><td>{width} × {height}</td>'
                        f'<td>{timings["nearest"]:.2f} ms</td><td>{timings["bilinear"]:.2f} ms</td></tr>')

    mosaics = ''.join(figure(f'mosaic_{i}.jpg', f'Images {start}–{start+2} and their feather-blended mosaic.')
                      for i, start in enumerate([6205, 6208, 6211], 1))
    page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Warping and Mosaicing | Rishbha Jain</title>
<meta name="description" content="CS280A Project 4 Part A: homographies, image rectification, mosaics, and cylindrical projection.">
<style>
body{{max-width:1000px;margin:36px auto;padding:0 20px;font:17px/1.6 system-ui,sans-serif;color:#222;background:#fff}}
h1{{font-size:2rem;line-height:1.2}}h2{{margin-top:3rem;padding-top:1rem;border-top:1px solid #ddd}}h3{{margin-top:2rem}}
a{{color:#174f91}}nav{{display:flex;gap:8px 20px;flex-wrap:wrap;margin:24px 0}}
figure{{margin:24px 0}}img{{display:block;max-width:100%;height:auto;margin:auto}}figcaption,.small{{font-size:14px;color:#444}}
figcaption{{margin-top:8px}}pre{{overflow-x:auto;background:#f5f5f5;padding:16px;font-size:14px;line-height:1.55}}
table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{text-align:left;padding:10px 8px;border-bottom:1px solid #ddd}}
details{{margin:18px 0}}summary{{cursor:pointer}}code{{font-size:.9em}}footer{{margin:48px 0 24px;border-top:1px solid #ddd;padding-top:16px}}
@media(max-width:600px){{body{{padding:0 14px;font-size:16px}}h1{{font-size:1.7rem}}th,td{{padding:8px 4px;font-size:13px}}}}
</style>
</head>
<body>
<h1>Image Warping and Mosaicing</h1>
<p>Rishbha Jain · CS280A · Project 4, Part A</p>
<nav aria-label="Sections"><a href="#photos">A.1 Photos</a><a href="#homography">A.2 Homography</a><a href="#rectification">A.3 Rectification</a><a href="#mosaics">A.4 Mosaics</a><a href="#cylindrical">A.5 Cylindrical</a><a href="#code">Code</a></nav>

<h2 id="photos">A.1 — Photos</h2>
<p>These nine photos show overlapping views of the same courtyard. I used three groups for the mosaics: 6205–6207, 6208–6210, and 6211–6213. The working images are 900 × 1200 pixels.</p>
{figure('sources.jpg', 'The nine source photos, in order. Original HEIC files are included with the code.')}

<h2 id="homography">A.2 — Recovering the homography</h2>
<p>A homography maps a point (x, y) to (u, v). After multiplying by H, divide the first two coordinates by the third. Setting the last entry of H to 1 leaves eight unknowns.</p>
<pre>H = [ h₁ h₂ h₃ ]      h = [h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈]ᵀ
    [ h₄ h₅ h₆ ]
    [ h₇ h₈  1 ]

xh₁ + yh₂ + h₃ − uxh₇ − uyh₈ = u
xh₄ + yh₅ + h₆ − vxh₇ − vyh₈ = v

[ x y 1  0 0 0  −ux −uy ] h = [ u ]
[ 0 0 0  x y 1  −vx −vy ]     [ v ]</pre>
<p>Stacking the equations for the nine selected pairs gives A with shape 18 × 8 and b with 18 entries. <code>computeH</code> solves <code>np.linalg.lstsq(A, b, rcond=None)</code>.</p>
{figure('correspondences.jpg', 'The nine manually selected matches. The lower rows enlarge the two clusters; matching numbers identify the same point.')}
<p>The recovered H maps IMG_6205 to IMG_6206 in the original 4284 × 5712 coordinates:</p>
<pre>{h_text}</pre>
<p>The median reprojection error is {np.median(errors):.2f} pixels, with a maximum of {errors.max():.2f}. These are fitting errors. The points are clustered, so the mosaics below use more widely distributed matches.</p>
<p class="small">Numerical system: <a href="results/A.csv">A</a> · <a href="results/b.csv">b</a> · <a href="results/manual_homography.json">Points, H, and errors</a></p>

<h2 id="rectification">A.3 — Warping and rectification</h2>
<p>Both warps use inverse mapping. Nearest neighbor rounds the source position; bilinear interpolation weights its four neighbors. I mapped four corners of each frame to a rectangle. Target aspect ratios are chosen for display; physical dimensions were not measured.</p>
{rect_html}
<p>Nearest neighbor leaves more visible steps along edges. Bilinear is smoother, but takes longer and slightly softens detail.</p>
<table><thead><tr><th scope="col">Image</th><th scope="col">Output size</th><th scope="col">Nearest</th><th scope="col">Bilinear</th></tr></thead><tbody>{timing_rows}</tbody></table>
<p class="small">Median of seven runs after one warm-up, alternating method order, on an arm64 Mac. Both methods use NumPy; times exclude loading and saving. <a href="results/rectification.json">Corners, homographies, and timings</a>.</p>

<h2 id="mosaics">A.4 — Blending mosaics</h2>
<p>The additional matches come from SIFT with a custom RANSAC filter. Each of 11 pairs has 64 saved matches. The homography solve centers and scales coordinates first, then undoes that normalization. Median fitting errors range from 0.15 to 0.60 working-image pixels.</p>
<details><summary>Additional correspondences</summary>{figure('additional_matches.jpg', 'Sixteen of the 64 automatic matches for one pair from each group.')}</details>
<p>I transform the image boundaries to find the canvas, round minima down and maxima up, then translate the result into array coordinates. Each source is sampled with bilinear interpolation. Feather weights increase with distance from its edges:</p>
<pre>I(out) = Σ wᵢ Iᵢ / Σ wᵢ</pre>
<p>Coverage masks keep real black pixels valid. Uncovered parts of the canvas stay black. Feathering reduces seams, though small alignment errors remain around some foreground edges.</p>
{mosaics}

<h2 id="cylindrical">A.5 — Bells and whistles: cylindrical projection</h2>
<p>I combined all nine images on a cylinder, using IMG_6209 as the reference. For a cylinder pixel (u, v), the source sample is:</p>
<pre>θ = u/f
r = [sin θ, v/f, cos θ]ᵀ
pᵢ = Hᵢ⁻¹ K r       (divide by the third coordinate)</pre>
<p>K uses focal length f and the reference image center. This samples each original source directly, then applies the same feather blend. The choice f = 900 is visual, not a camera calibration.</p>
{figure('projections.jpg', 'All nine images: planar projection on the left, cylindrical projection with f = 900 on the right.')}
{figure('focal_lengths.jpg', 'Cylindrical results with f = 600, 900, and 1200 working-image pixels.')}
<p>The cylindrical result reduces stretching at the sides. Smaller f gives stronger curvature. Feathering cannot fully correct parallax or errors accumulated while composing homographies.</p>

<h2 id="code">Code and data</h2>
<p><a href="https://github.com/RishbhaJain/computer-vision-cs280A/tree/main/project4">GitHub folder</a> · <a href="a2.ipynb" download>Notebook</a> · <a href="code.zip">Code ZIP</a> · <a href="README.md">Run instructions</a></p>
<p class="small">Implementations: <a href="part_a.py">Homography and inverse warps</a> · <a href="panorama_extensions.py">Mosaics and cylindrical projection</a>. <a href="results/panoramas.json">Mosaic transforms and fitting errors</a>. OpenCV is used only to prepare SIFT matches; homography fitting, RANSAC, warping, and interpolation are implemented in NumPy.</p>
<footer class="small"><a href="../">Project 3</a> · Photos by Rishbha Jain</footer>
</body>
</html>
'''
    (destination/'index.html').write_text(page)
    print(f'Wrote {destination}/index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='report')
    build(parser.parse_args().output)
