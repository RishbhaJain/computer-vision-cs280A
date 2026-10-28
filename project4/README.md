# Image Warping and Mosaicing

Rishbha Jain · CS280A · Project 4, Part A

[Report](https://rishbhajain.github.io/computer-vision-cs280A/project4/)

Use Python 3.11. From this folder:

```sh
python -m pip install -r requirements.txt
```

Open `a2.ipynb` in Jupyter or VS Code and run all cells. Keep `images/`, the Python files, and both correspondence JSON files beside it. The GitHub notebook has outputs cleared to keep it small; figures are on the report.

The notebook covers homography recovery, two rectifications, three mosaics, and cylindrical projection. It writes figures to `outputs/`. To rebuild the webpage after running it:

```sh
python build_panorama_examples.py
python build_report.py
```

The webpage is written to `report/index.html`. The code ZIP contains code and point data; the original photos are in the repository's `images/` folder.

`img1_img2.json` contains the nine manually selected pairs. Additional matches in `panorama_correspondences.json` were prepared with SIFT and custom RANSAC. Regenerating them requires `opencv-python-headless` and `python prepare_panorama_points.py`; the saved matches need no OpenCV installation.
