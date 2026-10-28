"""Small numerical checks for the new panorama helpers."""
import numpy as np
from panorama_extensions import computeH_normalized, project_points, sample_bilinear, make_panorama


def run_checks():
    rng = np.random.default_rng(7)
    src = rng.uniform(0, 1000, (20, 2))
    expected_H = np.array([[1.05, -.04, -28], [.02, .96, 14], [.0001, -.00006, 1]])
    dst = project_points(expected_H, src)
    fitted = computeH_normalized(src, dst)
    assert np.allclose(project_points(fitted, src), dst, atol=1e-8)

    gray = np.arange(35, dtype=np.uint8).reshape(5, 7)
    rgb = np.stack([gray, gray+40, gray+80], axis=-1)
    for image in [gray, rgb]:
        result, mask, info = make_panorama({"ref": image}, {"ref": np.eye(3)}, reference="ref")
        assert np.array_equal(result, image)
        assert mask.all() and mask[0, 0]

    block = np.array([[0, 40], [80, 120]], np.uint8)
    values, weights = sample_bilinear(block, np.array([[.5]]), np.array([[.5]]))
    assert values[0, 0] == 60 and weights[0, 0] > 0

    translation = np.array([[1, 0, -3], [0, 1, 2], [0, 0, 1]])
    result, mask, info = make_panorama({"ref": gray, "moving": gray},
                                      {"ref": np.eye(3), "moving": translation}, reference="ref")
    assert result.shape == (7, 10) and info["origin"] == [-3, 0]
    assert result[0, 3] == gray[0, 0] and mask[0, 3]
    assert not mask[0, 0]

    result, mask, info = make_panorama({"ref": rgb}, {"ref": np.eye(3)},
                                      reference="ref", projection="cylindrical", focal_length=7)
    x, y = -info["origin"][0], -info["origin"][1]
    assert np.array_equal(result[y, x], rgb[2, 3]) and mask[y, x]

    # Cylindrical forward/backward coordinate equations agree.
    f, cx, cy = 900., 449.5, 599.5
    points = rng.uniform([0, 0], [899, 1199], (30, 2))
    dx = points[:, 0]-cx
    theta = np.arctan2(dx, f)
    vertical = f*(points[:, 1]-cy)/np.sqrt(dx**2+f**2)
    recovered = np.column_stack((cx+f*np.tan(theta), cy+vertical/np.cos(theta)))
    assert np.allclose(points, recovered, atol=1e-10)

    try:
        computeH_normalized([[0, 0], [1, 1], [2, 2], [3, 3]],
                            [[0, 0], [1, 1], [2, 2], [3, 3]])
    except ValueError:
        pass
    else:
        raise AssertionError("Collinear correspondences were accepted")
    print("PASS: homography recovery, grayscale/RGB identity, valid black pixels, "
          "bilinear average, translated canvas, and cylindrical round trip.")


if __name__ == "__main__":
    run_checks()
