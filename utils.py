import colorsys
import json

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np

LABEL_COLORS = {
    "background": [0.0, 0.0, 0.0],
    "skin": [0.843137, 0.0, 0.0],
    "l_brow": [0.007843, 0.533333, 0.0],
    "r_brow": [0.713725, 0.0, 1.0],
    "l_eye": [0.023529, 0.67451, 0.776471],
    "r_eye": [0.596078, 1.0, 0.0],
    "eye_g": [1.0, 0.647059, 0.188235],
    "l_ear": [1.0, 0.560784, 0.784314],
    "r_ear": [0.47451, 0.321569, 0.372549],
    "ear_r": [0.0, 0.996078, 0.811765],
    "nose": [0.690196, 0.647059, 1.0],
    "mouth": [0.580392, 0.678431, 0.517647],
    "u_lip": [0.603922, 0.411765, 0.0],
    "l_lip": [0.215686, 0.415686, 0.384314],
    "neck": [0.827451, 0.0, 0.54902],
    "neck_l": [0.996078, 0.960784, 0.564706],
    "cloth": [0.784314, 0.435294, 0.4],
    "hair": [0.619608, 0.890196, 1.0],
    "hat": [0.0, 0.788235, 0.27451],
}


def sample_colors():
    label2id_old = {
        "background": 0,
        "skin": 1,
        "nose": 2,
        "eye_g": 3,
        "l_eye": 4,
        "r_eye": 5,
        "l_brow": 6,
        "r_brow": 7,
        "l_ear": 8,
        "r_ear": 9,
        "mouth": 10,
        "u_lip": 11,
        "l_lip": 12,
        "hair": 13,
        "hat": 14,
        "ear_r": 15,
        "neck_l": 16,
        "neck": 17,
        "cloth": 18,
    }
    id2label = {
        "0": "background",
        "1": "skin",
        "2": "l_brow",
        "3": "r_brow",
        "4": "l_eye",
        "5": "r_eye",
        "6": "eye_g",
        "7": "l_ear",
        "8": "r_ear",
        "9": "ear_r",
        "10": "nose",
        "11": "mouth",
        "12": "u_lip",
        "13": "l_lip",
        "14": "neck",
        "15": "neck_l",
        "16": "cloth",
        "17": "hair",
        "18": "hat",
    }
    labels = {index: value for index, value in enumerate(list(id2label.values()))}

    # # Sample 18 different colors from HLS colorspace
    # hues = np.linspace(0, 1, len(labels), endpoint=False)  # Avoid repeating the same color
    # lightness = 0.5  # Lightness value (range: 0-1)
    # saturation = 0.7  # Saturation value (range: 0-1)

    # # Convert HLS to RGB
    # colors_rgb = [colorsys.hls_to_rgb(h, lightness, saturation) for h in hues]
    # color_dict = {labels[i]: colors_rgb[i] for i in range(len(labels))}

    # existing_cmap = plt.cm.viridis

    # # Pick 18 equally spaced colors
    # num_colors = len(labels)
    # colors = [existing_cmap(i)[:3] for i in np.linspace(0, 1, num_colors)]

    # color_dict = {labels[i]: colors[i] for i in range(len(labels))}

    colors_rgb = cc.glasbey_bw_minc_20_minl_30[: len(labels) - 1]  # glasbey_dark
    colors_rgb.insert(0, [0.0, 0.0, 0.0])  # Use black as background color
    color_dict = {labels[i]: colors_rgb[i] for i in range(len(labels))}

    with open("color_dict.json", "w") as f:
        json.dump(color_dict, f)


def label2rgb(mask: np.ndarray):
    assert isinstance(mask, np.ndarray), "mask needs to be numpy array"
    label_to_rgb = np.array(list(LABEL_COLORS.values()))

    # Replace labels with RGB values
    # Use the segmentation map as an index into the label_to_rgb array (advanced indexing)
    rgb_mask = label_to_rgb[mask]
    return rgb_mask


if __name__ == "__main__":
    sample_colors()
