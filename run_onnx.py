import os

import cv2
import numpy as np


def main(
    img_dir: str = "data/025_08.jpg",
    log_dir: str = "logs_pl/mit-b0/version_0",
    ckpt_file: str = "best_model.onnx",
):
    img = cv2.imread(img_dir)
    print(img.shape)

    # Read model and set input_size
    print("Loading onnx model..")
    model = cv2.dnn.readNetFromONNX(os.path.join(log_dir, ckpt_file))
    input_size = (512, 512)
    output_layer_names = model.getUnconnectedOutLayersNames()[0]
    print("output names:", output_layer_names)

    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1 / (127.5),
        size=input_size,
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    )
    # Run forward pass
    model.setInput(blob)
    logits = model.forward(output_layer_names)

    print("logits.shape:", logits.shape)
    labels = np.argmax(logits, axis=1)[0]
    print("labels.shape:", labels.shape)


if __name__ == "__main__":
    main()
