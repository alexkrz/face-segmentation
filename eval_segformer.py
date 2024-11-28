import torch
from PIL import Image
from torchvision import transforms

from src.datamodule import SegmentationDataset
from src.pl_module import SegformerModule


def main(
    img_dir: str = "data/025_08.jpg",
    ckpt_dir: str = "logs_pl/segformer/version_3/checkpoints/epoch=1-step=6750.ckpt",
):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)

    pil_img = Image.open(img_dir)
    print(pil_img)
    img_transforms = SegmentationDataset.segformer_img_tfms

    img_in = img_transforms(pil_img)
    img_in = img_in.unsqueeze(0)  # Add batch dimension
    print(img_in.shape)

    pl_module = SegformerModule.load_from_checkpoint(ckpt_dir)

    img_in = img_in.to(device)
    pl_module.to(device)
    outs = pl_module.forward(pixel_values=img_in, labels=None)
    logits = outs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

    # resize output to match input image dimensions
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=pil_img.size[::-1],
        mode="bilinear",
        align_corners=False,  # H x W
    )

    # get label masks
    labels = upsampled_logits.argmax(dim=1)[0]

    # move to CPU to visualize in matplotlib
    labels_viz = labels.cpu().numpy()
    print(labels_viz.shape)


if __name__ == "__main__":
    main()
