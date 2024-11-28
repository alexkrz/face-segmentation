from pathlib import Path

import torch
from safetensors import safe_open

from src.pl_module import SegformerConfig, SegformerForSemanticSegmentation


def main(
    log_dir: str = "logs_pl/mit-b0/version_0",
    model_name: str = "best_model",
):
    # Load model from checkpoint
    log_dir = Path(log_dir)  # type: Path
    config = SegformerConfig.from_json_file(log_dir / "model_config.json")
    model = SegformerForSemanticSegmentation(config)

    state_dict = {}
    with safe_open(log_dir / f"{model_name}.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    # print(model.state_dict().keys())
    msg = model.load_state_dict(state_dict)
    print(msg)

    # Put model into eval mode
    model.eval()

    # Input to the model
    batch_size = 2
    input = torch.randn(batch_size, 3, 512, 512, requires_grad=True)
    outs = model.forward(pixel_values=input, labels=None)
    logits = outs.logits

    print("logits shape:", logits.shape)

    # Export the model
    print("Exporting model..")
    torch.onnx.export(
        model,  # model being run
        input,  # model input (or a tuple for multiple inputs)
        f"{str(log_dir)}/{model_name}.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["pixel_values"],  # the model's input names
        output_names=["logits"],  # the model's output names
        dynamic_axes={
            "pixel_values": {0: "batch_size"},  # variable length axes
            "logits": {0: "batch_size"},
        },
    )

    print(f"Model successfully exported to {str(log_dir)}/{model_name}.onnx")


if __name__ == "__main__":
    main()
