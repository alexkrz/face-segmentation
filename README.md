# Fine-Tune Semantic Segmentation Models for Face-Parsing

## Setup

Setup conda environment:

```bash
conda env create -n $YOUR_ENV_NAME -f environment.yml
conda activate $YOUR_ENV_NAME
pip install -r requirements.txt
pre-commit install
```

## Prepare training dataset

The original CelebAMask-HQ dataset can be found here: <https://github.com/switchablenorms/CelebAMask-HQ>. \
However, we use the dataset uploaded to Huggingface from here: <https://huggingface.co/datasets/v-xchen-v/celebamask_hq>.

To download and convert the dataset to single images, you should inspect and run

```bash
python prepare_dataset.py
```

## Train model

Details on the SegFormer model can be found here: <https://huggingface.co/docs/transformers/model_doc/segformer>

Load model from hf hub and store it locally:

```bash
python train_segformer_hf.py
```

Train model:

```bash
python train_segformer.py
```

## Evaluate model

Add sample image to `data/` directory.

Evaluate model with

```bash
python eval_segformer.py --img_dir $YOUR_IMG_FILE
```

## Optional: Convert model to onnx

Script to convert model: `model2onnx.py`

Script to test converted model: `run_onnx.py`
