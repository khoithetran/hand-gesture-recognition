# 🖼️ BLIP Image Captioning on COCO Dataset

This project fine-tunes the BLIP (Bootstrapped Language Image Pretraining) model for generating English captions on images from the COCO 2017 dataset.

## 📦 Dataset

We use the [COCO 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) from Kaggle, which contains:
- 118,000+ training images
- 5 captions per image
- JSON format annotation files

Images and annotations were used to fine-tune a BLIP-based captioning model using a sequence-to-sequence objective.

## 🤖 Model

The base model is [Salesforce/BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base), fine-tuned on the COCO dataset for English caption generation.

We export the final model in `.safetensors` format for deployment or inference.

## 🧪 Usage

You can load the model using Hugging Face Transformers:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

processor = BlipProcessor.from_pretrained("path/to/model")
model = BlipForConditionalGeneration.from_pretrained("path/to/model").to("cuda")

img = Image.open("example.jpg")
inputs = processor(img, return_tensors="pt").to("cuda")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("📝 Caption:", caption)
