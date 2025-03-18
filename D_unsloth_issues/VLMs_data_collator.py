from transformers import DataCollatorWithPadding
from typing import Any, Dict, List, Union
import torch
from PIL import Image

class MultiModalCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer,
        image_processor,
        image_token="<image>",
        max_image_size=224,
        **kwargs
    ):
        super().__init__(tokenizer, padding=True, **kwargs)
        self.image_processor = image_processor
        self.image_token = image_token
        self.max_image_size = max_image_size
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    def _process_images(self, images: List[Image.Image]) -> torch.Tensor:
        return self.image_processor(
            images,
            return_tensors="pt",
            size={"height": self.max_image_size, "width": self.max_image_size},
        ).pixel_values

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Split features into text-only and multimodal
        text_only = []
        multimodal = []
        image_counts = []
        
        for f in features:
            if "images" in f:
                multimodal.append(f)
                image_counts.append(len(f["images"]))
            else:
                text_only.append(f)
                image_counts.append(0)

        # Process images in parallel
        all_images = []
        for f in multimodal:
            all_images.extend(f["images"])
        image_tensors = self._process_images(all_images) if all_images else None

        # Reconstruct features with image placeholders
        reconstructed = []
        image_ptr = 0
        for i, count in enumerate(image_counts):
            if count > 0:
                f = multimodal.pop(0)
                f["pixel_values"] = image_tensors[image_ptr:image_ptr+count]
                image_ptr += count
                reconstructed.append(f)
            else:
                f = text_only.pop(0)
                f["pixel_values"] = torch.zeros(0, 3, self.max_image_size, self.max_image_size)
                reconstructed.append(f)

        # Pad text inputs
        text_features = [{
            "input_ids": f["input_ids"],
            "attention_mask": f["attention_mask"],
        } for f in reconstructed]
        batch = super().__call__(text_features)

        # Pad image inputs efficiently
        pixel_values = [f["pixel_values"] for f in reconstructed]
        image_mask = [torch.ones(len(x), dtype=bool) if len(x) > 0 else torch.zeros(0, dtype=bool) 
                     for x in pixel_values]
        
        max_images = max(len(p) for p in pixel_values)
        if max_images > 0:
            padded_pixels = torch.stack([
                torch.cat([p, torch.zeros(
                    max_images - len(p), 
                    3, 
                    self.max_image_size, 
                    self.max_image_size
                )]) if len(p) > 0 else torch.zeros(
                    max_images, 
                    3, 
                    self.max_image_size, 
                    self.max_image_size
                ) for p in pixel_values
            ])
            image_mask = torch.stack([
                torch.cat([m, torch.zeros(max_images - len(m), dtype=bool)]) 
                for m in image_mask
            ])
        else:
            padded_pixels = torch.zeros(len(features), 0, 3, self.max_image_size, self.max_image_size)
            image_mask = torch.zeros(len(features), 0, dtype=bool)

        batch.update({
            "pixel_values": padded_pixels,
            "image_mask": image_mask,
            "image_token_positions": self._get_image_token_positions(batch["input_ids"]),
        })
        return batch

    def _get_image_token_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.zeros_like(input_ids, dtype=bool)
        for i, seq in enumerate(input_ids):
            positions[i] = seq == self.image_token_id
        return positions
    
    
#Usage

from transformers import AutoTokenizer, AutoImageProcessor

tokenizer = AutoTokenizer.from_pretrained("unsloth/qwen-vl")
image_processor = AutoImageProcessor.from_pretrained("unsloth/qwen-vl")

collator = MultiModalCollator(
    tokenizer=tokenizer,
    image_processor=image_processor,
    image_token="<img>",  # Model-specific token
    max_image_size=448,
)

# Sample dataset
dataset = [
    {"text": "A cat sitting <image>", "images": [Image.open("cat.jpg")]},
    {"text": "Pure text example"},
]

# Tokenization should be done first
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = dataset.map(preprocess)

# Training loop
for batch in DataLoader(tokenized_dataset, collate_fn=collator):
    model(**batch)