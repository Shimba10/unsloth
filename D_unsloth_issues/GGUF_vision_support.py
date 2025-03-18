# gguf_vision_export.py

import json
import torch
from gguf import GGUFWriter
from transformers import AutoModel, AutoProcessor

class VisionGGUFExporter:
    def __init__(self, model, processor, model_type="llava"):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.arch_adapters = {
            "llava": self._handle_llava,
            "qwen_vl": self._handle_qwen_vl,
        }
        
    def _handle_llava(self):
        self.vision_model = self.model.vision_tower
        self.projector = self.model.mm_projector
        self.image_size = self.model.config.image_size
        self.patch_size = 14  # CLIP-specific
        
    def _handle_qwen_vl(self):
        self.vision_model = self.model.transformer.visual
        self.projector = self.model.transformer.visual.projector
        self.image_size = self.model.config.visual["image_size"]
        self.patch_size = 14  # Qwen-specific
        
    def _get_model_metadata(self):
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "vision_arch": self.vision_model.__class__.__name__,
            "mm_hidden_size": self.projector.out_features,
        }
    
    def _save_vision_components(self, gguf_writer):
        # Save vision encoder
        for name, param in self.vision_model.named_parameters():
            full_name = f"vision_encoder.{name.replace('.weight', '').replace('.bias', '')}"
            gguf_writer.add_tensor(full_name, param.data.cpu())
            
        # Save projection layer
        for name, param in self.projector.named_parameters():
            full_name = f"mm_projector.{name.replace('.weight', '').replace('.bias', '')}"
            gguf_writer.add_tensor(full_name, param.data.cpu())
    
    def export(self, save_path):
        # Initialize GGUF writer
        gguf_writer = GGUFWriter(save_path, "llama")
        
        # Handle architecture-specific components
        if self.model_type not in self.arch_adapters:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.arch_adapters[self.model_type]()
        
        # Add vision metadata
        metadata = self._get_model_metadata()
        gguf_writer.add_uint32("image_size", metadata["image_size"])
        gguf_writer.add_uint32("patch_size", metadata["patch_size"])
        gguf_writer.add_string("vision_arch", metadata["vision_arch"])
        gguf_writer.add_uint32("mm_hidden_size", metadata["mm_hidden_size"])
        
        # Save vision components
        self._save_vision_components(gguf_writer)
        
        # Save language model components (simplified)
        for name, param in self.model.lm_head.named_parameters():
            gguf_writer.add_tensor(f"lm_head.{name}", param.data.cpu())
        
        # Save tokenizer config
        gguf_writer.add_tokenizer_model(self.processor.tokenizer.__class__.__name__)
        gguf_writer.add_token_list(self.processor.tokenizer.get_vocab().keys())
        
        # Finalize
        gguf_writer.write_header()
        gguf_writer.write_kv_data()
        gguf_writer.write_tensors()
        gguf_writer.close()

# Usage Example
if __name__ == "__main__":
    # Load finetuned model
    model = AutoModel.from_pretrained("your-finetuned-vlm")
    processor = AutoProcessor.from_pretrained("your-finetuned-vlm")
    
    # Initialize exporter
    exporter = VisionGGUFExporter(
        model=model,
        processor=processor,
        model_type="llava"  # or "qwen_vl"
    )
    
    # Export to GGUF
    exporter.export("vision_model.gguf")
    
    # Test loading (requires llama.cpp with vision support)
    def test_gguf_loading():
        from llama_cpp import Llama
        llm = Llama(
            model_path="vision_model.gguf",
            n_gpu_layers=-1,
            n_ctx=2048,
            logits_all=True,
        )
        # Image processing and inference would go here
        
    # test_gguf_loading()