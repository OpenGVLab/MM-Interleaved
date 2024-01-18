import os
import torch

from transformers import CLIPModel, CLIPProcessor
from transformers import LlamaTokenizer, LlamaForCausalLM
from diffusers import StableDiffusionPipeline

version = 'lmsys/vicuna-13b-v1.3'
path = os.path.join('./assets', version)
os.makedirs(path, exist_ok=True)
llm_tokenizer:LlamaTokenizer = LlamaTokenizer.from_pretrained(version)
llm_tokenizer.save_pretrained(path)
llm_model = LlamaForCausalLM.from_pretrained(version, force_download=True, resume_download=False)
llm_model.save_pretrained(path)

version = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(version)
clip_processor = CLIPProcessor.from_pretrained(version)
path = os.path.join('./assets', version)
os.makedirs(path, exist_ok=True)
clip_model.save_pretrained(path)
clip_processor.save_pretrained(path)

version = 'stabilityai/stable-diffusion-2-base'
path = os.path.join('./assets', version)
os.makedirs(path, exist_ok=True)
pipe = StableDiffusionPipeline.from_pretrained(version, torch_dtype=torch.float32)
pipe.save_pretrained(path)
