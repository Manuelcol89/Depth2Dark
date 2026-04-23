import argparse
import os
import pandas as pd

import torch

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler
)
from diffusers.utils import load_image

args = argparse.ArgumentParser()
args.add_argument('--base_model_path', type=str, default='ByteDance/sd2.1-base-zsnr-laionaes6')
args.add_argument('--controlnet_path', type=str, default='/path/to/depth2dark/controlnet')
args.add_argument('--depth_dir', type=str, default='/path/to/other_bright_dataset/depth')
args.add_argument('--output_dir', type=str, default='/path/to/output/folder')
args.add_argument('--sample_num', type=int, default=4)
args = args.parse_args()

metadata_path = '/path/to/other_bright_dataset_captions/metadata.jsonl'
conditioning_images_dir = args.depth_dir

controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16, use_safetensors=True,
                                             local_file_only=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    args.base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# comment following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()


generator = torch.manual_seed(0)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

metadata = pd.read_json(metadata_path, lines=True)
metadata.pop("image")

for _, row in metadata.iterrows():

    filename = row["conditioning_image"]
    print(filename)

    # conditioning_image_path = row["conditioning_image"]
    conditioning_image_path = os.path.join(
        conditioning_images_dir, row["conditioning_image"]
    )

    prompt = row["text"]
    control_image = load_image(conditioning_image_path)

    # generate image
    index = 0
    for i in range(args.sample_num):
        image = pipe(prompt, num_inference_steps=20, generator=generator, image=control_image, guidance_scale=5).images[
            0]
        image.save(os.path.join(args.output_dir, "{:s}_{:02d}.png".format(filename[:-4], index)))
        index += 1
