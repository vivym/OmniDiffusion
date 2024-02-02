import argparse
from pathlib import Path

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="data/coco/fid_fake")

    args = parser.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.set_progress_bar_config(disable=True)

    metadata = pd.read_csv("data/coco/fid_dataset/metadata.csv", sep="\t")
    prompts = metadata["caption"].tolist()

    for cfg_scale in range(2, 15):
        output_dir = Path(args.output_path) / f"{cfg_scale:03d}"

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        batch_size = 64
        for i in tqdm(range(0, len(prompts), batch_size)):
            images = pipe(
                prompts[i:i + batch_size],
                width=512,
                height=512,
                guidance_scale=cfg_scale,
                guidance_rescale=0.7,
                generator=[
                    torch.Generator(device="cuda").manual_seed(42 + j)
                    for j in range(batch_size)
                ],
            ).images

            for j, image in enumerate(images):
                image.save(output_dir / f"{i + j:06d}.jpg")


if __name__ == "__main__":
    main()
