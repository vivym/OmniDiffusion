import io
import os
import json
import random
import shutil
from typing import Any

import numpy as np
import torch
from PIL import Image
from streaming import StreamingDataset
from transformers import AutoTokenizer
from torchvision.transforms import functional as TrF, InterpolationMode


def select_prompt(
    row: dict[str, Any],
    prompt_columns: list[str],
    proportion_empty_prompts: float = 0.0,
) -> str:
    if random.random() < proportion_empty_prompts:
        row["prompt"] = ""
        return row

    prompts = []

    for prompt_column in prompt_columns:
        prompt = row[prompt_column]
        if isinstance(prompt, str):
            prompts.append(prompt)
        elif isinstance(prompt, (list, np.ndarray)):
            prompts.extend(list(prompt))
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

    assert len(prompts) > 0, prompts

    return random.choice(prompts)


def process_image(
    image_bytes: bytes,
    target_size: tuple[int, int] | None = None,
    center_crop: bool = False,
    random_flip: bool = False,
) -> dict[str, Any]:
    image: Image.Image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    original_size = (image.height, image.width)
    if target_size is None:
        target_size = original_size

    image: Image.Image = TrF.resize(
        image,
        size=min(*target_size),
        interpolation=InterpolationMode.LANCZOS,
    )

    if center_crop:
        y1 = max(0, int(round(image.height - target_size[0]) / 2.0))
        x1 = max(0, int(round(image.width - target_size[1]) / 2.0))
    else:
        y1 = random.randint(0, max(0, image.height - target_size[0]))
        x1 = random.randint(0, max(0, image.width - target_size[1]))

    image = TrF.crop(image, y1, x1, target_size[0], target_size[1])

    if random_flip and random.random() < 0.5:
        x1 = image.width - x1 - target_size[1]
        image = TrF.hflip(image)

    image: torch.Tensor = TrF.to_tensor(image)
    image = TrF.normalize(image, mean=[0.5], std=[0.5])

    original_size = np.array(original_size, dtype=np.int64)
    crop_top_left = np.array((y1, x1), dtype=np.int64)
    target_size = np.array(target_size, dtype=np.int64)

    return image.numpy(), original_size, crop_top_left, target_size


class MMStreamingDataset(StreamingDataset):
    def __init__(
        self,
        dataset_path: str,
        model_name_or_path: str,
        model_revision: str | None = None,
        num_tokenizers: int = 1,
        resolution: int | tuple[int, int] | None = None,
        proportion_empty_prompts: float = 0.0,
        center_crop: bool = True,
        random_flip: bool = False,
        shuffle_seed: int = 42,
    ):
        cache_path = "/home/mingyang/projs/OmniDiffusion/tmp/cache16"

        # if os.path.exists(cache_path):
        #     shutil.rmtree(cache_path)

        super().__init__(
            local=cache_path,
            remote=dataset_path,
            shuffle=True,
            cache_limit="80gb",
            shuffle_seed=shuffle_seed,
        )

        self.model_name_or_path = model_name_or_path
        self.model_revision = model_revision
        self.num_tokenizers = num_tokenizers
        self.resolution = resolution
        self.proportion_empty_prompts = proportion_empty_prompts
        self.center_crop = center_crop
        self.random_flip = random_flip

    def get_item(self, idx: int):
        obj = super().get_item(idx)

        if random.random() < self.proportion_empty_prompts:
            prompt = ""
        else:
            prompt = random.choice(json.loads(obj["prompts"]))

        if not hasattr(self, "tokenizer_one"):
            self.tokenizer_one = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                revision=self.model_revision,
                subfolder="tokenizer",
                use_fast=True,
            )

            if self.num_tokenizers > 1:
                self.tokenizer_two = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    revision=self.model_revision,
                    subfolder="tokenizer_2",
                    use_fast=True,
                )
            else:
                self.tokenizer_two = None

        input_ids_one = self.tokenizer_one(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_one.model_max_length,
            return_tensors="np",
        ).input_ids[0]

        res = {}
        if self.tokenizer_two is None:
            res["input_ids"] = input_ids_one
        else:
            input_ids_two = self.tokenizer_two(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_two.model_max_length,
                return_tensors="np",
            ).input_ids[0]

            res["input_ids_one"] = input_ids_one
            res["input_ids_two"] = input_ids_two

        image, original_size, crop_top_left, target_size = process_image(
            obj["image"],
            target_size=(512, 512),
            center_crop=self.center_crop,
            random_flip=self.random_flip,
        )
        res["image"] = image
        res["original_size"] = original_size
        res["crop_top_left"] = crop_top_left
        res["target_size"] = target_size

        return res
