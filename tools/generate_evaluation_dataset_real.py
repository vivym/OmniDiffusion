import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision.transforms import functional as TrF, InterpolationMode
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_path",
        type=str,
        default="data/coco",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/coco/fid_dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
    )
    args = parser.parse_args()

    coco_path = Path(args.coco_path)
    output_path = Path(args.output_path)

    with open(coco_path / "annotations/captions_val2014.json") as f:
        captions = json.load(f)
        annotations = captions["annotations"]

    df = pd.DataFrame(annotations)
    df["caption"] = df["caption"].apply(lambda x: x.replace("\n", "").strip())

    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    df = df.drop_duplicates(subset=["image_id"], keep="first")

    df = df[:30000]

    df = df.sort_values(by=["id"])

    if not output_path.exists():
        output_path.mkdir(parents=True)

    df.to_csv(output_path / "metadata.csv", sep="\t", index=False)

    output_images_path = output_path / "images"
    if not output_images_path.exists():
        output_images_path.mkdir(parents=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_fname = f"COCO_val2014_{row['image_id']:012}.jpg"

        image = Image.open(coco_path / "val2014" / image_fname)

        image = TrF.resize(
            image,
            size=512,
            interpolation=InterpolationMode.LANCZOS,
        )

        image = TrF.center_crop(image, output_size=(512, 512))

        image.save(output_images_path / image_fname)


if __name__ == "__main__":
    main()
