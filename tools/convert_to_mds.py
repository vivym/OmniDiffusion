import json
from pathlib import Path

import numpy as np
import pandas as pd
from streaming import MDSWriter
from tqdm import tqdm


def main():
    source_root = Path("./data/midjourney-v52-mar")
    target_root = Path("./data/midjourney-v52-mar-mds")

    for i, sub_dir in enumerate(sorted(source_root.iterdir())):
        if not sub_dir.is_dir():
            continue

        target_dir = target_root / sub_dir.name
        # target_dir.mkdir(exist_ok=True)

        print(i, sub_dir, "+" * 50)
        with MDSWriter(
            columns={
                "image": "bytes",
                "prompts": "json",
            },
            out=str(target_dir),
            compression="zstd",
            hashes=['sha1', 'xxh64'],
            size_limit=512 * 1024 * 1024,
        ) as out:
            for parquet_path in tqdm(sorted(sub_dir.glob("*.parquet"))):
                df = pd.read_parquet(parquet_path, columns=["image", "improved_prompt", "prompt"])

                for _, row in df.iterrows():
                    if isinstance(row["improved_prompt"], np.ndarray):
                        row["improved_prompt"] = row["improved_prompt"].tolist()
                    assert isinstance(row["improved_prompt"], list), row["improved_prompt"]
                    assert isinstance(row["prompt"], str), row["prompt"]
                    prompts = json.dumps(row["improved_prompt"] + [row["prompt"]])
                    out.write({
                        "image": row["image"],
                        "prompts": prompts,
                    })


if __name__ == "__main__":
    main()
