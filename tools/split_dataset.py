from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():
    root_path = Path("./data/midjourney-v52-mar")

    for sub_dir in root_path.iterdir():
        if sub_dir.is_dir():
            if sub_dir.name != "1344_0768_002219155":
                continue

            print(sub_dir)
            for parquet_path in tqdm(sub_dir.glob("*.parquet")):
                df = pd.read_parquet(parquet_path)

                print(df.shape)
                print(df.columns)

                batch_size = 1024

                for i in range(0, df.shape[0], batch_size):
                    batch_df = df.iloc[i:i + batch_size]

                    batch_df.to_parquet(parquet_path.parent / f"{parquet_path.stem}_{i // batch_size:04d}.parquet")

                parquet_path.unlink()


if __name__ == "__main__":
    main()
