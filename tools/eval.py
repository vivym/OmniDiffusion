import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    state_dict = torch.load(args.ckpt, map_location="cpu")
    print(state_dict.keys())


if __name__ == "__main__":
    main()
