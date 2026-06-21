import argparse
from process_data import split_data
from pre_train import pre_train
from fn_train import fine_tune


def main():
    parser = argparse.ArgumentParser(description="Drug-Drug Interaction Prediction")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = sub.add_parser(
        "train", help="Start training (pre-training, fine-tuning, or both)"
    )
    p_train.add_argument(
        "-m",
        "--run-mode",
        default="all",
        choices=["pre-train", "fine-tune", "all"],
        help="run pre-training or fine-tuning or both (default: 'all')",
    )
    p_train.add_argument(
        "configs",
        nargs="+",
        help="One or more BaseConfig subclass names in the config module",
    )
    p_train.set_defaults(func=lambda args: train(args))

    # ---- split-data ----
    p_split = sub.add_parser("split", help="Split raw data into train/val/test")
    p_split.add_argument(
        "--split-type",
        default="random",
        choices=["random", "cluster"],
        help="Split strategy (default: random)",
    )
    p_split.add_argument(
        "--ratio-tuple",
        type=float,
        nargs=3,
        default=(0.7, 0.1, 0.2),
        metavar=("TRAIN", "VAL", "TEST"),
        help="split ratio for train/val/test (default: 0.7 0.1 0.2)",
    )
    p_split.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    p_split.set_defaults(
        func=lambda args: split_data(
            split_type=args.split_type, ratio_tuple=args.ratio_tuple, seed=args.seed
        )
    )

    args = parser.parse_args()
    args.func(args)


def train(args):
    configs = args.configs
    mode = args.run_mode

    if mode in ("pre-train", "all"):
        pre_train(configs)
    elif mode in ("fine-tune", "all"):
        fine_tune(configs)


if __name__ == "__main__":
    main()
