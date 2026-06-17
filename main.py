import argparse
from process_data import split_data
from pre_training import pre_train


def main():
    parser = argparse.ArgumentParser(description="Drug-Drug Interaction Prediction")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- pre-train ----
    p_train = sub.add_parser("pre-train", help="Start one or more pre-training runs")
    p_train.add_argument(
        "configs",
        nargs="+",
        help="One or more BaseConfig subclass names in the config module",
    )
    p_train.add_argument(
        "--run-mode",
        default="all",
        choices=["train", "test", "all"],
        help="Run mode: 'train' (training only), 'test' (testing only), or 'all' (both, default)",
    )
    p_train.set_defaults(func=lambda args: pre_train(args.configs,args.run_mode))

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


if __name__ == "__main__":
    main()
