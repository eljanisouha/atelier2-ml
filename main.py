import argparse
from pipeline import prepare, train, evaluate, test

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ML - Wine Quality"
    )

    parser.add_argument(
        "step",
        nargs="?",
        choices=["prepare", "train", "evaluate", "test", "all"],
        default="all",
        help="Étape à exécuter"
    )

    args = parser.parse_args()

    if args.step == "prepare":
        prepare()

    elif args.step == "train":
        train()

    elif args.step == "evaluate":
        evaluate()

    elif args.step == "test":
        test()

    elif args.step == "all":
        prepare()
        train()
        evaluate()
        test()

if __name__ == "__main__":
    main()
