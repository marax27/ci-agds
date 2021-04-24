from argparse import ArgumentParser
from typing import List
import pandas as pd

from .agds import Agds


def main(dataset_path: str):
    df = pd.read_csv(dataset_path)
    print(df.dtypes)
    print(df.head())

    agds = Agds.from_dataframe(df)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('dataset_path')
    return parser


if __name__ == '__main__':
    try:
        ARGS = create_parser().parse_args()
        main(ARGS.dataset_path)
    except Exception as exc:
        print(f'Application failed: {exc}')
        raise
