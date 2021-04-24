from argparse import ArgumentParser
import pandas as pd

from .agds import Agds
from .nodes import AttributeNode


def main(dataset_path: str):
    df = pd.read_csv(dataset_path)
    print(df.dtypes)
    print(df.head())

    agds = Agds.from_dataframe(df)
    for attribute in agds.attributes:
        display_attribute_summary(attribute)


def display_attribute_summary(attribute: AttributeNode):
    print(f'== {attribute.label} ==')
    if attribute.details:
        print(f'Details: {attribute.details}')
    print(', '.join(f'{val.label}' + (f' ({val.get_count()})' if val.get_count() > 1 else '') for val in attribute.values))


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
