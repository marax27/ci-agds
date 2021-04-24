from argparse import ArgumentParser
import pandas as pd

from .agds import Agds
from .nodes import AttributeNode, ObjectNode


def main(dataset_path: str):
    df = pd.read_csv(dataset_path)
    print(df.dtypes)
    print(df.head())

    agds = Agds.from_dataframe(df)
    for attribute in agds.attributes:
        display_attribute_summary(attribute)

    agds.find_similarity_to_object('O111')
    for obj in sorted(agds.objects, key=lambda obj: -obj.similarity)[:10]:
        print(f'({obj.similarity}) : ' + write_object_summary(obj))


def display_attribute_summary(attribute: AttributeNode):
    print(f'== {attribute.label} ==')
    if attribute.details:
        print(f'Details: {attribute.details}')
    print(', '.join(f'{val.label}' + (f' ({val.get_count()})' if val.get_count() > 1 else '') for val in attribute.values))


def write_object_summary(node: ObjectNode) -> str:
    return f'{node.label}:[' + ', '.join(f'{x.attribute.label}: {x.label}' for x in node.values)


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
