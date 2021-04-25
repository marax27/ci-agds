from argparse import ArgumentParser
from collections import Counter
import pandas as pd
from itertools import islice

from .agds import Agds
from .nodes import AttributeNode, ObjectNode


def main(dataset_path: str):
    df = pd.read_csv(dataset_path)
    print(df.dtypes)
    print(df.head())

    # Dataset presets.
    # start_iris_test(df)
    # start_penguin_test(df)

    if 'iris' in dataset_path.lower():
        start_iris_test(df)
    elif 'penguin' in dataset_path.lower():
        start_penguin_test(df)


def start_iris_test(df: pd.DataFrame):
    agds = Agds.from_dataframe(df, 'species')

    display_attribute_summary(agds)

    print('\nTop 10 objects most similar to "O111":')
    for obj in islice(agds.find_similar_to_object('O111'), 10):
        print(f'({obj.similarity}) : ' + write_object_summary(obj))

    given_values = { 'sepal_length': 4.8, 'sepal_width': 3, 'petal_length': 1.55, 'petal_width': 0.2 }
    print(f'\nObjects most similar to the value set: {given_values}')
    for obj in islice(agds.find_similar_by_values(given_values), 5):
        print(f'({obj.similarity}) : ' + write_object_summary(obj))

    print('\nTrying to classify the value set: ' + agds.classify(given_values))


def start_penguin_test(df: pd.DataFrame):
    one_hot_sex = pd.get_dummies(df['Sex'], prefix='Sex', drop_first=True)
    df = pd.concat([
        df.drop(columns=['studyName', 'Sample Number', 'Region', 'Island', 'Stage', 'Individual ID', 'Clutch Completion', 'Date Egg', 'Sex', 'Comments']),
        one_hot_sex,
    ], axis=1)
    df.dropna(inplace=True)

    print(df.head())

    agds = Agds.from_dataframe(df, 'Species')

    display_attribute_summary(agds)

    example = {
        'Culmen Length (mm)': 46.1,
        'Culmen Depth (mm)': 13.2,
        'Flipper Length (mm)': 211,
        'Body Mass (g)': 4500,
        'Delta 15 N (o/oo)': 7.993,
        'Delta 13 C (o/oo)': -25.5139,
    }
    print(f'\nClassified as {agds.classify(example)}')


def display_attribute_summary(graph: Agds):
    print('Attribute summary:')
    for attribute in graph.attributes:
        print(f'== {attribute.label} ==')
        if attribute.details:
            print(f'Value range: {attribute.details}')

        all_values = (f'{val.label}' + (f' ({val.get_count()})' if val.get_count() > 1 else '') for val in attribute.values)
        print('All values: ' + ', '.join(all_values))


def write_object_summary(node: ObjectNode) -> str:
    return f'{node.label}:[' + ', '.join(f'{x.attribute.label}: {x.label}' for x in node.values) + ']'


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('dataset_path')
    return parser


if __name__ == '__main__':
    try:
        ARGS = create_parser().parse_args()
        main(ARGS.dataset_path)
    except Exception as exc:
        print(f'Application failed. {exc.__class__.__name__}: {exc}')
        raise
