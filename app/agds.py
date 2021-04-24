from typing import Dict, Iterable, List
import pandas as pd
from .nodes import ObjectNode, AttributeNode


class Agds:
    def __init__(self) -> None:
        self.objects: List[ObjectNode] = []
        self.attributes: List[AttributeNode] = []

    def get_attribute(self, label: str) -> AttributeNode:
        return next(attr for attr in self.attributes if attr.label == label)

    def get_object(self, label: str) -> ObjectNode:
        return next(node for node in self.objects if node.label == label)

    def find_similar_to_object(self, object_id: str) -> Iterable[ObjectNode]:
        for obj in self.objects:
            obj.similarity = 0.0
        target_obj = next(x for x in self.objects if x.label == object_id)
        target_obj.infer(1.0)
        return sorted(self.objects, key=lambda obj: -obj.similarity)

    def find_similar_by_values(self, values: Dict[str, float]) -> Iterable[ObjectNode]:
        for obj in self.objects:
            obj.similarity = 0.0
        corresponding_nodes = [self.get_attribute(name).get_nearest(value) for name, value in values.items()]
        for value in corresponding_nodes:
            value.infer(1.0)
        return sorted(self.objects, key=lambda obj: -obj.similarity)

    def add_attribute(self, label: str, column_data):
        self.attributes.append(AttributeNode(label, column_data, self))

    def add_object(self, label: str, row: pd.Series):
        self.objects.append(ObjectNode(label, row, self))
        pass

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> 'Agds':
        graph = Agds()

        for label, column in df.items():
            graph.add_attribute(label, column)

        for index, row in df.iterrows():
            graph.add_object(f'O{index + 1}', row)
        return graph

