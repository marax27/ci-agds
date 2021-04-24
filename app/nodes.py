from enum import Enum
from typing import Iterable, List, Union
import pandas as pd
import numpy as np

from .utilities import NumericalDetails, binary_search


def is_class_attribute(attribute_name: str) -> bool:
    return attribute_name == 'class'


class Node:
    def __init__(self, label):
        self.label = label

    def __lt__(self, other: 'Node') -> bool:
        return self.label < other.label

    def __str__(self) -> str:
        return f'<{self.__class__.__name__}({self.label})>'


class ObjectNode(Node):
    def __init__(self, label, row: pd.Series, root):
        super().__init__(label)
        self.values: List[ValueNode] = [root.get_attribute(name).get_value(value) for name, value in row.items()]
        for value in self.values:
            value.connect_object(self)
        self.similarity = None

    def get_weight(self) -> float:
        return 1.0

    def infer(self, similarity):
        self.similarity = similarity
        for value in self.values:
            value.infer(similarity * self.get_weight(), self)


class ValueNode(Node):

    def __init__(self, value, attribute: 'AttributeNode'):
        super().__init__(value)
        self.attribute = attribute
        self.objects: List[ObjectNode] = []
        self.similarity = None
        self.previous = None
        self.next = None

    def get_count(self) -> int:
        return len(self.objects)

    def connect_object(self, object_node: ObjectNode):
        self.objects.append(object_node)

    def get_weight(self, other: Union[ObjectNode, 'ValueNode']) -> float:
        attribute = self.attribute
        if isinstance(other, ObjectNode):
            return 1.0 / len(attribute.root.attributes)
        else:
            return 1.0 - abs(self.label - other.label) / (attribute.details.max - attribute.details.min)

    def get_neighbors(self) -> Iterable['ValueNode']:
        if self.previous:  yield self.previous
        if self.next:  yield self.next

    def infer(self, similarity: float, from_node: Node = None):
        self.similarity = similarity
        for value in self.get_neighbors():
            if value is not from_node:
                value.infer(similarity * self.get_weight(value), self)
        for object_node in self.objects:
            if object_node is not from_node:
                object_node.similarity += similarity * self.get_weight(object_node)


class AttributeNode(Node):

    class Types(Enum):
        NUMERICAL = 1
        CATEGORICAL = 2

    def __init__(self, label: str, column_data: pd.Series, root):
        super().__init__(label)
        self.root = root
        self.type = self.Types.CATEGORICAL if is_class_attribute(label) else self.Types.NUMERICAL
        self.values = self._initialise_values(column_data)
        self.details = self._initialise_numerical_details(column_data) if self.type == self.Types.NUMERICAL else None

    def get_value(self, value):
        return next(x for x in self.values if x.label == value)

    def get_nearest(self, value) -> ValueNode:
        idx = binary_search(self.values, value, lambda v: v.label)
        return self.values[idx]

    def _initialise_values(self, column: pd.Series) -> List[ValueNode]:
        values = list(sorted(set(column)))
        nodes = [ValueNode(value, self) for value in values]
        if self.type == self.Types.NUMERICAL:
            self._link_nodes_in_list(nodes)
        return nodes

    def _initialise_numerical_details(self, data: pd.Series) -> NumericalDetails:
        values = list(data)
        avg = sum(values) / len(values)
        return NumericalDetails(min(values), max(values), avg)

    def _link_nodes_in_list(self, nodes: List[ValueNode]):
        for i in range(len(nodes)):
            if i > 0:
                nodes[i].previous = nodes[i - 1]
            if i < len(nodes) - 1:
                nodes[i].next = nodes[i + 1]
