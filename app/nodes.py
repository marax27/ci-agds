from enum import Enum
from typing import List
import pandas as pd
import numpy as np


def is_class_attribute(attribute_name: str) -> bool:
    return attribute_name == 'class'


class Node:
    def __init__(self, label):
        self.label = label

    def __lt__(self, other: 'Node') -> bool:
        return self.label < other.label

    def __str__(self) -> str:
        return f'<Node({self.label})>'


class ObjectNode(Node):
    def __init__(self, label, row: pd.Series, root):
        super().__init__(label)
        self.values: List[ValueNode] = [root.get_attribute(name).get_value(value) for name, value in row.items()]


class ValueNode(Node):

    def __init__(self, value, attribute: 'AttributeNode'):
        super().__init__(value)
        self.attribute = attribute
        self.objects = []

    def connect_object(self, object_node: ObjectNode):
        self.objects.append(object_node)


class AttributeNode(Node):

    class Types(Enum):
        NUMERICAL = 1
        CATEGORICAL = 2

    class NumericalDetails:
        def __init__(self):
            self.min = None
            self.max = None

    def __init__(self, label: str, column_data: pd.Series, root):
        super().__init__(label)
        self.root = root
        self.type = self.Types.CATEGORICAL if is_class_attribute(label) else self.Types.NUMERICAL
        self.values = self._initialise_values(column_data)

    def get_value(self, value):
        return next(x for x in self.values if x.label == value)

    def _initialise_values(self, column: pd.Series) -> List[ValueNode]:
        values = list(sorted(set(column)))
        return [ValueNode(value, self) for value in values]

