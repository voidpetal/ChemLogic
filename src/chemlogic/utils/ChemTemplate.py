from abc import abstractmethod

from neuralogic.core import Model
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.core.constructs.rule import Rule


class ChemTemplate(Model):
    """
    Abstract class representing a template.

    Inherits from neuralogic.core.Model and adds functionality for template operations.
    """

    def __init__(self):
        super().__init__()

    def __add__(self, other):
        if isinstance(other, Model):
            self.add_rules(list(other))

        elif isinstance(other, list):
            self.add_rules(other)
        else:
            raise NotImplementedError(f"Cannot add `{type(self)}` and `{type(other)}`")
        return self

    # TODO: integrate this with adding logic to make it faster
    def flatten(self):
        template = []
        for rule in self:
            if isinstance(rule, BaseRelation | WeightedRelation | Rule):
                template.append(rule)
        self._model = template

    @abstractmethod
    def create_template(self):
        pass
