from __future__ import annotations

import abc
import itertools
import typing

from .core import Config


class Constraint(abc.ABC):

    def __init__(self, *args: str, msg: str):
        self._involved = args
        self._msg = msg

    def description(self) -> str:
        return self._msg

    def applies(self, active_namespaces: list[str]) -> bool:
        namespaces = [arg.rsplit('.', maxsplit=1)[0] for arg in self._involved]
        for namespace in namespaces:
            if not any(active.startswith(namespace) for active in active_namespaces):
                return False
        return True

    @property
    @abc.abstractmethod
    def involved_arguments(self) -> list[str]:
        pass

    @abc.abstractmethod
    def impose(self, conf: Config) -> bool:
        pass

    @abc.abstractmethod
    def to_json(self):
        pass


class MutuallyExclusive(Constraint):

    def __init__(self, *args: ValueExpression, message: str, exactly_one=False):
        assert len(args) >= 2
        super().__init__(
            *(itertools.chain(*(arg.involved_arguments for arg in args))),
            msg=message
        )
        self._exactly_one = exactly_one
        self._args = args

    @property
    def involved_arguments(self) -> list[str]:
        return list(itertools.chain(*(arg.involved_arguments for arg in self._args)))

    def impose(self, conf: Config) -> bool:
        if self._exactly_one:
            return sum(arg.evaluate(conf) for arg in self._args) == 1
        return sum(arg.evaluate(conf) for arg in self._args) <= 1

    def to_json(self):
        yield {
            'arguments': self.involved_arguments,
            'constraint': {
                'type': 'mutually-exclusive',
                'options': {
                    'rules': [
                        arg.to_json() for arg in self._args
                    ]
                }
            }
        }


class Forbids(Constraint):

    def __init__(self, *,
                 main: ValueExpression,
                 message: str,
                 forbids: list[ValueExpression],
                 add_reverse_constraints=False):
        super().__init__(
            *(itertools.chain(arg.involved_arguments for arg in forbids)),
            *main.involved_arguments,
            msg=message
        )
        self._main = main
        self._forbids = forbids
        self._additional = []
        if add_reverse_constraints:
            for spec in self._forbids:
                f = Forbids(main=spec,
                            message=message,
                            forbids=[self._main],
                            add_reverse_constraints=False)
                self._additional.append(f)

    @property
    def involved_arguments(self) -> list[str]:
        involved = list(itertools.chain(*(arg.involved_arguments for arg in self._forbids)))
        return self._main.involved_arguments + involved

    def impose(self, conf: Config) -> bool:
        if self._main.evaluate(conf):
            if any(f.evaluate(conf) for f in self._forbids):
                return False
        return all(extra.impose(conf) for extra in self._additional)

    def to_json(self):
        yield {
            'arguments': self.involved_arguments,
            'constraint': {
                'type': 'forbids',
                'options': {
                    'antecedent': self._main.to_json(),
                    # Not a typo
                    'consequents': [f.to_json() for f in self._forbids]
                }
            }
        }
        for x in self._additional:
            yield from x.to_json()


class BooleanConstraint(Constraint):

    def __init__(self, expr: ValueExpression, *, message):
        super().__init__(*expr.involved_arguments, msg=message)
        self._expr = expr

    @property
    def involved_arguments(self) -> list[str]:
        return self._expr.involved_arguments

    def impose(self, conf: Config) -> bool:
        return self._expr.evaluate(conf)

    def to_json(self):
        yield {
            'arguments': self.involved_arguments,
            'constraint': {
                'type': 'boolean-expression',
                'options': {
                    'expression': self._expr.to_json()
                }
            }
        }


class ValueExpression(abc.ABC):

    @property
    @abc.abstractmethod
    def involved_arguments(self) -> list[str]:
        pass

    @abc.abstractmethod
    def evaluate(self, conf: Config) -> bool:
        pass

    @abc.abstractmethod
    def to_json(self):
        pass


class ValueSpecifier(abc.ABC):

    @property
    @abc.abstractmethod
    def involved_arguments(self) -> list[str]:
        pass

    @abc.abstractmethod
    def get_value(self, conf: Config) -> typing.Any:
        pass

    @abc.abstractmethod
    def to_json(self):
        pass


class Constant(ValueSpecifier):

    def __init__(self, value):
        self._value = value

    @property
    def involved_arguments(self) -> list[str]:
        return []

    def get_value(self, conf: Config) -> typing.Any:
        return self._value

    def to_json(self):
        return {
            'type': 'value-specifier',
            'payload': {
                'type': 'constant',
                'value': self._value
            }
        }


class ArgumentRef(ValueSpecifier):

    def __init__(self, name: str):
        self._name = name

    @property
    def involved_arguments(self) -> list[str]:
        return [self._name]

    def get_value(self, conf: Config) -> typing.Any:
        return conf.get(self._name)

    def to_json(self):
        return {
            'type': 'value-specifier',
            'payload': {
                'type': 'argument-reference',
                'name': self._name
            }
        }


class LengthOfArgument(ValueSpecifier):

    def __init__(self, name: str):
        self._name = name

    @property
    def involved_arguments(self) -> list[str]:
        return [self._name]

    def get_value(self, conf: Config) -> typing.Any:
        return len(conf.get(self._name))

    def to_json(self):
        return {
            'type': 'value-specifier',
            'payload': {
                'type': 'length-of-argument-reference',
                'name': self._name
            }
        }


class Equal(ValueExpression):

    def __init__(self, lhs: ValueSpecifier, rhs: ValueSpecifier):
        self._lhs = lhs
        self._rhs = rhs

    @property
    def involved_arguments(self) -> list[str]:
        return self._lhs.involved_arguments + self._rhs.involved_arguments

    def evaluate(self, conf: Config) -> bool:
        return self._lhs.get_value(conf) == self._rhs.get_value(conf)

    def to_json(self):
        return {
            'type': 'value-check',
            'payload': {
                'lhs': self._lhs.to_json(),
                'operation': 'equal',
                'rhs': self._rhs.to_json()
            }
        }


class NotEqual(ValueExpression):

    def __init__(self, lhs: ValueSpecifier, rhs: ValueSpecifier):
        self._lhs = lhs
        self._rhs = rhs

    @property
    def involved_arguments(self) -> list[str]:
        return self._lhs.involved_arguments + self._rhs.involved_arguments

    def evaluate(self, conf: Config) -> bool:
        return self._lhs.get_value(conf) != self._rhs.get_value(conf)

    def to_json(self):
        return {
            'type': 'value-check',
            'payload': {
                'lhs': self._lhs.to_json(),
                'operation': 'not-equal',
                'rhs': self._rhs.to_json()
            }
        }


class ListContains(ValueExpression):

    def __init__(self, lhs: ValueSpecifier, rhs: ValueSpecifier):
        self._lhs = lhs
        self._rhs = rhs

    @property
    def involved_arguments(self) -> list[str]:
        return self._lhs.involved_arguments + self._rhs.involved_arguments

    def evaluate(self, conf: Config) -> bool:
        return self._rhs.get_value(conf) in self._lhs.get_value(conf)

    def to_json(self):
        return {
            'type': 'value-check',
            'payload': {
                'container': self._lhs.to_json(),
                'operation': 'contains',
                'value': self._rhs.to_json()
            }
        }


class ListNotContains(ValueExpression):

    def __init__(self, lhs: ValueSpecifier, rhs: ValueSpecifier):
        self._lhs = lhs
        self._rhs = rhs

    @property
    def involved_arguments(self) -> list[str]:
        return self._lhs.involved_arguments + self._rhs.involved_arguments

    def evaluate(self, conf: Config) -> bool:
        return self._rhs.get_value(conf) not in self._lhs.get_value(conf)

    def to_json(self):
        return {
            'type': 'value-check',
            'payload': {
                'container': self._lhs.to_json(),
                'operation': 'not-contains',
                'value': self._rhs.to_json()
            }
        }


class Or(ValueExpression):

    def __init__(self, lhs: ValueExpression, rhs: ValueExpression):
        self._lhs = lhs
        self._rhs = rhs

    @property
    def involved_arguments(self) -> list[str]:
        return self._lhs.involved_arguments + self._rhs.involved_arguments

    def evaluate(self, conf: Config) -> bool:
        return self._lhs.evaluate(conf) or self._rhs.evaluate(conf)

    def to_json(self):
        return {
            'type': 'value-expression',
            'payload': {
                'operation': 'or',
                'lhs': self._lhs.to_json(),
                'rhs': self._rhs.to_json()
            }
        }


class AlwaysTrue(ValueExpression):

    @property
    def involved_arguments(self) -> list[str]:
        return []

    def evaluate(self, conf: Config) -> bool:
        return True

    def to_json(self):
        return {
            'type': 'value-expression',
            'payload': {
                'operation': 'always-true'
            }
        }
