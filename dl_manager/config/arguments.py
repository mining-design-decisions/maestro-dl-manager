from __future__ import annotations

import abc
import re

import issue_db_api

from .constraints import Constraint
from .. import db_util


from . import core
from .parsing import ArgumentConsumer
from .parsing import ArgumentListParser
from .parsing import ArgumentParsingError
from . import schemas
from . import constraints


class Argument(abc.ABC):
    _NOT_SET = object()

    def __init__(self,
                 name: str, description: str, data_type, default=_NOT_SET, *,
                 enabled_if: constraints.ValueExpression | None = None):
        self._name = name
        self._description = description
        self._data_type = data_type
        self._default = default
        self.enabled_if = enabled_if

    @property
    def argument_name(self):
        return self._name

    @property
    def argument_description(self):
        return self._description

    @property
    def default(self):
        return self._default

    @property
    def has_default(self):
        return self._default is not self._NOT_SET

    @property
    def argument_type(self):
        return self._data_type

    def depends_on(self) -> list[str]:
        if self.enabled_if is None:
            return []
        return self.enabled_if.involved_arguments

    def is_enabled(self, conf: core.Config) -> bool:
        return self.enabled_if is None or self.enabled_if.evaluate(conf)

    @abc.abstractmethod
    def validate(self, value, *, tuning=False):
        pass

    @abc.abstractmethod
    def legal_values(self):
        pass

    @abc.abstractmethod
    def get_json_spec(self):
        pattern = re.compile(r'([A-Z])')
        parts = [
            x
            for x in pattern.split(self.__class__.__name__.replace('Argument', ''))
            if x
        ]
        arg_type = '-'.join(
            f'{x}{y}' for x, y in zip(parts[::2], parts[1::2])
        ).lower()
        return {
            "name": self._name,
            "description": self._description,
            "type": self._data_type.__name__,
            'argument_type': arg_type,
            "has-default": self.has_default,
            "default": self._default if self.has_default else None,
            "readable-options": self.legal_values(),
            "supported-hyper-param-specs": self.supported_hyper_param_specs(),
            'enabled-if': self.enabled_if.to_json() if self.enabled_if is not None else None
        }

    @staticmethod
    @abc.abstractmethod
    def supported_hyper_param_specs():
        return []

    def raise_invalid(self, msg):
        raise ArgumentParsingError(f"Argument {self.argument_name!r} is invalid: {msg}")


class ListArgument(Argument):

    def __init__(self,
                *,
                default=Argument._NOT_SET,
                inner: Argument):
        super().__init__(inner.argument_name,
                         f'{inner.argument_description} (multi-valued)',
                         list,
                         default,
                         enabled_if=inner.enabled_if)
        self._inner = inner

    def validate(self, value, *, tuning=False):
        if not isinstance(value, list):
            self.raise_invalid(f'Expected a list of values')
        return [self._inner.validate(x, tuning=tuning) for x in value]

    def get_json_spec(self):
        return super().get_json_spec() | {
            'inner': self._inner.get_json_spec()
        }

    def legal_values(self):
        return self._inner.legal_values()

    @staticmethod
    def supported_hyper_param_specs():
        return ['values']



class FloatArgument(Argument):
    def __init__(
        self,
        name: str,
        description: str,
        default=Argument._NOT_SET,
        minimum: float | None = None,
        maximum: float | None = None, *,
        enabled_if: constraints.ValueExpression | None = None
    ):
        super().__init__(name, description, float, default, enabled_if=enabled_if)
        self._min = minimum
        self._max = maximum

    def validate(self, value, *, tuning=False):
        if not isinstance(value, float):
            self.raise_invalid(f"Must be float, got {value.__class__.__name__}")
        if self._min is not None and value < self._min:
            self.raise_invalid(f"Must be >= {self._min}")
        if self._max is not None and value > self._max:
            self.raise_invalid(f"Must be <= {self._max}")
        return value

    def legal_values(self):
        lo = self._min if self._min is not None else -float("inf")
        hi = self._max if self._max is not None else float("inf")
        return f"[{lo}, {hi}]"

    def get_json_spec(self):
        return super().get_json_spec() | {"minimum": self._min, "maximum": self._max}

    @staticmethod
    def supported_hyper_param_specs():
        return ["values", "floats"]


class IntArgument(Argument):
    def __init__(
        self,
        name: str,
        description: str,
        default=Argument._NOT_SET,
        minimum: int | None = None,
        maximum: int | None = None, *,
        enabled_if: constraints.ValueExpression | None = None
    ):
        super().__init__(name, description, int, default, enabled_if=enabled_if)
        self._min = minimum
        self._max = maximum

    def validate(self, value, *, tuning=False):
        if not isinstance(value, int):
            self.raise_invalid(f"Must be int, got {value.__class__.__name__}")
        if self._min is not None and value < self._min:
            self.raise_invalid(f"Must be >= {self._min}")
        if self._max is not None and value > self._max:
            self.raise_invalid(f"Must be <= {self._max}")
        return value

    def legal_values(self):
        lo = self._min if self._min is not None else -float("inf")
        hi = self._max if self._max is not None else float("inf")
        return f"[{lo}, {hi}]"

    def get_json_spec(self):
        return super().get_json_spec() | {"minimum": self._min, "maximum": self._max}

    @staticmethod
    def supported_hyper_param_specs():
        return ["values", "range"]


class EnumArgument(Argument):
    def __init__(
        self,
        name: str,
        description: str,
        default=Argument._NOT_SET,
        options: list[str] = None, *,
        enabled_if: constraints.ValueExpression | None = None
    ):
        super().__init__(name, description, str, default, enabled_if=enabled_if)
        if options is None:
            self._options = []
        else:
            self._options = options

    def validate(self, value, *, tuning=False):
        if not isinstance(value, str):
            self.raise_invalid(f"Must be string, got {value.__class__.__name__}")
        if value not in self._options:
            self.raise_invalid(f'Must be one of {", ".join(self._options)}')
        return value

    def legal_values(self):
        return self._options

    def get_json_spec(self):
        return super().get_json_spec() | {"options": self._options}

    @staticmethod
    def supported_hyper_param_specs():
        return ["values"]


class DynamicEnumArgument(EnumArgument):

    def __init__(self,
                 name: str,
                 description: str,
                 default=Argument._NOT_SET, *,
                 lookup_map,
                 enabled_if: constraints.ValueExpression | None = None):
        super().__init__(name, description, default, list(lookup_map), enabled_if=enabled_if)


class BoolArgument(Argument):
    def __init__(self,
                 name: str,
                 description: str,
                 default=Argument._NOT_SET, *,
                 enabled_if: constraints.ValueExpression | None = None):
        super().__init__(name, description, bool, default, enabled_if=enabled_if)

    def validate(self, value, *, tuning=False):
        if isinstance(value, bool) or (isinstance(value, int) and value in (0, 1)):
            return value
        self.raise_invalid(f"Must be Boolean, got {value.__class__.__name__}")

    def legal_values(self):
        return [False, True]

    def get_json_spec(self):
        return super().get_json_spec() | {}

    @staticmethod
    def supported_hyper_param_specs():
        return ["values"]


class StringArgument(Argument):
    def __init__(self,
                 name: str,
                 description: str,
                 default=Argument._NOT_SET, *,
                 enabled_if: constraints.ValueExpression | None = None):
        super().__init__(name, description, str, default, enabled_if=enabled_if)

    def validate(self, value, *, tuning=False):
        if not isinstance(value, str):
            self.raise_invalid(f"Must be string, got {value.__class__.__name__}")
        return value

    def legal_values(self):
        return "Any"

    def get_json_spec(self):
        return super().get_json_spec() | {}

    @staticmethod
    def supported_hyper_param_specs():
        return ["values"]


class JSONArgument(Argument):

    def __init__(self,
                 name: str,
                 description: str,
                 schema: schemas.JSONSchema,
                 default=Argument._NOT_SET, *,
                 enabled_if: constraints.ValueExpression | None = None):
        super().__init__(name, description, object, default, enabled_if=enabled_if)
        self._schema = schema


    def validate(self, value, *, tuning=False):
        try:
            self._schema.validate(value)
        except schemas.SchemaValueMismatch as e:
            self.raise_invalid(f'Value {value} does not match schema: {e}')
        return value

    def legal_values(self):
        return self._schema.serialize()

    def get_json_spec(self):
        super().get_json_spec() | {
            'schema': self._schema.serialize()
        }

    @staticmethod
    def supported_hyper_param_specs():
        return ['values']


class QueryArgument(Argument):
    def __init__(self,
                 name: str,
                 description: str,
                 default=Argument._NOT_SET, *,
                 enabled_if: constraints.ValueExpression | None = None):
        super().__init__(name, description, issue_db_api.Query, default, enabled_if=enabled_if)

    def validate(self, value, *, tuning=False):
        if value is None:
            return value
        try:
            return db_util.object_to_query(value)
        except Exception as e:
            self.raise_invalid(f"Invalid query: {e}")

    def legal_values(self):
        return ""

    def get_json_spec(self):
        return super().get_json_spec() | {}

    @staticmethod
    def supported_hyper_param_specs():
        return ["values"]


class NestedArgument(Argument):

    class _Wrapper(ArgumentConsumer):
        def __init__(self, payload, const):
            self._payload = payload
            self._constraint = const

        def get_constraints(self) -> list[Constraint]:
            return self._constraint

        def get_arguments(self) -> dict[str, Argument]:
            return self._payload


    def __init__(self,
                 name: str,
                 description: str, *,
                 spec: dict[str, dict[str, Argument]],
                 constraint_spec: dict[str, list[Constraint]] | None = None,
                 tunable=False,
                 multi_valued,
                 enabled_if: constraints.ValueExpression | None = None):
        # Check if all children have defaults
        if all(x.has_default for cls in spec.values() for x in cls.values()):
            default = {
                key: [{k: v.default for k, v in value.items()}]
                for key, value in spec.items()
            }
        else:
            default = self._NOT_SET
        super().__init__(name, description, dict, default=default, enabled_if=enabled_if)
        self._tunable = tunable
        self._raw_spec = spec
        if constraint_spec is None:
            constraint_spec = {}
        self._raw_constraints = constraint_spec
        self._multi_valued = multi_valued
        self._spec = {
            key: self._Wrapper(value, [] if constraint_spec == {} else constraint_spec[key])
            for key, value in spec.items()
        }
        self._parser = ArgumentListParser(name,
                                          self._spec,
                                          multi_valued=self._multi_valued,
                                          tunable_arguments=False)
        self._hyper_parser = ArgumentListParser(name,
                                                self._spec,
                                                multi_valued=self._multi_valued,
                                                tunable_arguments=True)

    def validate(self, value, *, tuning=False):
        try:
            if tuning and self._tunable:
                return self._hyper_parser.validate(value)
            return self._parser.validate(value)
        except ArgumentParsingError as e:
            self.raise_invalid(e.message)

    def legal_values(self):
        return {}

    def get_json_spec(self):
        return super().get_json_spec() | {
            'spec': {
                key: {k: v.get_json_spec() for k, v in value.items()}
                for key, value in self._raw_spec.items()
            },
            'constraint-spec': {
                key: [v.to_json() for v in value]
                for key, value in self._raw_constraints.items()
            },
            'tunable': self._tunable,
            'multi-valued': self._multi_valued
        }

    @staticmethod
    def supported_hyper_param_specs():
        return ['nested']
