from __future__ import annotations

import abc
import graphlib
import logging
import warnings

from .arguments import Argument, NestedArgument
from .constraints import Constraint
from .core import ConfigFactory, NoSuchSetting, NotSet
from . import schemas


class ArgumentConsumer:

    @classmethod
    @abc.abstractmethod
    def get_arguments(cls) -> dict[str, Argument]:
        return {}

    @classmethod
    @abc.abstractmethod
    def get_constraints(cls) -> list[Constraint]:
        return []
    
    
class ArgumentParsingError(Exception):
    
    def __init__(self, message: str, *, is_user_error=True):
        super().__init__(message)
        self.message = message 
        self.is_user_error = is_user_error


class ArgumentListParser:

    def __init__(self, name: str, lookup_map, *,
                 multi_valued=False,
                 tunable_arguments=False,
                 logger: logging.Logger | None = None):
        self._map = lookup_map
        self._name = name
        self._multi_valued = multi_valued
        self._tunable = tunable_arguments
        self._log = logger if logger is not None else lambda *a, **kw: ...

    # ----- Validation Utilities -----

    def _raise_invalid(self, message: str):
        raise ArgumentParsingError(f'Error while parsing argument {self._name!r}: {message}')

    def _check_type(self, key, obj, tp):
        if not isinstance(obj, tp):
            self._raise_invalid(f'{key} expects a value of type {tp.__name__}, got {obj.__class__.__name__}')

    # ----- Helpers -----

    def _find_class(self, name: str):
        if name not in self._map:
            self._raise_invalid(f'Unknown class name: {name}')
        return self._map[name]

    # ----- Main Validation Logic -----

    def validate(self, values):
        self._check_type(self._name, values, dict)
        if len(values) != 1 and not self._multi_valued:
            self._raise_invalid('Single-valued argument list can only contain a single key.')
        args_by_instance = {}
        default_by_class = {}
        defaults = None
        for name, args in values().items():
            if '.' in name:
                cls, index, parsed = self._validate_instance_args(name, args)
                args_by_instance.setdefault(cls, {})[index] = parsed
            elif name != 'default':
                if not self._multi_valued:
                    self._raise_invalid('Single-valued argument list may not contain a class-default section.')
                parsed = self._validate_cls_defaults(name, args)
                default_by_class[name] = parsed
            else:
                if not self._multi_valued:
                    self._raise_invalid('Single-valued argument list may not contain a default section.')
                defaults = self._validate_global_defaults(args)
        self._apply_defaults(args_by_instance, default_by_class, defaults)
        self.impose_constraints(args_by_instance)
        return args_by_instance


    def _validate_instance_args(self, name: str, args):
        cls, index = name.split('.')
        try:
            index = int(index)
        except ValueError:
            self._raise_invalid(f'Invalid instance name: {name}')
        return cls, index, self._validate_args(self._find_class(cls), args)

    def _validate_cls_defaults(self, name: str, args):
        return self._validate_args(self._find_class(name), args)

    def _validate_global_defaults(self, args):
        # Find a common base class for all classes in the map
        classes = list(self._map.values())
        proposal_class, *test_classes = classes
        result = None
        for base in reversed(proposal_class.__mro__):
            if all(issubclass(c, base) for c in test_classes):
                result = base
            else:
                break
        if result is None or not issubclass(result, ArgumentConsumer):
            raise ArgumentParsingError(
                f"Failed to find base class for resolving global defaults for {self._name!r}",
                is_user_error=False
            )
        return self._validate_args(result, args)

    def _validate_args(self, cls, obj):
        self._log.info(f'Parsing argument list for class {cls.__name__}')
        args = cls.get_arguments()
        result = {}
        for arg_name in self._sort_args(args):
            arg = args[arg_name]
            try:
                if not arg.is_enabled(ConfigFactory.dict_config(result)):
                    self._log.info(f'Skipping disabled argument: {arg_name}')
            except (NoSuchSetting, NotSet) as e:
                raise ValueError(
                    f'Error while evaluating enablement constraint. '
                    f'Is one of the required arguments not enabled?') from e
            if arg_name not in obj:
                if arg.has_default:
                    parsed = self.validate_default(arg, arg.default)
                    result[arg_name] = parsed
                    self._log.info(f'Assigned default for argument {arg_name}: {parsed}')
                else:
                    self._raise_invalid(f'Missing required argument {arg_name} for {cls.__name__}')
            else:
                parsed = self.validate_value(arg, obj[arg_name])
                result[arg_name] = parsed
                self._log.info(f'Validated value for argument {arg_name}: {parsed}')
        return result

    def _sort_args(self, args: dict[str, Argument]) -> list[str]:
        graph = {}
        for name, arg in args.items():
            graph[name] = set(arg.depends_on())
        sorter = graphlib.TopologicalSorter(graph)
        try:
            return list(sorter.static_order())
        except graphlib.CycleError as e:
            msg = 'Cannot parse arguments because of cycle in enabling conditions'
            raise Exception(msg) from e

    def _apply_defaults(self, args_by_instance, defaults_by_class, defaults):
        for cls_name, value in defaults_by_class.items():
            if cls_name not in args_by_instance:
                self._raise_invalid(f"No entries (indices) to apply default {cls_name!r} to")
            for v in args_by_instance[cls_name].values():
                v.update(value)
        if defaults is not None:
            if not args_by_instance:
                self._raise_invalid(f"No entries (indices) to apply defaults to in arglist")
            for indices in args_by_instance.values():
                for v in indices.values():
                    v.update(defaults)

    def validate_value(self, argument: Argument, value):
        if self._tunable:
            return self._validate_tunable(argument, value)
        return argument.validate(value)

    def validate_default(self, argument: Argument, value):
        if self._tunable:
            if isinstance(argument, NestedArgument):
                default = {
                    f'{key}.0': {
                        k: {'type': 'values', 'options': {'values': [v]}}
                        for k, v in value[0].items()
                    }
                    for key, value in value.items()
                }
                return argument.validate(default, tuning=True)
            return self.validate_value(
                argument,
                {"type": "values", "options": {"values": [argument.validate(value, tuning=True)]}},
            )
            return self._validate_tunable(argument, value)
        if isinstance(argument, NestedArgument):
            return argument.default
        return argument.validate(value)

    # ----- Tuner Validation Logic  -----

    _SCHEMA = schemas.Choice(
        schemas.FixedObject(
            type=schemas.String('range'),
            options=schemas.FixedObject(
                ['step', 'sampling'],   # Optional arguments
                start=schemas.Integer(),
                stop=schemas.Integer(),
                step=schemas.Integer(),
                sampling=schemas.StringEnum(options=['linear', 'log', 'reverse_log']),
            )
        ),
        schemas.FixedObject(
            type=schemas.String('values'),
            options=schemas.FixedObject(
                values=schemas.Array(schemas.Any)
            )
        ),
        schemas.FixedObject(
            type=schemas.String('floats'),
            options=schemas.FixedObject(
                ['step', 'sampling'],  # Optional arguments
                start=schemas.Float(),
                stop=schemas.Float(),
                step=schemas.Float(),
                sampling=schemas.StringEnum(options=['linear', 'log', 'reverse_log']),
            )
        )
    )

    def _validate_tunable(self, argument: Argument, value):
        if isinstance(argument, NestedArgument):        # Ugly, but necessary with current design
            return argument.validate(value, tuning=True)
        arg_name = f'{self._name}.{argument.argument_name}'
        try:
            self._SCHEMA.validate(value)
        except schemas.SchemaValueMismatch as e:
            self._raise_invalid(f'{arg_name} has invalid format: {e}')
        match value['type']:
            case 'range':
                return {'type': 'range',
                        'options': self._validate_range(argument, value['options'])}
            case 'values':
                return {'type': 'values',
                        'options': self._validate_values(argument, value['options'])}
            case 'floats':
                return {'type': 'floats',
                        'options': self._validate_floats(argument, value['options'])}
            case _ as x:
                raise NotImplementedError(x)

    def _validate_range(self, validator: Argument, options: dict):
        if "range" not in validator.supported_hyper_param_specs():
            self._raise_invalid(f"Hyper param arglist item of type "
                                f"{validator.__class__.__name__} does not support range.")
        result = {'start': options['start'], 'stop': options['stop']}
        if "step" not in options:
            options["step"] = None
        if "sampling" not in options:
            options["sampling"] = "linear"
        return result

    def _validate_values(self, validator: Argument, options: dict):
        if "values" not in validator.supported_hyper_param_specs():
            self._raise_invalid(f'Hyper param arglist item of type '
                                f'{validator.__class__.__name__} does not support "values".')
        return {"values": [validator.validate(value) for value in options]}

    def _validate_floats(self, validator: Argument, options: dict):
        if "floats" not in validator.supported_hyper_param_specs():
            self._raise_invalid(f'Hyper param arglist item of type '
                                f'{validator.__class__.__name__} does not support "floats".')
        result = {'start': options['start'], 'stop': options['stop']}
        if "step" not in options:
            options["step"] = None
        if "sampling" not in options:
            options["sampling"] = "linear"
        return result

    # ----- Constraint logic -----

    def impose_constraints(self, args_by_instance):
        if self._tunable:
            warnings.warn('Constraints are not imposed in tunable argument lists.')
            return
        self._impose_constraints_single_valued(args_by_instance)

    def _impose_constraints_single_valued(self, args_by_instance):
        for cls, indices in args_by_instance.items():
            for index, instance_args in indices.items():
                constraints: list[Constraint] = cls.get_constraints()
                conf = ConfigFactory.dict_config(instance_args)
                for constraint in constraints:
                    constraint.impose(conf)
