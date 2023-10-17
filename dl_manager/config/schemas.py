from __future__ import annotations

import abc


class _SchemaBase(abc.ABC):

    @abc.abstractmethod
    def serialize(self) -> object:
        pass

    @abc.abstractmethod
    def validate(self, o: object):
        pass


class SchemaValueMismatch(Exception):

    def __init__(self, schema: JSONSchema, value: object, message: str):
        super().__init__(schema, value, message)
        self.schema = schema
        self.value = value
        self.message = message


class Choice(_SchemaBase):

    def __init__(self, *options: JSONSchema):
        self.options = options

    def serialize(self) -> object:
        return {
            'type': 'choice',
            'payload': {
                'options': [
                    opt.serialize() for opt in self.options
                ]
            }
        }

    def validate(self, o: object):
        errors = []
        for opt in self.options:
            try:
                opt.validate(o)
            except SchemaValueMismatch as e:
                errors.append(e)
            else:
                return
        group = ExceptionGroup('No option matched', errors)
        raise SchemaValueMismatch(
            self.serialize(), o, 'No matching choice found'
        ) from group


class DynamicObject(_SchemaBase):

    def serialize(self) -> object:
        return {
            'type': 'dynamic-object',
            'payload': {}
        }

    def validate(self, o: object):
        if not isinstance(o, dict):
            raise SchemaValueMismatch(self.serialize(),
                                      o,
                                      'Expected an object')



class FixedObject(_SchemaBase):

    def __init__(self, optional=None, /, **fields: JSONSchema):
        self.optional = {} if optional is None else optional
        self.fields = fields

    def serialize(self) -> object:
        return {
            'type': 'fixed-object',
            'payload': {
                'fields': {
                    name: value.serialize() for name, value in self.fields.items()
                },
                'optionals': self.optional
            }
        }

    def validate(self, o: object):
        # Check type
        if not isinstance(o, dict):
            raise SchemaValueMismatch(self.serialize(),
                                      o,
                                      'Expected an object')
        # Check that all required keys are present
        for field, schema in self.fields.items():
            if field not in o and field not in self.optional:
                raise SchemaValueMismatch(self.serialize(),
                                          o,
                                          f'Missing field {field!r}')
            try:
                schema.validate(o[field])
            except SchemaValueMismatch as inner:
                raise SchemaValueMismatch(
                    self.serialize(), o, 'Validation failure in descendant value'
                ) from inner
                # Check for superfluous fields
        for field in o:
            if field not in self.fields:
                raise SchemaValueMismatch(self.serialize(),
                                          o,
                                          f'Unexpected field {field!r}')


class Array(_SchemaBase):

    def __init__(self, item_type: JSONSchema):
        self.item_type = item_type

    def serialize(self) -> object:
        return {
            'type': 'array',
            'payload': {
                'item_type': self.item_type.serialize()
            }
        }

    def validate(self, o: object):
        if not isinstance(o, list):
            raise SchemaValueMismatch(self.serialize(), o, 'Expected an array')
        for item in o:
            try:
                self.item_type.validate(item)
            except SchemaValueMismatch as inner:
                raise SchemaValueMismatch(
                    self.serialize(), o,'Validation failure in descendant value'
                ) from inner


class Boolean(_SchemaBase):

    def __init__(self, const=None):
        self.const = const

    def serialize(self) -> object:
        return {
            'type': 'boolean',
            'payload': {
                'has-exact-value': self.const is not None,
                'exact-value': self.const
            }
        }

    def validate(self, o: object):
        if not isinstance(o, bool):
            raise SchemaValueMismatch(self.serialize(), o, 'Expected a Boolean')
        if self.const is not None and o != self.const:
            raise SchemaValueMismatch(self.serialize(), o, f'Expected exact value {self.const}')


class String(_SchemaBase):

    def __init__(self, const=None):
        self.const = const

    def serialize(self) -> object:
        return {
            'type': 'string',
            'payload': {
                'has-exact-value': self.const is not None,
                'exact-value': self.const
            }
        }

    def validate(self, o: object):
        if not isinstance(o, str):
            raise SchemaValueMismatch(self.serialize(), o, 'Expected a string')
        if self.const is not None and o != self.const:
            raise SchemaValueMismatch(self.serialize(), o, f'Expected exact value {self.const}')


class StringEnum(_SchemaBase):

    def __init__(self, const=None, *, options: list[str]):
        self.const = const
        self.options = options

    def serialize(self) -> object:
        return {
            'type': 'string-enum',
            'payload': {
                'options': self.options,
                'has-exact-value': self.const is not None,
                'exact-value': self.const
            }
        }

    def validate(self, o: object):
        if not isinstance(o, str):
            raise SchemaValueMismatch(self.serialize(), o, 'Expected a string')
        if self.const is not None and o != self.const:
            raise SchemaValueMismatch(self.serialize(), o, f'Expected exact value {self.const}')
        if o not in self.options:
            raise SchemaValueMismatch(self.serialize(), o, f'Expected one of {self.options}')


class Integer(_SchemaBase):

    def __init__(self, const=None):
        self.const = const

    def serialize(self) -> object:
        return {
            'type': 'integer',
            'payload': {
                'has-exact-value': self.const is not None,
                'exact-value': self.const
            }
        }

    def validate(self, o: object):
        if not isinstance(o, int):
            raise SchemaValueMismatch(self.serialize(), o, 'Expected an integer')
        if self.const is not None and o != self.const:
            raise SchemaValueMismatch(self.serialize(), o, f'Expected exact value {self.const}')


class Float(_SchemaBase):

    def __init__(self, const=None):
        self.const = const

    def serialize(self) -> object:
        return {
            'type': 'float',
            'payload': {
                'has-exact-value': self.const is not None,
                'exact-value': self.const
            }
        }

    def validate(self, o: object):
        if not isinstance(o, float):
            raise SchemaValueMismatch(self.serialize(), o, 'Expected a float')
        if self.const is not None and o != self.const:
            raise SchemaValueMismatch(self.serialize(), o, f'Expected exact value {self.const}')


class Optional(_SchemaBase):

    def __init__(self, inner: JSONSchema):
        self.inner = inner

    def serialize(self) -> object:
        return {
            'type': 'optional',
            'payload': {
                'inner': self.inner.serialize()
            }
        }

    def validate(self, o: object):
        if o is not None:
            try:
                self.inner.validate(o)
            except SchemaValueMismatch as inner:
                raise SchemaValueMismatch(
                    self.serialize(), o, 'Validation failure in descendant value'
                ) from inner

class Null(_SchemaBase):

    def serialize(self) -> object:
        return {
            'type': 'null',
            'payload': {}
        }

    def validate(self, o: object):
        if o is not None:
            raise SchemaValueMismatch(self.serialize(), o, 'Expected null value')


class Any(_SchemaBase):

    def serialize(self) -> object:
        return {
            'type': 'any',
            'payload': {}
        }

    def validate(self, o: object):
        pass


Object = FixedObject | DynamicObject
IntLike = Integer | Boolean
Numeric = Integer | Float
Value = IntLike | Float | Boolean | String | Null
JSONSchema = FixedObject | Array | Value | Optional | Choice
