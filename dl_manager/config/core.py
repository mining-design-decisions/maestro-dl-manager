from __future__ import annotations


class NoSuchSetting(LookupError):
    def __init__(self, attribute, action):
        message = (
            f"Cannot perform action {action!r} on setting "
            f"{attribute!r} since it does not exist"
        )
        super().__init__(message)


class NotSet(Exception):
    def __init__(self, attribute):
        message = f"Attribute {attribute!r} has not been initialized"
        super().__init__(message)


class IllegalNamespace(Exception):
    def __init__(self, attribute):
        message = f"The namespace containing {attribute!r} is currently not accessible"
        super().__init__(message)


class ConfigFactory:
    def __init__(self):
        self._namespace = {}

    @staticmethod
    def _normalize_name(x: str) -> str:
        return x.lower().replace("-", "_")

    def register_namespace(self, name: str):
        if not name:
            raise ValueError("Name must be non-empty")
        parts = self._normalize_name(name).split(".")
        current = self._namespace
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
            if current is None:
                raise ValueError(
                    f"{name} is a property, and cannot be (part of) a namespace"
                )

    def register(self, name: str):
        if not name:
            raise ValueError("Name must be non-empty")
        parts = self._normalize_name(name).split(".")
        if len(parts) < 2:
            raise ValueError(
                f"A property must be contained in the non-global namespace ({name})"
            )
        current = self._namespace
        for part in parts[:-1]:
            if part not in current:
                raise ValueError(
                    f"Cannot register property {name}; namespace does not exist"
                )
            current = current[part]
        current[parts[-1]] = None

    def build_config(self, *namespaces) -> Config:
        legal = [self._normalize_name(n) for n in namespaces]
        for n in legal:
            if "." in n:
                raise ValueError(
                    f"Can only register top-level namespaces as legal, not {n}"
                )
        return Config(legal, self._namespace, self._new_namespace_tree(self._namespace))

    def _new_namespace_tree(self, obj):
        if obj is None:
            return Config.NOT_SET
        return {key: self._new_namespace_tree(value) for key, value in obj.items()}


class Config:
    NOT_SET = object()

    def __init__(self, legal_namespaces, namespaces, data):
        self._legal = legal_namespaces
        self._namespaces = namespaces
        self._data = data

    @staticmethod
    def _normalize_name(x):
        return x.lower().replace("-", "_").split(".")

    def _resolve(self, name, action, path):
        if path[0] not in self._legal:
            raise IllegalNamespace(name)
        current_n = self._namespaces
        current_d = self._data
        for part in path:
            if current_n is None:
                raise NoSuchSetting(name, action)
            if part not in current_n:
                raise NoSuchSetting(name, action)
            current_n = current_n[part]
            current_d = current_d[part]
        return current_d

    def get_all(self, name: str):
        return {
            key: (value if value is not self.NOT_SET else None)
            for key, value in self._resolve(
                name, "get_all", self._normalize_name(name)
            ).items()
        }

    def get(self, name: str):
        *path, prop = self._normalize_name(name)
        namespace = self._resolve(name, "get", path)
        if prop not in namespace:
            raise NoSuchSetting(name, "get")
        value = namespace[prop]
        if value is self.NOT_SET:
            raise NotSet(name)
        return value

    def set(self, name: str, value):
        *path, prop = self._normalize_name(name)
        namespace = self._resolve(name, "set", path)
        if prop not in namespace:
            raise NoSuchSetting(name, "set")
        namespace[prop] = value

    def clone(self, from_: str, to: str):
        self.set(to, self.get(from_))

    def transfer(self, target: Config, *properties):
        for prop in properties:
            target.set(prop, self.get(prop))

    def update(self, prefix: str | None = None, /, **items):
        for key, value in items.items():
            if prefix is not None:
                key = f"{prefix}.{key}"
            self.set(key, value)
