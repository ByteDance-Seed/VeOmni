from typing import Callable, List, MutableMapping, Optional, Type, Union


class Registry(MutableMapping):
    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    registry = []

    def __init__(self, name: str):
        self._name = name
        self.registry.append(name)
        self._local_mapping = {}
        self._global_mapping = {}

    def __getitem__(self, key):
        # First check if instance has a local override
        if key not in self.valid_keys():
            raise ValueError(f"Unknown {self._name} name: {key}. No {self._name} registered for this source.")
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        # Allow local update of the default functions without impacting other instances
        self._local_mapping.update({key: value})

    def __delitem__(self, key):
        del self._local_mapping[key]

    def __iter__(self):
        # Ensure we use all keys, with the overwritten ones on top
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    def register(self, key: str, cls_or_func: Optional[Union[Type, Callable]] = None):
        if cls_or_func is not None:
            self._global_mapping[key] = cls_or_func
            return cls_or_func

        def decorator(cls_or_func):
            if key in self._global_mapping:
                raise ValueError(
                    f"{self._name} for '{key}' is already registered. Cannot register duplicate {self._name}."
                )
            self._global_mapping.update({key: cls_or_func})
            return cls_or_func

        return decorator

    def valid_keys(self) -> List[str]:
        return list(self.keys())
