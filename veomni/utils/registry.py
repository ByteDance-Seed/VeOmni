from functools import lru_cache
from typing import Callable, List, MutableMapping, Optional, Type, Union

from packaging.specifiers import SpecifierSet
from packaging.version import Version


@lru_cache
def _get_transformers_version() -> Version:
    from veomni.utils.import_utils import _get_package_version

    return _get_package_version("transformers")


def _matches_transformers_version(spec_str: str) -> bool:
    return _get_transformers_version() in SpecifierSet(spec_str)


# Default version constraint for existing v4 registrations
_DEFAULT_TRANSFORMERS_VERSION = ">=4,<5"


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
        if key in self._local_mapping:
            return self._local_mapping[key]

        if key not in self._global_mapping:
            raise ValueError(f"Unknown {self._name} name: {key}. No {self._name} registered for this source.")

        entries = self._global_mapping[key]
        for version_spec, func in entries:
            if _matches_transformers_version(version_spec):
                return func

        raise ValueError(
            f"{self._name} '{key}' is registered but not compatible with "
            f"transformers {_get_transformers_version()}. "
            f"Available version ranges: {[e[0] for e in entries]}"
        )

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

    def register(
        self,
        key: str,
        cls_or_func: Optional[Union[Type, Callable]] = None,
        *,
        transformers_version: str = _DEFAULT_TRANSFORMERS_VERSION,
    ):
        """Register a class or function with a transformers version constraint.

        Args:
            key: The registry key to register under.
            cls_or_func: The class or function to register. If None, returns a decorator.
            transformers_version: Version specifier string, e.g. ">=4,<5" or ">=5,<6".
                Defaults to ">=4,<5" for backward compatibility.
        """
        if cls_or_func is not None:
            self._global_mapping.setdefault(key, [])
            self._global_mapping[key].append((transformers_version, cls_or_func))
            return cls_or_func

        def decorator(cls_or_func):
            entries = self._global_mapping.get(key, [])
            # Check for duplicate version range registration
            for existing_spec, _ in entries:
                if existing_spec == transformers_version:
                    raise ValueError(
                        f"{self._name} for '{key}' with transformers_version='{transformers_version}' "
                        f"is already registered. Cannot register duplicate {self._name}."
                    )
            self._global_mapping.setdefault(key, [])
            self._global_mapping[key].append((transformers_version, cls_or_func))
            return cls_or_func

        return decorator

    def valid_keys(self) -> List[str]:
        """Return keys that have at least one entry matching the current transformers version."""
        valid = set()
        for key in self._local_mapping:
            valid.add(key)
        for key, entries in self._global_mapping.items():
            for version_spec, _ in entries:
                if _matches_transformers_version(version_spec):
                    valid.add(key)
                    break
        return list(valid)
