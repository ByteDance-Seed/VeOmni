# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
