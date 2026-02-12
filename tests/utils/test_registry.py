from unittest.mock import patch

import pytest
from packaging.version import Version

from veomni.utils.registry import _DEFAULT_TRANSFORMERS_VERSION, Registry


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """Clear the _get_transformers_version lru_cache before each test."""
    from veomni.utils.registry import _get_transformers_version

    _get_transformers_version.cache_clear()
    yield
    _get_transformers_version.cache_clear()


def _mock_version(version_str):
    """Return a patch context that mocks _get_transformers_version to return the given version."""
    return patch(
        "veomni.utils.registry._get_transformers_version",
        return_value=Version(version_str),
    )


class TestVersionAwareRegistry:
    def test_default_version_constraint_is_v4(self):
        """Registrations without explicit version default to >=4,<5."""
        reg = Registry("test")

        @reg.register("foo")
        def foo_func():
            return "foo_v4"

        entries = reg._global_mapping["foo"]
        assert len(entries) == 1
        assert entries[0][0] == _DEFAULT_TRANSFORMERS_VERSION

    def test_lookup_matches_v4(self):
        """With transformers 4.x installed, v4 registration is returned."""
        reg = Registry("test")

        @reg.register("foo")
        def foo_v4():
            return "v4"

        @reg.register("foo", transformers_version=">=5,<6")
        def foo_v5():
            return "v5"

        with _mock_version("4.57.0"):
            assert reg["foo"]() == "v4"

    def test_lookup_matches_v5(self):
        """With transformers 5.x installed, v5 registration is returned."""
        reg = Registry("test")

        @reg.register("foo")
        def foo_v4():
            return "v4"

        @reg.register("foo", transformers_version=">=5,<6")
        def foo_v5():
            return "v5"

        with _mock_version("5.1.0"):
            assert reg["foo"]() == "v5"

    def test_lookup_raises_on_version_mismatch(self):
        """If only v4 is registered but v5 is installed, a clear error is raised."""
        reg = Registry("test")

        @reg.register("bar")
        def bar_v4():
            return "v4"

        with _mock_version("5.0.0"):
            with pytest.raises(ValueError, match="not compatible with transformers 5.0.0"):
                reg["bar"]

    def test_lookup_raises_on_unknown_key(self):
        """Looking up an unregistered key raises ValueError."""
        reg = Registry("test")
        with _mock_version("4.57.0"):
            with pytest.raises(ValueError, match="Unknown test name"):
                reg["nonexistent"]

    def test_duplicate_version_registration_raises(self):
        """Registering the same key with the same version range raises ValueError."""
        reg = Registry("test")

        @reg.register("dup")
        def dup1():
            return "first"

        with pytest.raises(ValueError, match="already registered"):

            @reg.register("dup")
            def dup2():
                return "second"

    def test_valid_keys_filters_by_version(self):
        """valid_keys() only returns keys compatible with the current transformers version."""
        reg = Registry("test")

        @reg.register("v4_only")
        def v4_only():
            pass

        @reg.register("v5_only", transformers_version=">=5,<6")
        def v5_only():
            pass

        @reg.register("both_v4")
        def both_v4():
            pass

        @reg.register("both_v4", transformers_version=">=5,<6")
        def both_v5():
            pass

        with _mock_version("4.57.0"):
            keys = reg.valid_keys()
            assert "v4_only" in keys
            assert "v5_only" not in keys
            assert "both_v4" in keys

        with _mock_version("5.1.0"):
            keys = reg.valid_keys()
            assert "v4_only" not in keys
            assert "v5_only" in keys
            assert "both_v4" in keys

    def test_local_override_bypasses_version_check(self):
        """Local overrides (via __setitem__) bypass version matching."""
        reg = Registry("test")

        @reg.register("foo")
        def foo_v4():
            return "v4"

        def local_foo():
            return "local"

        reg["foo"] = local_foo

        with _mock_version("5.0.0"):
            assert reg["foo"]() == "local"

    def test_direct_registration_without_decorator(self):
        """register(key, func) direct call works with version constraint."""
        reg = Registry("test")

        def my_func():
            return "direct"

        reg.register("direct_key", my_func, transformers_version=">=5,<6")

        with _mock_version("5.2.0"):
            assert reg["direct_key"]() == "direct"

        with _mock_version("4.57.0"):
            with pytest.raises(ValueError, match="not compatible"):
                reg["direct_key"]

    def test_iter_includes_all_keys(self):
        """__iter__ includes all global keys regardless of version."""
        reg = Registry("test")

        @reg.register("a")
        def a():
            pass

        @reg.register("b", transformers_version=">=5,<6")
        def b():
            pass

        with _mock_version("4.57.0"):
            all_keys = list(reg)
            assert "a" in all_keys
            assert "b" in all_keys

    def test_len_counts_all_keys(self):
        """__len__ counts all keys regardless of version."""
        reg = Registry("test")

        @reg.register("a")
        def a():
            pass

        @reg.register("b", transformers_version=">=5,<6")
        def b():
            pass

        assert len(reg) == 2
