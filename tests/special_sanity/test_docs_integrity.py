"""CPU-only regression tests for the documentation gate (run by check_docs.yml)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from markdown_it import MarkdownIt


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/ci/check_docs_integrity.py"
_SPEC = importlib.util.spec_from_file_location("check_docs_integrity", _SCRIPT)
docs_check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(docs_check)


class DocumentationIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_markdown_reference_links_checked_but_code_examples_ignored(self):
        path = self.write("README.md", "")
        self.write("configs/train.yaml", "model: {}")
        text = """[Recipe][recipe]

[recipe]: configs/train.yaml

```markdown
[Illustration](does-not-exist.md)
```
"""
        page = docs_check.parse_html(MarkdownIt().render(text))
        self.assertEqual(docs_check.check_links(path, page, self.root, check_fragments=False, cache={}), [])
        (self.root / "configs/train.yaml").unlink()
        errors = docs_check.check_links(path, page, self.root, check_fragments=False, cache={})
        self.assertEqual(len(errors), 1)
        self.assertIn("configs/train.yaml", errors[0])

    def test_rendered_links_decode_paths_and_require_existing_anchors(self):
        target = self.write("guide page.html", '<h1 id="setup">Setup</h1>')
        self.write("assets/logo.svg", "<svg/>")
        path = self.write("nested/index.html", "")
        page = docs_check.parse_html(
            '<a href="../guide%20page.html?view=full#setup">Guide</a>'
            '<img src="/assets/logo.svg">'
            '<a href="https://example.invalid/missing#anchor">External</a>'
        )
        self.assertEqual(docs_check.check_links(path, page, self.root, check_fragments=True, cache={}), [])
        target.write_text('<h1 id="renamed">Setup</h1>', encoding="utf-8")
        errors = docs_check.check_links(path, page, self.root, check_fragments=True, cache={})
        self.assertEqual(len(errors), 1)
        self.assertIn("missing HTML anchor", errors[0])

    def test_missing_image_and_path_outside_site_fail(self):
        path = self.write("index.html", '<img src="gone.svg"><a href="../outside.html">Outside</a>')
        errors = docs_check.check_links(
            path, docs_check.parse_html(path.read_text()), self.root, check_fragments=True, cache={}
        )
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("missing local target" in error for error in errors))
        self.assertTrue(any("escapes checked tree" in error for error in errors))

    def test_navigation_rejects_disconnected_cycle_but_allows_explicit_orphan(self):
        found = {"index", "guide", "cycle-a", "cycle-b", "old-url"}
        includes = {"index": ["guide"], "guide": ["index"], "cycle-a": ["cycle-b"], "cycle-b": ["cycle-a"]}
        errors = docs_check.check_navigation(found, "index", includes, {"old-url"})
        self.assertEqual(len(errors), 2)
        self.assertTrue(errors[0].startswith("cycle-a:"))
        self.assertTrue(errors[1].startswith("cycle-b:"))

    def test_missing_build_is_a_failure(self):
        for name in ("README.md", "CONTRIBUTING.md", "docs/README.md"):
            self.write(name, "# Documentation")
        errors = docs_check.check(self.root, self.root / "docs/_build/html", self.root / "missing.pickle")
        self.assertEqual(len(errors), 1)
        self.assertIn("Build the documentation first", errors[0])


if __name__ == "__main__":
    unittest.main()
