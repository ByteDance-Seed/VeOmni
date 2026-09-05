"""Behavioral regressions for the standalone documentation scanner; no VeOmni/GPU imports."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scan_docs.py"
SPEC = importlib.util.spec_from_file_location("veomni_doc_scanner", SCRIPT)
scanner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)


class ScannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="veomni-doc-scan-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def write(self, name, content=""):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def scan(self, content, name="guide.md"):
        path = self.write(name, content)
        engine = scanner.Scanner(self.root)
        return engine.scan([path]), engine

    def cli(self, *args, no_dependencies=False):
        command = [sys.executable]
        if no_dependencies:
            command.append("-S")
        return subprocess.run(
            [*command, str(SCRIPT), "--root", str(self.root), "--format", "json", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

    def test_empty_links_and_page_top(self):
        findings, _ = self.scan('[page]() [spaced]( ) [angle](<>) [title](<> "title") [top](#)')
        self.assertEqual(findings, [])

    def test_bad_url_does_not_interrupt_other_files_or_json(self):
        self.write("bad.md", "[external](http://[invalid)\n")
        self.write("good.md", "[page]()\n[missing](missing.md)\n")
        result = self.cli("bad.md", "good.md")
        report = json.loads(result.stdout)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["files_scanned"], 2)
        self.assertEqual({item["rule"] for item in report["findings"]}, {"invalid-url", "broken-local-link"})
        self.assertNotIn("Traceback", result.stderr)

    def test_footnote_prose_and_real_link(self):
        findings, _ = self.scan("Text[^1].\n\n[^1]: This is a note. See [guide](missing.md).\n")
        self.assertEqual([(item.rule, item.line) for item in findings], [("broken-local-link", 3)])
        self.assertIn("missing.md", findings[0].message)

    def test_reference_link_is_checked_at_use(self):
        self.write("target.md")
        findings, _ = self.scan("[ok][target] [broken][missing]\n\n[target]: target.md\n[missing]: absent.md\n")
        self.assertEqual([(item.rule, item.line) for item in findings], [("broken-local-link", 1)])

    def test_literals_and_comments_do_not_create_links_or_anchors(self):
        cases = [
            "    [literal](missing.md)\n",
            "> ```text\n> [literal](missing.md)\n> ```\n",
            "- Example:\n\n  ```text\n  [literal](missing.md)\n  ```\n",
            "<!--\n[hidden](missing.md)\n# Hidden\n-->\n",
            "`[literal](missing.md)`\n",
            "``one ` [literal](missing.md)``\n",
            "<pre><a href='missing.md'>literal</a></pre>\n",
            "<code>[literal](missing.md)</code>\n",
        ]
        for content in cases:
            with self.subTest(content=content):
                findings, engine = self.scan(content)
                self.assertEqual(findings, [])
                self.assertNotIn("hidden", engine.document(self.root / "guide.md").anchors)

    def test_fences_with_language_text_inside_are_not_closers(self):
        findings, _ = self.scan("```markdown\n```python\n[literal](missing.md)\n```\n")
        self.assertEqual(findings, [])

    def test_unclosed_fences(self):
        cases = ["```python\ncode\n", "```python\ncode\n``` trailing\n", "> ```text\n> code\n"]
        for content in cases:
            with self.subTest(content=content):
                findings, _ = self.scan(content)
                self.assertEqual([item.rule for item in findings], ["unclosed-fence"])

    def test_empty_and_longer_closed_fences(self):
        for content in ["```\n```\n", "~~~text\ncode\n~~~~\n", "   ```\n\n   ```\n"]:
            with self.subTest(content=content):
                self.assertEqual(self.scan(content)[0], [])

    def test_unicode_separators_do_not_change_fence_line_count(self):
        for separator in ["\u2028", "\u2029", "\x85", "\v"]:
            with self.subTest(separator=repr(separator)):
                self.assertEqual(self.scan(f"```text\na{separator}b\n```\n")[0], [])
                findings, _ = self.scan(f"```text\na{separator}b\n")
                self.assertEqual([item.rule for item in findings], ["unclosed-fence"])

    def test_rendered_headings_and_emoji_anchors(self):
        cases = [
            ("# _Overview_", "overview"),
            ("## [Overview](target.md)", "overview"),
            ("Overview\n========", "overview"),
            ("   ## Overview", "overview"),
            ("## 📚 Overview", "-overview"),
            ("## 🛠️ Add Your Own Model", "-add-your-own-model"),
            ("## `some_name`", "some_name"),
            ("## A  B", "a--b"),
            ("## 中文标题", "中文标题"),
            ("## ~~Old~~ New", "old-new"),
            ("## A &amp; B", "a--b"),
        ]
        self.write("target.md")
        for heading, anchor in cases:
            with self.subTest(heading=heading):
                findings, _ = self.scan(f"{heading}\n\n[go](#{anchor})\n")
                self.assertEqual(findings, [])

    def test_global_heading_id_collisions(self):
        cases = [
            ("# foo\n# foo\n# foo-1\n", {"foo", "foo-1", "foo-1-1"}),
            ("# foo\n# foo-1\n# foo\n", {"foo", "foo-1", "foo-2"}),
        ]
        for content, expected in cases:
            with self.subTest(content=content):
                _, engine = self.scan(content)
                self.assertEqual(engine.document(self.root / "guide.md").anchors, expected)

    def test_missing_anchor_remains_warning(self):
        findings, _ = self.scan("# Visible\n\n[bad](#absent)\n")
        self.assertEqual(
            [(item.rule, item.severity, item.line) for item in findings], [("missing-anchor", "warning", 3)]
        )

    def test_html_images_links_ids_and_multiline_attributes(self):
        content = "<div id='section'>\n<img\n src='missing.png'>\n<a class='link' href='missing.md'>Guide</a>\n</div>\n\n[section](#section)\n"
        findings, _ = self.scan(content)
        self.assertEqual(
            [(item.rule, item.line) for item in findings], [("broken-local-link", 2), ("broken-local-link", 4)]
        )

    def test_html_entities_and_attribute_order(self):
        self.write("a&b.png")
        findings, _ = self.scan(
            '<img alt="ok" src="a&amp;b.png"/>\n\n<a class="anchor" name="custom"></a>\n\n[go](#custom)'
        )
        self.assertEqual(findings, [])

    def test_html_comments_do_not_check_missing_image(self):
        self.assertEqual(self.scan('<!-- <img src="missing.png"> -->')[0], [])

    def test_literal_html_element_keeps_its_own_anchor(self):
        for tag in ["pre", "code"]:
            with self.subTest(tag=tag):
                findings, _ = self.scan(
                    f'<{tag} id="example"><a href="missing.md">literal</a></{tag}>\n\n[go](#example)\n'
                )
                self.assertEqual(findings, [])

    def test_html_code_is_excluded_from_prose_but_following_text_is_checked(self):
        for literal in ["<code>word word</code>", "<code>word\nword</code>"]:
            with self.subTest(literal=literal):
                findings, _ = self.scan(f"{literal}\n\nThis this repeats.\n")
                self.assertEqual(
                    [(item.rule, item.line) for item in findings], [("duplicate-word", literal.count("\n") + 3)]
                )
        self.assertEqual(self.scan("word <code>word</code> word")[0], [])
        findings, _ = self.scan("word <em>word</em>")
        self.assertEqual([item.rule for item in findings], ["duplicate-word"])

    def test_case_mismatch_on_files_and_directories(self):
        self.write("Assets/README.md")
        findings, _ = self.scan("[guide](assets/readme.md)")
        self.assertEqual([item.rule for item in findings], ["path-case-mismatch"])
        self.assertIn("Assets/README.md", findings[0].message)
        self.assertEqual(self.scan("[guide](Assets/README.md)")[0], [])

    def test_balanced_escaped_and_encoded_paths(self):
        self.write("file(v1).md")
        self.write("two words.md")
        content = '[a](file(v1).md) [b](file\\(v1\\).md) [c](file%28v1%29.md) [d](<two words.md> "title")'
        self.assertEqual(self.scan(content)[0], [])

    def test_parent_root_and_query_paths(self):
        self.write("README.md", "# Overview\n")
        findings, _ = self.scan("[a](../README.md#overview) [b](/README.md?x=1#overview)", "docs/guide.md")
        self.assertEqual(findings, [])

    def test_external_and_outside_targets_are_recorded_not_fetched(self):
        findings, engine = self.scan(
            "[remote](https://example.invalid/) [outside](../outside.md) [template]({path}.md)"
        )
        self.assertEqual(findings, [])
        self.assertEqual(dict(engine.skipped), {"external-url": 1, "outside-root": 1, "template-target": 1})

    def test_myst_requires_separate_verification(self):
        content = "(label)=\n# Title\n\n[label](#label)\n\n```{note}\n[inside](missing.md)\n```\n"
        findings, engine = self.scan(content)
        self.assertEqual(findings, [])
        self.assertEqual(dict(engine.skipped), {"myst-directive": 1, "myst-target": 1})

    def test_duplicate_word_line_and_code_exclusion(self):
        findings, _ = self.scan("# Title\n\nThis this is repeated.\n\n`code code`\n")
        self.assertEqual([(item.rule, item.line) for item in findings], [("duplicate-word", 3)])

    def test_heading_style_is_candidate_not_syntax_error(self):
        findings, _ = self.scan("#MissingSpace\n")
        self.assertEqual([(item.rule, item.severity) for item in findings], [("heading-space", "warning")])

    def test_unreadable_target_is_not_a_missing_anchor(self):
        target = self.write("binary.md")
        target.write_bytes(b"\xff")
        findings, _ = self.scan("[go](binary.md#heading)")
        self.assertEqual([item.rule for item in findings], ["unverified-anchor"])

    def test_encoding_error_does_not_drop_good_files(self):
        self.write("binary.md").write_bytes(b"\xff")
        self.write("good.md", "# Good\n")
        result = self.cli("binary.md", "good.md")
        report = json.loads(result.stdout)
        self.assertEqual(report["files_scanned"], 2)
        self.assertEqual([item["rule"] for item in report["findings"]], ["encoding"])

    def test_cli_exit_codes_and_coverage(self):
        self.write("good.md", "# Good\n")
        self.write("warning.md", "This this repeats.\n")
        self.write("error.md", "[missing](missing.md)\n")
        for args, expected in [
            (("good.md",), 0),
            (("warning.md",), 0),
            (("--fail-on", "warning", "warning.md"), 1),
            (("error.md",), 1),
            (("--fail-on", "never", "error.md"), 0),
        ]:
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertIn("not_checked", json.loads(result.stdout)["coverage"])

    def test_cli_invalid_scope_and_missing_dependencies(self):
        self.write("good.md", "# Good\n")
        self.write("not-markdown.txt")
        for args in [("absent.md",), ("not-markdown.txt",), ("../outside.md",)]:
            result = self.cli(*args)
            self.assertEqual(result.returncode, 2)
            self.assertIn("error", json.loads(result.stdout))
        result = self.cli("good.md", no_dependencies=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("pip install", json.loads(result.stdout)["error"])

    def test_no_git_fallback_and_empty_scope(self):
        self.write("guide.md")
        self.write(".venv/ignored.md")
        with patch.object(scanner.subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual(scanner.repository_markdown(self.root), [self.root / "guide.md"])
        (self.root / "empty").mkdir()
        with self.assertRaises(ValueError):
            scanner.selected_markdown(self.root / "empty", [])

    @unittest.skipUnless(shutil.which("git"), "Git is unavailable")
    def test_git_scope_excludes_ignored_and_deleted_files(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        self.write(".gitignore", "ignored/\n")
        deleted = self.write("deleted.md")
        subprocess.run(["git", "-C", str(self.root), "add", "deleted.md"], check=True, capture_output=True)
        deleted.unlink()
        self.write("ignored/no.md")
        visible = self.write("docs/new.md")
        self.assertEqual(scanner.selected_markdown(self.root, []), [visible])
        self.assertEqual(scanner.selected_markdown(self.root, ["."]), [visible])


if __name__ == "__main__":
    unittest.main()
