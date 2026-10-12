import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import build_post


METADATA = "---\ntitle: Example\ndate: 2026-10-13\n---\n\n"
START = "<!-- build:ignore:start -->"
END = "<!-- build:ignore:end -->"


class BuildIgnoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "example.md"

    def read(self, body):
        self.source.write_text(METADATA + body, encoding="utf-8")
        return build_post.read_post(self.source)

    def test_notes_do_not_change_published_article_or_index(self):
        before = "Public [reference][ref].\n\n"
        after = "[ref]: https://example.com/public\n\n## Public ending\n\nDone.\n"
        metadata, body = self.read(before + after)
        expected_article = build_post.render_post(self.source, metadata, body)
        expected_index = build_post.render_index([("example.html", metadata)])
        notes = (
            f"{START}\n## 次の記事にいれる\n\nINTERNAL_NOTE\n\n"
            "<script>window.INTERNAL_NOTE = true;</script>\n\n"
            "```python\nINTERNAL_CODE = True\n```\n\n"
            "[ref]: https://example.com/internal\n"
            f"{END}\n\n{START}\nSECOND_NOTE\n{END}\n\n"
        )
        self.read(before + notes + after)
        output = Path(self.temporary.name) / "output"
        result = subprocess.run(
            [sys.executable, str(build_post.ROOT / "build_post.py"),
             str(self.source), "--output-dir", str(output)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((output / "example.html").read_text(), expected_article)
        self.assertEqual((output / "index.html").read_text(), expected_index)

    def test_markers_in_code_examples_remain_literal(self):
        for fence in ("```", "~~~~"):
            with self.subTest(fence=fence):
                _, body = self.read(
                    f"{fence}markdown\n{START}\nEXAMPLE_NOTE\n{END}\n{fence}\n"
                )
                self.assertIn("<pre><code", body)
                self.assertIn("&lt;!-- build:ignore:start --&gt;", body)
                self.assertIn("EXAMPLE_NOTE", body)
                self.assertIn("&lt;!-- build:ignore:end --&gt;", body)
        _, body = self.read(f"    {START}\n    INDENTED_EXAMPLE\n    {END}\n")
        self.assertIn("INDENTED_EXAMPLE", body)
        self.assertIn("&lt;!-- build:ignore:start --&gt;", body)

    def test_removal_keeps_surrounding_paragraphs_separate(self):
        _, body = self.read(f"Before.\n{START}\nNOTE\n{END}\nAfter.\n")
        self.assertEqual(body, "<p>Before.</p>\n<p>After.</p>")

    def test_malformed_notes_stop_build_before_any_output_is_written(self):
        output = Path(self.temporary.name) / "output"
        output.mkdir()
        previous = output / "example.html"
        previous.write_text("previous successful build")
        valid = Path(self.temporary.name) / "valid.md"
        valid.write_text(METADATA + "Public content.\n")
        for notes in (f"{START}\nNOTE", END, f"{START}\n{START}\n{END}"):
            with self.subTest(notes=notes):
                self.source.write_text(METADATA + notes)
                result = subprocess.run(
                    [sys.executable, str(build_post.ROOT / "build_post.py"),
                     str(valid), str(self.source), "--output-dir", str(output)],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("build:ignore", result.stderr)
                self.assertIn(str(self.source), result.stderr)
                self.assertEqual(previous.read_text(), "previous successful build")
                self.assertFalse((output / "valid.html").exists())
                self.assertFalse((output / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
