#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = ["Markdown==3.7"]
# ///
"""Render blog/*.md beside their sources, or into --output-dir for publishing."""

import argparse
from datetime import date
from html import escape
from pathlib import Path
from urllib.parse import quote
from xml.etree.ElementTree import Element

try:
    import markdown
    from markdown.inlinepatterns import InlineProcessor
    from markdown.util import AtomicString
except ImportError:
    raise SystemExit("Install Markdown: python3 -m pip install Markdown==3.7\n"
                     "Or run: uv run build_blog.py")


ROOT = Path(__file__).resolve().parent
AUTHOR = "Hirokazu Ishida"
SITE_URL = "https://hiroishida.github.io"
STYLE = """
      * { box-sizing: border-box; }
      :root {
        color-scheme: light;
        --ink: #111;
        --muted: #66717c;
        --accent: #446884;
        --line: #dce2e8;
        --wash: #f3f5f7;
        --paper: #fdfdfd;
      }
      body {
        margin: 0;
        color: var(--ink);
        background: var(--paper);
        font: 400 16px/1.8 "Helvetica Neue", Helvetica, Arial, "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
        overflow-wrap: break-word;
      }
      .wrapper { max-width: 800px; margin: 0 auto; padding: 0 30px; }
      .site-header { border-top: 5px solid #424242; border-bottom: 1px solid #e8e8e8; }
      .site-header .wrapper { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding-top: 12px; padding-bottom: 12px; }
      .site-name { color: var(--ink); font-size: 20px; font-weight: 300; letter-spacing: -.5px; }
      .back-link { white-space: nowrap; font-size: 14px; }
      a { color: var(--accent); text-decoration: none; }
      a:hover { color: var(--ink); text-decoration: underline; }
      a:focus-visible, [tabindex]:focus-visible { outline: 2px solid var(--accent); outline-offset: 5px; }
      .site-name { text-decoration: none; }
      main.wrapper { padding-top: 30px; padding-bottom: 30px; }
      .post-header { margin-bottom: 26px; }
      h1 { margin: 0 0 8px; font-size: 28px; line-height: 1.5; font-weight: 400; }
      .metadata { display: flex; flex-wrap: wrap; gap: 4px 16px; color: var(--muted); font-size: 13px; }
      p, ul, ol { margin: 0 0 18px; }
      h2 { margin: 34px 0 16px; font-size: 23px; line-height: 1.5; font-weight: 400; }
      h3 { margin: 24px 0 8px; font-size: 17px; font-weight: 600; }
      .aside { color: var(--muted); font-size: 14px; }
      .citation { font-size: 12px; vertical-align: super; white-space: nowrap; }
      .equation { margin: 24px 0; padding: 12px 20px; border: 1px solid var(--line); background: var(--wash); }
      .math-scroll { overflow-x: auto; overflow-y: hidden; }
      .equation figcaption { color: var(--muted); text-align: right; font-size: 13px; }
      .equation mjx-container[display="true"] { margin: 12px 0 !important; }
      table { width: 100%; margin: 20px 0; border-collapse: collapse; font-size: 15px; }
      table caption { padding-bottom: 10px; text-align: left; color: var(--muted); font-size: 13px; }
      table th, table td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; }
      table thead { background: var(--wash); }
      table th { font-weight: 600; }
      table td:last-child { color: var(--muted); }
      .diagram { margin: 24px 0 28px; }
      .diagram-panel { padding: 12px 0; }
      .flow { display: grid; grid-template-columns: minmax(0, 1fr) 30px minmax(0, 1fr) 30px minmax(0, 1fr); align-items: stretch; }
      .flow-node { padding: 16px 14px; border: 1px solid var(--line); background: var(--wash); }
      .flow-node strong { display: block; margin-bottom: 8px; font-size: 17px; font-weight: 500; }
      .node-input { min-height: 44px; margin-bottom: 12px; font-size: 13px; line-height: 1.7; }
      .node-output { margin: 0; color: var(--muted); font-size: 13px; }
      .flow-arrow { display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 20px; }
      .arrow-mobile { display: none; }
      .feedback { width: 66%; margin: 0 0 0 auto; padding: 12px; border: 1px solid var(--line); border-top: 0; color: var(--muted); text-align: center; font-size: 12px; }
      .diagram figcaption { margin-top: 10px; color: var(--muted); font-size: 13px; line-height: 1.8; }
      .references { padding-left: 26px; font-size: 14px; }
      .references li { margin-bottom: 14px; padding-left: 5px; }
      .references li:target { background: var(--wash); outline: 5px solid var(--wash); }
      .references cite { font-style: normal; }
      .post-footer { margin-top: 38px; padding-top: 20px; border-top: 1px solid var(--line); font-size: 14px; }
      @media (max-width: 600px) {
        .wrapper { padding-right: 15px; padding-left: 15px; }
        .site-name { font-size: 17px; }
        h1 { font-size: 26px; }
        h2 { font-size: 21px; }
        .equation { padding: 10px; }
        table th, table td { padding: 10px 7px; font-size: 13px; }
        table td:first-child { width: 34%; }
        .flow { grid-template-columns: minmax(0, 1fr); }
        .flow-node { padding: 14px 18px; }
        .node-input { min-height: 0; margin-bottom: 8px; }
        .flow-arrow { height: 34px; }
        .arrow-desktop { display: none; }
        .arrow-mobile { display: inline; }
        .feedback { width: 90%; margin: 0 auto; }
      }
      @media print {
        .site-header, .post-footer { display: none; }
        main.wrapper { padding-top: 0; }
        .equation, .diagram, table { break-inside: avoid; }
      }"""


class MathJaxPattern(InlineProcessor):
    """Keep TeX delimiters and underscores intact, after processing code spans."""

    def handleMatch(self, match, data):
        element = Element("span")
        element.text = AtomicString(match.group(0))
        return element, match.start(0), match.end(0)


def render(source):
    md = markdown.Markdown(
        extensions=["meta", "tables", "fenced_code", "attr_list", "md_in_html"],
        output_format="html",
    )
    md.inlinePatterns.register(
        MathJaxPattern(r"(?s)\\\(.+?\\\)|\\\[.+?\\\]", md), "mathjax", 185
    )
    body = md.convert(source.read_text(encoding="utf-8"))
    metadata = {key: " ".join(value) for key, value in md.Meta.items()}
    for field in ("title", "date"):
        if not metadata.get(field):
            raise ValueError("missing metadata: " + field)
    published = date.fromisoformat(metadata["date"])
    title = escape(metadata["title"])
    author = escape(metadata.get("author", AUTHOR))
    description = escape(metadata.get("description", metadata["title"]))
    canonical = SITE_URL + "/blog/" + quote(source.with_suffix(".html").name)
    return f'''<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title} | {author}</title>
    <meta name="description" content="{description}">
    <link rel="canonical" href="{canonical}">
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@4.0.0/tex-svg.js"></script>
    <style>{STYLE}
    </style>
  </head>
  <body>
    <header class="site-header">
      <div class="wrapper">
        <a class="site-name" href="../index.html">{AUTHOR}</a>
        <a class="back-link" href="../index.html">← Home</a>
      </div>
    </header>
    <main class="wrapper">
      <article aria-labelledby="post-title">
        <header class="post-header">
          <h1 id="post-title">{title}</h1>
          <div class="metadata">
            <time datetime="{published.isoformat()}">{published.year}年{published.month}月{published.day}日</time>
            <span>{author}</span>
          </div>
        </header>
{body}
        <footer class="post-footer"><a href="../index.html">← トップページへ戻る</a></footer>
      </article>
    </main>
  </body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", type=Path, help="Markdown files (default: blog/*.md)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: next to each source)")
    args = parser.parse_args()
    sources = args.sources or sorted((ROOT / "blog").glob("*.md"))
    if not sources:
        parser.error("no Markdown posts found")
    for source in sources:
        if source.suffix != ".md":
            parser.error("expected a .md file: " + str(source))
        destination = source.with_suffix(".html")
        if args.output_dir:
            destination = args.output_dir / destination.name
        try:
            page = render(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(page, encoding="utf-8")
        except (OSError, ValueError) as error:
            parser.exit(1, f"{source}: {error}\n")
        print(f"{source} -> {destination}")


if __name__ == "__main__":
    main()
