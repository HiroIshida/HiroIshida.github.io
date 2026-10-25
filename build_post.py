#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = ["Markdown==3.7", "Pygments==2.18.0"]
# ///
"""Render post/*.md and an article index, optionally into --output-dir."""

import argparse
from datetime import date
from html import escape
from pathlib import Path
from urllib.parse import quote
from xml.etree.ElementTree import Element

try:
    import markdown
    from markdown.inlinepatterns import InlineProcessor
    from markdown.preprocessors import Preprocessor
    from markdown.util import AtomicString
    from pygments.formatters import HtmlFormatter
except ImportError:
    raise SystemExit("Install dependencies: python3 -m pip install Markdown==3.7 Pygments==2.18.0\n"
                     "Or run: uv run build_post.py")


ROOT = Path(__file__).resolve().parent
AUTHOR = "Hirokazu Ishida"
SITE_URL = "https://hiroishida.github.io"
HIGHLIGHT_STYLE = HtmlFormatter(style="friendly", nobackground=True).get_style_defs(".codehilite")
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
      .post-nav { position: sticky; top: 0; z-index: 10; margin: 0 0 24px; padding: 12px 0; border-bottom: 1px solid var(--line); background: var(--paper); }
      .post-nav-title { display: none; }
      .post-nav ul { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 0; padding: 0; list-style: none; }
      .post-nav a { display: block; padding: 8px 12px; border-left: 2px solid var(--line); font-size: 14px; }
      .post-nav a:hover, .post-nav a:focus-visible { border-left-color: var(--accent); }
      .post-section, .post-nav ~ .references li { scroll-margin-top: 120px; }
      h1 { margin: 0 0 8px; font-size: 28px; line-height: 1.5; font-weight: 400; }
      .metadata { display: flex; flex-wrap: wrap; gap: 4px 16px; color: var(--muted); font-size: 13px; }
      .post-list { margin: 24px 0; padding: 0; list-style: none; border-top: 1px solid var(--line); }
      .post-list li { display: grid; grid-template-columns: 100px minmax(0, 1fr); gap: 16px; padding: 14px 0; border-bottom: 1px solid var(--line); }
      .post-list time { color: var(--muted); font-size: 14px; }
      p, ul, ol { margin: 0 0 18px; }
      h2 { margin: 34px 0 16px; font-size: 23px; line-height: 1.5; font-weight: 400; }
      h3 { margin: 24px 0 8px; font-size: 17px; font-weight: 600; }
      pre { margin: 20px 0; padding: 12px 16px; overflow-x: auto; border: 1px solid var(--line); background: var(--wash); font-size: 13px; line-height: 1.7; }
      pre code { overflow-wrap: normal; }
      .codehilite .c, .codehilite .ch, .codehilite .c1, .codehilite .cm, .codehilite .cs { color: var(--muted); font-style: normal; }
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
      .simt-panel { padding: 18px; border: 1px solid var(--line); }
      .simt-heading { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 4px 16px; margin-bottom: 12px; font-size: 13px; }
      .simt-legend { color: var(--muted); font-size: 12px; }
      .simt-timeline { table-layout: fixed; border-collapse: separate; border-spacing: 5px 8px; margin: 0; }
      .simt-timeline thead { background: none; }
      .simt-timeline th, .simt-timeline td { padding: 0; border: 0; text-align: center; font-size: 13px; font-weight: 400; }
      .simt-timeline th:first-child { width: 66px; }
      .simt-slot { display: flex; align-items: center; justify-content: center; min-height: 42px; border: 1px solid var(--line); color: var(--muted); background: var(--wash); }
      .simt-slot.simt-a { color: #315e91; background: #dceafb; border-color: #92b4de; }
      .simt-slot.simt-b { color: #97591f; background: #ffe9d4; border-color: #dbab7e; }
      .simt-slot.simt-c { color: #326d59; background: #d9eee4; border-color: #8cbda6; }
      .simt-note { margin: 10px 0 0; color: var(--muted); font-size: 12px; }
      .references { padding-left: 26px; font-size: 14px; }
      .references li { margin-bottom: 14px; padding-left: 5px; }
      .references li:target { background: var(--wash); outline: 5px solid var(--wash); }
      .references cite { font-style: normal; }
      .post-footer { margin-top: 38px; padding-top: 20px; border-top: 1px solid var(--line); font-size: 14px; }
      @media (min-width: 1200px) {
        .post-nav { position: fixed; top: 140px; left: max(20px, calc((100vw - 800px) / 2 - 200px)); width: 180px; max-height: calc(100vh - 160px); overflow-y: auto; margin: 0; padding: 0; border: 0; }
        .post-nav-title { display: block; margin-bottom: 12px; color: var(--muted); font-size: 12px; }
        .post-nav ul { display: block; }
        .post-nav li + li { margin-top: 6px; }
        .post-section, .post-nav ~ .references li { scroll-margin-top: 24px; }
      }
      @media (max-width: 600px) {
        .wrapper { padding-right: 15px; padding-left: 15px; }
        .site-name { font-size: 17px; }
        h1 { font-size: 26px; }
        h2 { font-size: 21px; }
        .post-list li { grid-template-columns: minmax(0, 1fr); gap: 2px; }
        .post-nav ul { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px 8px; }
        .post-nav a { min-height: 44px; padding: 9px 8px; font-size: 13px; }
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
        .simt-panel { padding: 10px; }
        .simt-timeline { border-spacing: 3px 6px; }
        .simt-timeline th, .simt-timeline td { font-size: 11px; }
        .simt-timeline th:first-child { width: 44px; }
        .simt-slot { min-height: 36px; }
      }
      @media print {
        .site-header, .post-footer, .post-nav { display: none; }
        main.wrapper { padding-top: 0; }
        .equation, .diagram, table { break-inside: avoid; }
      }"""


class MathJaxPattern(InlineProcessor):
    """Keep TeX delimiters and underscores intact, after processing code spans."""

    def handleMatch(self, match, data):
        element = Element("span")
        element.text = AtomicString(match.group(0))
        return element, match.start(0), match.end(0)


class BuildIgnorePreprocessor(Preprocessor):
    """Remove author notes entirely, while preserving fenced code examples."""

    def run(self, lines):
        visible = []
        ignoring = False
        for line in lines:
            marker = line.rstrip()
            if marker == "<!-- build:ignore:start -->":
                if ignoring:
                    raise ValueError("nested build:ignore:start marker")
                ignoring = True
                visible.append("")
            elif marker == "<!-- build:ignore:end -->":
                if not ignoring:
                    raise ValueError("build:ignore:end without a start marker")
                ignoring = False
            elif not ignoring:
                visible.append(line)
        if ignoring:
            raise ValueError("unclosed build:ignore:start (missing build:ignore:end)")
        return visible


def read_post(source):
    md = markdown.Markdown(
        extensions=["meta", "tables", "fenced_code", "codehilite", "attr_list", "md_in_html"],
        extension_configs={"codehilite": {"guess_lang": False, "pygments_style": "friendly"}},
        output_format="html",
    )
    # Fenced code is stashed at priority 25; raw HTML is handled at priority 20.
    md.preprocessors.register(BuildIgnorePreprocessor(md), "build_ignore", 24)
    md.inlinePatterns.register(
        MathJaxPattern(r"(?s)\\\(.+?\\\)|\\\[.+?\\\]", md), "mathjax", 185
    )
    body = md.convert(source.read_text(encoding="utf-8"))
    metadata = {key: " ".join(value) for key, value in md.Meta.items()}
    for field in ("title", "date"):
        if not metadata.get(field):
            raise ValueError("missing metadata: " + field)
    metadata["date"] = date.fromisoformat(metadata["date"])
    return metadata, body


def page(title, description, canonical, body, navigation, mathjax=False):
    script = '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@4.0.0/tex-svg.js"></script>' if mathjax else ""
    return f'''<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{escape(title)} | {AUTHOR}</title>
    <meta name="description" content="{escape(description)}">
    <link rel="canonical" href="{escape(SITE_URL + canonical)}">
    {script}
    <style>{HIGHLIGHT_STYLE}
    {STYLE}
    </style>
  </head>
  <body>
    <header class="site-header">
      <div class="wrapper">
        <a class="site-name" href="../index.html">{AUTHOR}</a>
        {navigation}
      </div>
    </header>
    <main class="wrapper">
{body}
    </main>
  </body>
</html>
'''


def render_post(source, metadata, body):
    published = metadata["date"]
    content = f'''
      <article aria-labelledby="post-title">
        <header class="post-header">
          <h1 id="post-title">{escape(metadata["title"])}</h1>
          <div class="metadata">
            <time datetime="{published.isoformat()}">{published.year}年{published.month}月{published.day}日</time>
            <span>{escape(metadata.get("author", AUTHOR))}</span>
          </div>
        </header>
{body}
        <footer class="post-footer"><a href="index.html">← 記事一覧へ戻る</a></footer>
      </article>
'''
    return page(
        metadata["title"], metadata.get("description", metadata["title"]),
        "/post/" + quote(source.with_suffix(".html").name), content,
        '<a class="back-link" href="index.html">← Posts</a>', mathjax=True,
    )


def render_index(posts):
    items = []
    for filename, metadata in sorted(posts, key=lambda post: (-post[1]["date"].toordinal(), post[0])):
        published = metadata["date"].isoformat()
        items.append(
            f'<li><time datetime="{published}">{published}</time>'
            f'<a href="{quote(filename)}">{escape(metadata["title"])}</a></li>'
        )
    body = '<h1>Posts</h1>\n<ul class="post-list">\n' + "\n".join(items) + '\n</ul>'
    return page(
        "Posts", "Hirokazu Ishidaの記事一覧。", "/post/", body,
        '<a class="back-link" href="../index.html">← Home</a>',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", type=Path, help="Markdown files (default: post/*.md)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: next to each source)")
    args = parser.parse_args()
    sources = args.sources or sorted((ROOT / "post").glob("*.md"))
    if not sources:
        parser.error("no Markdown posts found")
    outputs = {}
    indexes = {}
    for source in sources:
        if source.suffix != ".md":
            parser.error("expected a .md file: " + str(source))
        if source.stem == "index":
            parser.error("index.md is reserved for the generated article index")
        destination = source.with_suffix(".html")
        if args.output_dir:
            destination = args.output_dir / destination.name
        destination = destination.resolve()
        if destination in outputs:
            parser.error("duplicate output path: " + str(destination))
        try:
            metadata, body = read_post(source)
            outputs[destination] = render_post(source, metadata, body)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{source}: {error}\n")
        indexes.setdefault(destination.parent, []).append((destination.name, metadata))
    for directory, posts in indexes.items():
        outputs[directory / "index.html"] = render_index(posts)
    for destination, content in outputs.items():
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        except OSError as error:
            parser.exit(1, f"{destination}: {error}\n")
        print(destination)


if __name__ == "__main__":
    main()
