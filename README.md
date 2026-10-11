# Hirokazu Ishida's website

https://hiroishida.github.io/

Edit `index.html` to update the content or the CSS in its `<style>` block.
Images and other shared assets are kept in `asset/`. The homepage needs no build
step. Blog posts are written in Markdown and converted with `build_blog.py`.

## Blog

Edit `blog/curry.md`, then run:

```sh
python3 -m pip install Markdown==3.7  # once; Python 3.8 or newer
python3 build_blog.py
```

Or use `uv run build_blog.py` to install the script's declared dependency in an
isolated environment automatically.

By default, the script converts every `blog/*.md` and writes HTML next to each
source: `blog/curry.md` becomes `blog/curry.html`. The generated HTML is ignored
by Git; commit the Markdown and Python source instead. To build one post or
choose another output directory:

```sh
python3 build_blog.py blog/curry.md
python3 build_blog.py --output-dir _site/blog
```

The single Python file contains the page template and CSS. Each Markdown file
starts with simple metadata (`title`, `date` in `YYYY-MM-DD` format, and optional
`author` and `description`). The `---` header uses Python-Markdown's `meta`
extension, not full YAML: write plain values without quotes.

Normal Markdown, tables, fenced code blocks, and raw HTML/SVG/JavaScript are
supported. Put scripts or custom diagrams directly in the Markdown, outside
code fences. Use `markdown="1"` on an HTML container to parse Markdown inside it.
Put shared assets in `asset/` and refer to them as `../asset/filename` from posts.

Write inline math as `\(...\)` and display math as `\[...\]`, without blank lines
within an equation. `curry.md` keeps its display equations in HTML figures to
retain their borders and captions. MathJax loads from a CDN, so typesetting
requires an internet connection. The build itself does not fetch MathJax.

Posts remain accessible by their direct URLs and are not linked from the
homepage. The curry post stays at `/blog/curry.html`.

## Local preview

After building, open `blog/curry.html` directly in a browser, or serve this
directory (including the unchanged homepage):

```sh
python3 -m http.server 8000
```

Then visit http://localhost:8000/blog/curry.html.

## GitHub Pages

For the initial migration, set **Settings → Pages → Build and deployment →
Source → GitHub Actions**. Then push the workflow and sources to `master`.
`.github/workflows/pages.yml` builds the posts using the same Python script and
deploys `_site/` on each push, or when run manually from the Actions tab.

The published artifact contains root HTML files, `.nojekyll`, `asset/`, and the
generated `blog/*.html`. Markdown sources and local files are not published.
The workflow uploads the HTML artifact directly; it does not commit HTML back
to the repository. `_site/` is ignored by Git as well.

Keep `googlebce8ba87f93e52e5.html` at the repository root, with its name and
contents unchanged, to preserve Google's site ownership verification URL.
