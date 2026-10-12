# Hirokazu Ishida's website

https://hiroishida.github.io/

Edit `index.html` to update the content or the CSS in its `<style>` block.
Images and other shared assets are kept in `asset/`. The homepage needs no build
step. Posts are written in Markdown and converted with `build_post.py`.

## Posts

Writing guidelines for new posts and revisions are in [AGENTS.md](AGENTS.md).

Edit a Markdown file such as `post/nvidia_gpu_internal.md`, then run:

```sh
python3 -m pip install Markdown==3.7  # once; Python 3.8 or newer
python3 build_post.py
```

Or use `uv run build_post.py` to install the script's declared dependency in an
isolated environment automatically.

By default, the script converts every `post/*.md` and writes HTML next to each
source: `post/nvidia_gpu_internal.md` becomes `post/nvidia_gpu_internal.html`.
It also generates `post/index.html`, listing titles and dates with links to all
posts, newest first. Both the article HTML and index HTML are ignored by Git; commit the
Markdown and Python source instead. To choose another output directory:

```sh
python3 build_post.py --output-dir _site/post
```

You can also pass specific Markdown files, for example
`python3 build_post.py post/nvidia_gpu_internal.md`. In that case, the generated
index lists only those files. Run without filenames to rebuild the complete index.
`index.md` is reserved because `index.html` is generated automatically.

The single Python file contains the page template and CSS. Each Markdown file
starts with simple metadata (`title`, `date` in `YYYY-MM-DD` format, and optional
`author` and `description`). The `---` header uses Python-Markdown's `meta`
extension, not full YAML: write plain values without quotes.

Normal Markdown, tables, fenced code blocks, and raw HTML/SVG/JavaScript are
supported. Put scripts or custom diagrams directly in the Markdown, outside
code fences. Use `markdown="1"` on an HTML container to parse Markdown inside it.
Put shared assets in `asset/` and refer to them as `../asset/filename` from posts.

Write inline math as `\(...\)` and display math as `\[...\]`, without blank lines
within an equation. `nvidia_gpu_internal.md` wraps its display equation in a
`<div class="equation math-scroll">` for a border and horizontal scrolling.
MathJax loads from a CDN, so typesetting requires an internet connection.
The build itself does not fetch MathJax.

The article index is at `/post/`, and the GPU article is at
`/post/nvidia_gpu_internal.html`. Each article links back to the index.
The homepage does not link to the posts.

## Local preview

After building, open `post/index.html` directly in a browser, or serve this
directory (including the unchanged homepage):

```sh
python3 -m http.server 8000
```

Then visit http://localhost:8000/post/.

## GitHub Pages

For the initial migration, set **Settings → Pages → Build and deployment →
Source → GitHub Actions**. Then push the workflow and sources to `master`.
`.github/workflows/pages.yml` builds the posts using the same Python script and
deploys `_site/` on each push, or when run manually from the Actions tab.

The published artifact contains root HTML files, `.nojekyll`, `asset/`, and the
generated `post/*.html` (including the index). Markdown sources and local files
are not published.
The workflow uploads the HTML artifact directly; it does not commit HTML back
to the repository. `_site/` is ignored by Git as well.

Keep `googlebce8ba87f93e52e5.html` at the repository root, with its name and
contents unchanged, to preserve Google's site ownership verification URL.
