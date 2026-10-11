# Hirokazu Ishida's website

https://hiroishida.github.io/

Edit `index.html` to update the content or the CSS in its `<style>` block.
Images are kept in `asset/`. No dependencies or build step are required.

## Blog

Posts are plain HTML files in `blog/`. Edit the article HTML directly; each post
includes its own styles. Posts are currently accessible by their direct URLs
and are not linked from `index.html`.
The first post, `blog/curry.html`, also demonstrates a responsive HTML/CSS diagram
and LaTeX equations rendered by MathJax. MathJax is loaded from a CDN, so rendering
the equations requires an internet connection; no local installation is needed.

## Local preview

Open `index.html` directly in a browser, or serve this directory:

```sh
python3 -m http.server 8000
```

Then visit http://localhost:8000/.

## GitHub Pages

Publish the repository root from the site's publishing branch. Keep `.nojekyll`
in place so GitHub Pages serves the files without running Jekyll.

Keep `googlebce8ba87f93e52e5.html` at the repository root, with its name and
contents unchanged, to preserve Google's site ownership verification URL.
