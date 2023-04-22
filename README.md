[![Netlify Status](https://api.netlify.com/api/v1/badges/28614791-1d2a-4912-9bc2-472df3e6f8e4/deploy-status)](https://app.netlify.com/sites/aniket-blog/deploys)

# How to publish?

## Method 1:
[Github Action](./.github/workflows/publish.yml) automatically renders and pushes the rendered files to `gh-pages` branch.

## Method 2:
Run - `quarto publish gh-pages` and quarto will publish the rendered site into the gh-pages branch.
