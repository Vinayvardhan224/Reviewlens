# ReviewLens Progressive Web App (PWA) Shell

This directory contains the standalone Progressive Web App (PWA) distribution shell for ReviewLens.

## Structure
- `index.html`: Responsive app container embedding the production Streamlit URL in a dedicated fullscreen web view.
- `config.js`: Target `APP_URL` variable.
- `manifest.webmanifest`: Web app manifest configured with `#07051A` theme, standalone display mode, and icons.
- `sw.js`: Service worker caching static assets and providing offline fallback.
- `offline.html`: Branded offline notification page.
- `icons/`: PWA app icons (192x192 and 512x512).

## Deployment to GitHub Pages
The GitHub Actions workflow in `.github/workflows/pages.yml` automatically deploys this directory to GitHub Pages on pushes to `main`.

To update the Streamlit app target URL, run:
```bash
python scripts/set_app_url.py https://your-streamlit-app-url.streamlit.app
```
