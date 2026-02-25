# Deploying the Earthquake CNN Explainer

The app is a **Vite + React** static site. Build output is in `dist/`. These options work with zero code changes.

---

## Option 1: Vercel (recommended — easiest for React)

1. **Push your repo to GitHub** (if not already).

2. Go to [vercel.com](https://vercel.com) and sign in with GitHub.

3. **Import** your repo. Vercel will detect Vite and suggest:
   - **Root Directory:** `explainer-app` (or leave blank if you deploy from the repo root; see below)
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`

4. If the repo root is `tiny-cnn-seismicML` and the app lives in `explainer-app/`:
   - Set **Root Directory** to `explainer-app`.
   - Vercel will run `npm install` and `npm run build` inside that folder.

5. Click **Deploy**. Future pushes to your main branch will auto-deploy.

**From repo root (monorepo):**  
If you leave Root Directory blank, set **Build Command** to `cd explainer-app && npm ci && npm run build` and **Output Directory** to `explainer-app/dist`.

---

## Option 2: Netlify

1. Push to GitHub, then go to [netlify.com](https://netlify.com) and sign in with GitHub.

2. **Add new site → Import from Git** and choose your repo.

3. Configure:
   - **Base directory:** `explainer-app`
   - **Build command:** `npm run build`
   - **Publish directory:** `dist`

4. Deploy. Netlify will serve the SPA correctly (redirects for client-side routing are handled by default for SPAs).

---

## Option 3: GitHub Pages (free, no account beyond GitHub)

1. Install the deploy helper and add a deploy script:
   ```bash
   cd explainer-app
   npm install --save-dev gh-pages
   ```
   Add to `package.json` (inside `"scripts"`):
   ```json
   "deploy": "vite build && gh-pages -d dist"
   ```

2. **Set base path** for GitHub Pages (e.g. `https://<user>.github.io/tiny-cnn-seismicML/`).  
   In `explainer-app/vite.config.js`:
   ```js
   export default defineConfig({
     plugins: [react()],
     base: '/tiny-cnn-seismicML/',  // match your repo name
   })
   ```
   Then build with `npm run build` so assets use the correct paths.

3. In the repo on GitHub: **Settings → Pages → Source:** deploy from the **gh-pages** branch.

4. Run from the repo root (so `gh-pages` can find the repo):
   ```bash
   cd explainer-app && npm run deploy
   ```
   This builds and pushes `dist` to the `gh-pages` branch. The site will be at `https://<username>.github.io/tiny-cnn-seismicML/`.

---

## Before you deploy

- **Model weights:** Ensure `explainer-app/public/models/compact_weights.json` exists (run `python scripts/export_compact_weights_for_tfjs.py` from repo root if needed). Everything in `public/` is copied into `dist/` at the root.
- **Waveforms:** Ensure `explainer-app/public/waveforms.json` exists (run `python scripts/export_waveforms_for_explainer.py` if needed).
- **Images:** `public/images/` (test-predictions-grid.png, training-validation-curves.png) are included automatically.

---

## Quick comparison

|               | Vercel        | Netlify       | GitHub Pages   |
|---------------|----------------|---------------|----------------|
| Ease          | Easiest        | Easy          | A bit of setup |
| React/Vite    | Excellent      | Excellent     | Works          |
| Custom domain | Yes            | Yes           | Yes (subdomain)|
| Base path     | Usually `/`    | Usually `/`   | Often `/repo/` |
| Cost          | Free tier      | Free tier     | Free           |

**Recommendation:** Use **Vercel** or **Netlify** with **Root / Base directory** set to `explainer-app` for the smoothest React deploy with no base-path changes.
