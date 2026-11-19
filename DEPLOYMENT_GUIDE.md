# Production Deployment Guide
## EV Charging Station Optimization – Predicting Demand for EV Stations

This guide provides step-by-step instructions for deploying the complete full-stack machine learning application:
- **Frontend:** React 18 Single-Page Application deployed to **GitHub Pages**.
- **Backend:** Flask REST API with pre-trained XGBoost ML models deployed to **Render** (or any cloud container platform).

---

## 1. Architecture Overview

```
+-------------------------------------------------------------+
|                     CLIENT BROWSER                          |
|   https://harshithavijayakumar14.github.io/                 |
|            EV-Charging-Station-Optimizationc/               |
+------------------------------+------------------------------+
                               |
                               | HTTPS API Requests (VITE_API_BASE_URL)
                               v
+-------------------------------------------------------------+
|                      RENDER BACKEND                         |
|   https://<your-service-name>.onrender.com/api/...          |
|                                                             |
|   - Gunicorn WSGI Server                                    |
|   - Flask API Endpoints (/api/health, /api/dashboard, etc.) |
|   - Pre-trained XGBoost Models (1h & 24h demand forecaster) |
|   - Pre-computed Compact Telemetry & Hardware Inventory     |
+-------------------------------------------------------------+
```

---

## 2. Step 1: Deploy the Flask Backend on Render

Render provides free hosting for Python web services with native support for Gunicorn and Git integration.

### A. Create Web Service on Render
1. Sign in to [Render](https://render.com/) (create an account using your GitHub account if needed).
2. On your Render Dashboard, click **New +** and select **Web Service**.
3. Choose **Build and deploy from a Git repository** and connect your GitHub repository:
   `https://github.com/HarshithaVijayakumar14/EV-Charging-Station-Optimizationc`
4. Configure the service settings:
   - **Name:** `ev-charging-api` (or a name of your choice)
   - **Region:** Any region close to your users (e.g., *Oregon (US West)* or *Frankfurt (EU)*)
   - **Branch:** `main`
   - **Root Directory:** *(leave blank — defaults to repository root)*
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     gunicorn backend.app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
     ```
   - **Instance Type:** `Free`

### B. Configure Environment Variables on Render
Scroll down to the **Environment Variables** section and add:

| Key | Value | Purpose |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.11.9` | Ensures compatible Python runtime |
| `CORS_ORIGINS` | `https://harshithavijayakumar14.github.io,http://localhost:5173,http://localhost:5000` | Whitelists your GitHub Pages domain for API calls |

*(Note: Render automatically injects the `$PORT` variable).*

### C. Deploy and Copy Backend URL
1. Click **Create Web Service**.
2. Render will clone the repository, install Python packages, load the ML models and pre-computed cache, and start Gunicorn.
3. Once the build completes and status changes to **Live**, copy your public service URL from the top of the page.  
   *(Example: `https://ev-charging-api-xxxx.onrender.com`)*.
4. **Test the health endpoint:** Open `https://<your-render-url>/api/health` in your browser. You should see:
   ```json
   {
     "status": "healthy",
     "service": "EV Charging Station Demand Prediction API",
     "models": {
       "models_ready": true,
       "model_1h_loaded": true,
       "model_24h_loaded": true
     }
   }
   ```

---

## 3. Step 2: Connect Frontend on GitHub Pages to Render

Now configure your GitHub repository so GitHub Actions compiles the React frontend pointing to your live Render backend.

### A. Set the Repository Variable on GitHub
1. Open your repository on GitHub:  
   `https://github.com/HarshithaVijayakumar14/EV-Charging-Station-Optimizationc`
2. Go to **Settings** (top navigation tab).
3. In the left sidebar, expand **Secrets and variables** and click **Actions**.
4. Select the **Variables** tab (next to *Secrets*).
5. Click **New repository variable**.
6. Set:
   - **Name:** `VITE_API_BASE_URL`
   - **Value:** `https://<your-render-url>` *(your exact Render URL from Step 1, without a trailing slash)*
7. Click **Add variable**.

### B. Configure GitHub Pages Source
1. Still in repository **Settings**, click **Pages** in the left sidebar.
2. Under **Build and deployment** > **Source**, ensure **GitHub Actions** is selected.

### C. Trigger Automated Deployment
1. Go to the **Actions** tab of your repository.
2. Select the **Deploy to GitHub Pages** workflow on the left.
3. Click **Run workflow** > **Run workflow** on the `main` branch (or push a commit to `main`).
4. GitHub Actions will automatically:
   - Install Node dependencies (`npm ci`).
   - Inject `VITE_API_BASE_URL` into Vite.
   - Build the frontend bundle with base path `/EV-Charging-Station-Optimizationc/`.
   - Deploy to GitHub Pages.

---

## 4. Step 3: Verify the Live Website

Once the GitHub Actions workflow finishes:

1. Open your live website:  
   👉 **`https://harshithavijayakumar14.github.io/EV-Charging-Station-Optimizationc/`**
2. Check the status badge in the top right:
   - It will display **`API & Models Ready`** in green with an active pulse indicator.
3. Verify all views:
   - **Network Overview:** 259,411 sessions, 2.216 GWh delivered, 46 stations, and interactive charts load without 404 errors.
   - **Find a Station (Phase 8):** Select your vehicle model and target station to view live port availability estimates, CC/CV battery taper charging time, and nearby alternative recommendations.
   - **Demand Forecasts:** 1-hour and 24-hour ahead predictions generate live from XGBoost.
   - **Station Explorer:** Telemetry and dwell duration statistics load for all stations.
   - **Model Evaluation:** Validation 2018, Test 2019, and COVID 2020 metrics display accurately.

---

## 5. Important Production Notes & FAQ

### Render Free Tier Spin-Down (Cold Start)
- Render's free tier spins down inactive web services after 15 minutes of inactivity.
- When you first visit your site after inactivity, the first API request may take **30 to 50 seconds** while Render boots the container.
- The frontend handles this gracefully: the top-right badge will show `Connecting to API...` and automatically connect once the server responds. Subsequent requests are fast (< 100 ms).
- If desired, free external ping services (such as UptimeRobot or Cron-Job.org) can ping `/api/health` every 10 minutes to keep the instance warm.

### Local Development vs. Production
- **Local Dev:** Running `python backend/app.py` runs Flask on port 5000 and serves both API and frontend locally at `http://localhost:5000` without requiring external hosting.
- **Production:** The React frontend on GitHub Pages communicates over HTTPS with your Render backend via `VITE_API_BASE_URL`.
