# ReviewLens Deployment Guide

This document describes how to deploy ReviewLens across supported hosting targets: Streamlit Community Cloud, Docker / Docker Compose, and GitHub Pages (Progressive Web App Shell).

---

## 1. Streamlit Community Cloud (Recommended)

1. Fork or push this repository to your GitHub account (`https://github.com/Vinayvardhan224/Reviewlens`).
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and click **"New app"**.
3. Select:
   - **Repository**: `Vinayvardhan224/Reviewlens`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Expand **Advanced settings** > **Secrets** and paste your secrets configuration from `.streamlit/secrets.toml.example`:
   ```toml
   # Authentication Secrets
   REVIEWLENS_DEMO_MODE = "0"
   PBKDF2_ITERATIONS = "600000"

   # Gmail SMTP for Email OTP Verification
   SMTP_USER = "your-app-email@gmail.com"
   SMTP_APP_PASSWORD = "xxxx xxxx xxxx xxxx"

   # Twilio for SMS Mobile OTP (Optional)
   TWILIO_ACCOUNT_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   TWILIO_AUTH_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   TWILIO_PHONE_NUMBER = "+1234567890"

   # Streamlit Native Google OIDC Login (Optional)
   [auth]
   redirect_uri = "https://your-app.streamlit.app/oauth2callback"
   cookie_secret = "generate-a-secure-32-byte-hex-string"

   [auth.google]
   client_id = "your-client-id.apps.googleusercontent.com"
   client_secret = "GOCSPX-xxxxxxxxxxxxxxxxxxxxxxxx"
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   ```
5. Click **Deploy!**. Streamlit Cloud will automatically build dependencies from `requirements.txt` and launch the platform.

---

## 2. Docker & Docker Compose Deployment

ReviewLens provides containerized deployments for both the interactive Streamlit dashboard (`8501`) and the FastAPI REST microservice (`8000`).

### Single-Command Start:
```bash
docker compose up -d --build
```

### Services:
- **Streamlit Web UI**: `http://localhost:8501` (with automated healthcheck at `/_stcore/health`)
- **FastAPI REST API**: `http://localhost:8000` (with automated healthcheck at `/health`)
- **Interactive Swagger Docs**: `http://localhost:8000/docs`

---

## 3. Progressive Web App (PWA) on GitHub Pages

The `pwa/` folder contains a standalone Progressive Web App shell that can be installed on iOS, Android, and Desktop:

1. Enable **GitHub Pages** in your repository settings:
   - Go to **Settings** > **Pages**
   - Source: **GitHub Actions**
2. The workflow in `.github/workflows/pages.yml` will automatically build and publish the PWA whenever changes to `pwa/` are pushed to `main`.
3. Configure the PWA target URL:
   ```bash
   python scripts/set_app_url.py https://your-streamlit-app-url.streamlit.app
   ```
4. Once deployed, users visiting the GitHub Pages site can click **"Install App"** to install ReviewLens directly to their home screen or desktop.

---

## 4. Local Quick Start

### Windows:
Double-click `run_reviewlens.bat` or run:
```cmd
run_reviewlens.bat
```

### Linux / macOS:
```bash
chmod +x run_reviewlens.sh
./run_reviewlens.sh
```

### Running Tests:
```bash
python -m unittest discover tests
```
