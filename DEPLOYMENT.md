# Deployment and Production Guide

## AI-Powered Construction-Site Vehicle Classification System
**Project Codename:** WhiteVision  
**Public Live Deployment URL:** [https://67751f623e9f81.lhr.life](https://67751f623e9f81.lhr.life)  
**Local Endpoints:**  
- Web Dashboard: [http://localhost:8501](http://localhost:8501)  
- REST API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)  

---

## 1. Domain Enforcement (Strict Construction Site Verification)

The system enforces a **strict positive domain validation gate** ensuring that **no image other than genuine construction-site vehicle images is classified**:

1. **Semantic Category Interception**:
   - Zero-shot ImageNet semantic classification rejects passenger automobiles (`sports_car`, `racer`, `convertible`, `sedan`, `minivan`, `cab`), domestic animals, humans, apparel, pop culture media/artwork, and indoor household objects.
2. **Deep Geometric Centroid Distance Gate**:
   - Compares the 1,280-dimensional feature embedding of the query image against the precomputed mathematical centroids of all 8 construction machinery classes (`models/centroids.npy`).
   - If the maximum cosine similarity across all construction classes is $< 0.12$, the image is instantly rejected as out-of-distribution before classification.
3. **Operational Confidence Cutoff (50%)**:
   - Even if an image passes the initial gates, any output with $< 50.0\%$ confidence is automatically labeled `UNKNOWN / UNCERTAIN [LOW CONFIDENCE]`.
   - The UI completely suppresses construction vehicle labels and dossiers for uncertain predictions.

---

## 2. Deployment Methods

### Option A: Live Public Deployment (Active Now)
The system is currently exposed through a secure TLS/HTTPS public tunnel:
- **Public URL**: `https://67751f623e9f81.lhr.life`
- Accessible directly from mobile phones, tablets, or external desktop browsers without any local installation.

To start or restart this tunnel manually at any time:
```bash
ssh -o StrictHostKeyChecking=no -R 80:127.0.0.1:8501 nokey@localhost.run
```

---

### Option B: 1-Click Streamlit Community Cloud (Free Cloud Hosting)

1. Push your repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Deploy Construction Vehicle Classification System"
   git branch -M main
   git remote add origin https://github.com/<your-username>/construction-vehicle-classifier.git
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repository.
4. Set the Main file path to: `app/streamlit_app.py`.
5. Click **Deploy**. Streamlit Cloud will automatically build and host the application at a permanent `*.streamlit.app` URL.

---

### Option C: Hugging Face Spaces (Free Cloud Hosting with GPU Support)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the Space SDK.
3. Clone the Space repository and copy the project files:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/construction-vehicle-classifier
   cp -r * construction-vehicle-classifier/
   cd construction-vehicle-classifier
   git add .
   git commit -m "Deploy WhiteVision on Hugging Face Spaces"
   git push
   ```
4. Hugging Face Spaces will automatically launch your containerized application.

---

### Option D: Docker Container Deployment

Build and run both the Streamlit UI and FastAPI backend using Docker Compose:

```bash
# Build and start all services in detached mode
docker compose up -d --build

# View container logs
docker compose logs -f

# Stop containers
docker compose down
```

Services will be accessible at:
- Web Dashboard: `http://localhost:8501`
- REST API: `http://localhost:8000`

---

### Option E: Local Production Execution (Daemon Mode)

Run both the Web Application and the REST API locally:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Launch FastAPI backend service (port 8000)
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Launch Streamlit operations console (port 8501)
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
```
