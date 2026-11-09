# Packaging & Cross-Platform Deployment

MoS Analyzer is designed to run seamlessly across **Linux**, **macOS** (both Apple Silicon and Intel), and **Windows**.

All heavy scientific dependencies—including `scikit-fem`, `gmsh` (with OpenCASCADE geometry kernel), `pyvista` (VTK), `torch`, and `trame`—are bundled and prepared up front. The only components downloaded on demand are model weights from Hugging Face when AI computer-vision features are invoked.

---

## 1. One-Click Native Deployment (Recommended)

The project includes pre-configured universal launchers that use standalone `pixi` to manage pre-compiled C/C++ scientific binaries across platforms without requiring administrative privileges, pre-installed Python, or root access.

### On Linux & macOS
Run the universal launcher script:
```bash
./run_app.sh
```
What this does:
1. Automatically acquires a lightweight portable binary manager if not already on your system.
2. Installs the exact locked environment (`pixi.toml`) containing native C++ binaries for Gmsh, VTK, Scikit-FEM, and PyTorch.
3. Automatically launches the local web server and opens `http://localhost:8080` in your default browser.

### On Windows
Double-click `run_app.bat` or run from PowerShell / Command Prompt:
```cmd
run_app.bat
```
This sets up the native Windows environment with all pre-compiled dependencies and opens the studio in your browser.

---

## 2. Universal Docker / Podman Container

For systems with Docker or Podman (including Windows via Docker Desktop, macOS, and Linux servers):

```bash
# Build and start the container
docker compose up
```
Or directly with Docker:
```bash
docker build -t mos-analyzer .
docker run -p 8080:8080 -v mos_hf_cache:/root/.cache/huggingface mos-analyzer
```
Then navigate to `http://localhost:8080`.

**Key features of the Docker deployment:**
* **Pre-baked dependencies:** All system libraries (`OpenGL`, `Mesa`, `Gmsh`, `OpenCASCADE`) are baked into the image.
* **Persistent model cache:** Hugging Face weights are cached in a named Docker volume (`mos_hf_cache`), so AI models are only downloaded once.

---

## 3. Hugging Face Models & Offline Operation

### On-Demand Downloads
All core FEA simulation, CAD import, meshing, and Learning Lab features run **100% offline with zero network requests**.

When you trigger AI-assisted computer vision tools:
* **Depth Anything V2**: `depth-anything/Depth-Anything-V2-Small-hf` (~100 MB)
* **Grounding DINO & SAM 2**: `IDEA-Research/grounding-dino-tiny` and `facebook/sam2.1-hiera-tiny` (~300 MB total)

Weights are fetched once via `transformers` / `huggingface_hub` and cached permanently in `~/.cache/huggingface/hub/`.

### Pre-caching for Air-Gapped Environments
If deploying on an air-gapped machine without internet access, pre-download the models on an online machine:
```bash
python -c "from transformers import pipeline; pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf')"
```
Then copy your `~/.cache/huggingface/` folder to the target machine.

---

## 4. Global Task Cancellation

If an operation is started accidentally (such as an extensive AI download, high-resolution visual hull carving, or an oversized finite-element solve):

1. A prominent **Global Status Banner** appears at the top of the interface:
   `⏳ [Task Name] in progress...`
2. Click the bright red **Cancel Operation** button.
3. The server immediately sets a cooperative cancellation token, halts subprocesses, resets solver flags, and restores the UI without corrupting state or freezing the browser.
