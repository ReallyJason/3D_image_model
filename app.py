"""
3D Vision Lab — Unified Multi-Engine Backend.

Engines Supported:
1. TripoSR (Stability AI / VAST) — 100% Local Apple Silicon GPU (MPS), no token, no internet required.
2. InstantMesh (Tencent) — Cloud multi-view diffusion with FlexiCubes topology (~46,000 vertices).
3. TRELLIS.2 (Microsoft) — State-of-the-art flow matching diffusion model.
4. TinyImageToVoxelNet — Custom research baseline (32^3 voxel grid).
"""

import os
import sys
import time
import json
import base64
import io
import mimetypes
import glob
import shutil
from typing import Dict, Any, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
import trimesh
from PIL import Image
import torch
import torchvision.transforms as T

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRIPOSR_DIR = os.path.join(PROJECT_ROOT, "models", "triposr")
if os.path.exists(TRIPOSR_DIR) and TRIPOSR_DIR not in sys.path:
    sys.path.insert(0, TRIPOSR_DIR)

from src.model import TinyImageToVoxelNet
from src.dataset import DEFAULT_DATASET_DIR

WEB_DIR = os.path.join(PROJECT_ROOT, "web")
RECONSTRUCTIONS_DIR = os.path.join(PROJECT_ROOT, "reconstructions")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tiny_voxel_model.pt")

os.makedirs(WEB_DIR, exist_ok=True)
os.makedirs(RECONSTRUCTIONS_DIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# -------------------------------------------------------------
# 1. Local TripoSR Engine (Stability AI)
# -------------------------------------------------------------
class TripoSREngine:
    def __init__(self):
        print(f"🚀 Initializing Local TripoSR Engine on {DEVICE}...")
        self.device = DEVICE
        self.available = False
        try:
            from tsr.system import TSR
            from tsr.utils import remove_background, resize_foreground
            import rembg

            self.tsr = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt"
            )
            self.tsr.renderer.set_chunk_size(8192)
            self.tsr.to(self.device)
            self.rembg_session = rembg.new_session()
            self.remove_bg_fn = remove_background
            self.resize_fg_fn = resize_foreground
            self.available = True
            print("✅ Local TripoSR is active and ready on Apple Silicon!")
        except Exception as e:
            print(f"⚠️ TripoSR local engine not available: {e}")

    def predict_mesh(self, img: Image.Image, resolution: int = 192, remove_bg: bool = True) -> Dict[str, Any]:
        t0 = time.perf_counter()
        img_rgb = img.convert("RGB")
        prep_t0 = time.perf_counter()

        if remove_bg:
            nobg = self.remove_bg_fn(img_rgb, self.rembg_session)
            fg = self.resize_fg_fn(nobg, 0.85)
            fg_np = np.array(fg).astype(np.float32) / 255.0
            comp = fg_np[:, :, :3] * fg_np[:, :, 3:4] + (1 - fg_np[:, :, 3:4]) * 0.5
            proc_img = Image.fromarray((comp * 255.0).astype(np.uint8))
        else:
            proc_img = img_rgb

        prep_ms = (time.perf_counter() - prep_t0) * 1000.0

        infer_t0 = time.perf_counter()
        with torch.no_grad():
            scene_codes = self.tsr([proc_img], device=self.device)
            meshes = self.tsr.extract_mesh(scene_codes, True, resolution=resolution)

        if self.device.type == "mps":
            torch.mps.synchronize()

        inference_ms = (time.perf_counter() - infer_t0) * 1000.0
        mesh = meshes[0]

        ts = int(time.time() * 1000)
        obj_filename = f"triposr_{ts}.obj"
        glb_filename = f"triposr_{ts}.glb"
        obj_path = os.path.join(RECONSTRUCTIONS_DIR, obj_filename)
        glb_path = os.path.join(RECONSTRUCTIONS_DIR, glb_filename)

        mesh.export(glb_path)
        mesh.export(obj_path)

        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "success": True,
            "engine": "TripoSR (100% Local Apple Silicon GPU)",
            "obj_url": f"/reconstructions/{obj_filename}",
            "glb_url": f"/reconstructions/{glb_filename}",
            "inference_ms": round(inference_ms, 1),
            "total_latency_ms": round(total_ms, 1),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "has_colors": mesh.visual.vertex_colors is not None
        }

# -------------------------------------------------------------
# 2. Cloud InstantMesh Engine (Tencent)
# -------------------------------------------------------------
class InstantMeshEngine:
    def predict_mesh(self, temp_img_path: str, hf_token: Optional[str] = None) -> Dict[str, Any]:
        from gradio_client import Client, handle_file

        t0 = time.perf_counter()
        token = hf_token or os.environ.get("HF_TOKEN")
        client = Client("TencentARC/InstantMesh", hf_token=token)

        prep = client.predict(input_image=handle_file(temp_img_path), do_remove_background=True, api_name='/preprocess')
        mvs = client.predict(input_image=handle_file(prep), sample_steps=40, sample_seed=42, api_name='/generate_mvs')
        res = client.predict(api_name='/make3d')

        ts = int(time.time() * 1000)
        obj_filename = f"instantmesh_{ts}.obj"
        glb_filename = f"instantmesh_{ts}.glb"
        obj_path = os.path.join(RECONSTRUCTIONS_DIR, obj_filename)
        glb_path = os.path.join(RECONSTRUCTIONS_DIR, glb_filename)

        shutil.copy(res[0], obj_path)
        shutil.copy(res[1], glb_path)

        mesh = trimesh.load(glb_path, force="mesh")
        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "success": True,
            "engine": "InstantMesh (Tencent Multi-View Diffusion)",
            "obj_url": f"/reconstructions/{obj_filename}",
            "glb_url": f"/reconstructions/{glb_filename}",
            "inference_ms": round(total_ms, 1),
            "total_latency_ms": round(total_ms, 1),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "has_colors": True
        }

# -------------------------------------------------------------
# 3. Cloud TRELLIS.2 Engine (Microsoft)
# -------------------------------------------------------------
class TrellisEngine:
    def predict_mesh(self, temp_img_path: str, hf_token: Optional[str] = None) -> Dict[str, Any]:
        from gradio_client import Client, handle_file

        t0 = time.perf_counter()
        token = hf_token or os.environ.get("HF_TOKEN")
        client = Client("microsoft/TRELLIS.2", hf_token=token)

        client.predict(api_name='/start_session')
        prep = client.predict(input=handle_file(temp_img_path), api_name='/preprocess_image')
        prep_path = prep if isinstance(prep, str) else prep.get("path")

        client.predict(
            image=handle_file(prep_path),
            seed=0,
            resolution='1024',
            ss_guidance_strength=7.5,
            ss_guidance_rescale=0.7,
            ss_sampling_steps=12,
            ss_rescale_t=5.0,
            shape_slat_guidance_strength=7.5,
            shape_slat_guidance_rescale=0.5,
            shape_slat_sampling_steps=12,
            shape_slat_rescale_t=3.0,
            tex_slat_guidance_strength=1.0,
            tex_slat_guidance_rescale=0.0,
            tex_slat_sampling_steps=12,
            tex_slat_rescale_t=3.0,
            api_name='/image_to_3d'
        )

        res_glb = client.predict(
            decimation_target=100000,
            texture_size=1024,
            api_name='/extract_glb'
        )

        ts = int(time.time() * 1000)
        glb_filename = f"trellis_{ts}.glb"
        obj_filename = f"trellis_{ts}.obj"
        glb_path = os.path.join(RECONSTRUCTIONS_DIR, glb_filename)
        obj_path = os.path.join(RECONSTRUCTIONS_DIR, obj_filename)

        shutil.copy(res_glb[0], glb_path)
        mesh = trimesh.load(glb_path, force="mesh")
        mesh.export(obj_path)

        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "success": True,
            "engine": "TRELLIS.2 (Microsoft Flow Transformer)",
            "obj_url": f"/reconstructions/{obj_filename}",
            "glb_url": f"/reconstructions/{glb_filename}",
            "inference_ms": round(total_ms, 1),
            "total_latency_ms": round(total_ms, 1),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "has_colors": True
        }

# -------------------------------------------------------------
# 4. Custom Voxel Baseline
# -------------------------------------------------------------
class CustomVoxelEngine:
    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH):
        self.device = DEVICE
        self.model = TinyImageToVoxelNet(latent_dim=512, voxel_res=32).to(self.device)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_mesh(self, img: Image.Image, threshold: float = 0.45) -> Dict[str, Any]:
        t0 = time.perf_counter()
        img_rgb = img.convert("RGB")
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        if self.device.type == "mps":
            torch.mps.synchronize()

        inference_ms = (time.perf_counter() - t0) * 1000.0
        binary_grid = probs > threshold
        if int(binary_grid.sum()) < 8:
            binary_grid[14:18, 14:18, 14:18] = True

        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(binary_grid, pitch=1.0/32.0)
        mesh.vertices -= mesh.bounding_box.centroid

        ts = int(time.time() * 1000)
        obj_filename = f"voxel_{ts}.obj"
        glb_filename = f"voxel_{ts}.glb"
        obj_path = os.path.join(RECONSTRUCTIONS_DIR, obj_filename)
        glb_path = os.path.join(RECONSTRUCTIONS_DIR, glb_filename)

        mesh.export(obj_path)
        try:
            mesh.export(glb_path)
        except Exception:
            glb_filename = None

        return {
            "success": True,
            "engine": "TinyImageToVoxelNet (Custom 32³ Baseline)",
            "obj_url": f"/reconstructions/{obj_filename}",
            "glb_url": f"/reconstructions/{glb_filename}" if glb_filename else None,
            "inference_ms": round(inference_ms, 1),
            "total_latency_ms": round(inference_ms, 1),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "has_colors": False
        }

LOCAL_TRIPOSR = TripoSREngine()
CLOUD_INSTANTMESH = InstantMeshEngine()
CLOUD_TRELLIS = TrellisEngine()
CUSTOM_VOXEL = CustomVoxelEngine()

# -------------------------------------------------------------
# HTTP Request Handler
# -------------------------------------------------------------
class WebViewerHandler(BaseHTTPRequestHandler):
    def send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/samples":
            samples = []
            categories = ["shoe", "furniture", "car", "weapon", "characters", "stair"]
            for cat in categories:
                pattern = os.path.join(DEFAULT_DATASET_DIR, "objects", f"*{cat}*", "view_000.png")
                matches = glob.glob(pattern)
                for m in matches[:3]:
                    rel_path = os.path.relpath(m, PROJECT_ROOT)
                    obj_name = os.path.basename(os.path.dirname(m))
                    samples.append({
                        "name": obj_name.replace("_", " ").title(),
                        "category": cat.capitalize(),
                        "url": f"/{rel_path}"
                    })
            self.send_json(samples[:12])
            return

        if path.startswith("/reconstructions/"):
            filename = os.path.basename(path)
            file_path = os.path.join(RECONSTRUCTIONS_DIR, filename)
            if os.path.exists(file_path):
                query = parse_qs(parsed.query)
                is_download = "download" in query or "download" in parsed.query
                self.serve_file(file_path, as_attachment=is_download)
            else:
                self.send_error(404, "Reconstruction file not found")
            return

        if path.startswith("/my_dataset/"):
            file_path = os.path.join(PROJECT_ROOT, path.lstrip("/"))
            if os.path.exists(file_path):
                self.serve_file(file_path)
            else:
                self.send_error(404, "Dataset image not found")
            return

        if path == "/" or path == "/index.html":
            file_path = os.path.join(WEB_DIR, "index.html")
        else:
            file_path = os.path.join(WEB_DIR, path.lstrip("/"))

        if os.path.exists(file_path) and os.path.isfile(file_path):
            self.serve_file(file_path)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/reconstruct":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                engine_type = data.get("engine", "trellis")
                resolution = int(data.get("resolution", 192))
                remove_bg = bool(data.get("remove_bg", True))
                threshold = float(data.get("threshold", 0.45))
                hf_token = data.get("hf_token", "").strip() or None

                temp_path = os.path.join(RECONSTRUCTIONS_DIR, f"temp_input_{int(time.time()*1000)}.png")

                if "image_base64" in data:
                    raw_b64 = data["image_base64"]
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",", 1)[1]
                    img_bytes = base64.b64decode(raw_b64)
                    img = Image.open(io.BytesIO(img_bytes))
                    img.save(temp_path)
                elif "image_url" in data:
                    img_url = data["image_url"].lstrip("/")
                    local_path = os.path.join(PROJECT_ROOT, img_url)
                    if not os.path.exists(local_path):
                        self.send_json({"error": f"Image file not found: {local_path}"}, status=400)
                        return
                    img = Image.open(local_path)
                    shutil.copy(local_path, temp_path)
                else:
                    self.send_json({"error": "No image provided"}, status=400)
                    return

                # Route to selected engine
                if engine_type == "trellis":
                    result = CLOUD_TRELLIS.predict_mesh(temp_path, hf_token=hf_token)
                elif engine_type == "instantmesh":
                    result = CLOUD_INSTANTMESH.predict_mesh(temp_path, hf_token=hf_token)
                elif engine_type == "triposr":
                    result = LOCAL_TRIPOSR.predict_mesh(img, resolution=resolution, remove_bg=remove_bg)
                else:
                    result = CUSTOM_VOXEL.predict_mesh(img, threshold=threshold)

                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

                self.send_json(result)

            except Exception as e:
                import traceback
                traceback.print_exc()
                err_str = str(e)
                if "ZeroGPU quota" in err_str:
                    err_str = "ZeroGPU free anonymous quota reached. Please paste your free Hugging Face token into the 'Hugging Face Token' box (or open the official TRELLIS space directly)."
                self.send_json({"error": err_str}, status=500)
        else:
            self.send_error(404, "Endpoint Not Found")

    def serve_file(self, file_path: str, as_attachment: bool = False):
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            if file_path.endswith(".obj"):
                mime_type = "model/obj"
            elif file_path.endswith(".glb"):
                mime_type = "model/gltf-binary"
            else:
                mime_type = "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            if as_attachment:
                filename = os.path.basename(file_path)
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

def run_server(port: int = 8080):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, WebViewerHandler)
    url = f"http://localhost:{port}"

    print("\n" + "=" * 65)
    print("      🌐 3D VISION LAB — MULTI-ENGINE 3D VIEWER")
    print("=" * 65)
    print(f"  Server URL:    {url}")
    print(f"  Engines:       TRELLIS.2 (Microsoft) | InstantMesh | TripoSR | Custom")
    print("=" * 65 + "\n", flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port=port)
