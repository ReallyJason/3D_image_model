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
import threading
from collections import defaultdict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
import trimesh
from PIL import Image
import torch
import torchvision.transforms as T

# Protect against image decompression bombs
Image.MAX_IMAGE_PIXELS = 16_000_000

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
# Production Security & Rate Limiting System
# -------------------------------------------------------------
MAX_UPLOAD_SIZE = 12 * 1024 * 1024         # 12 MB max upload limit (prevents DoS / memory exhaustion)
RATE_LIMIT_WINDOW = 60                     # 60 second rolling window
MAX_RECONSTRUCT_PER_MINUTE = 15            # Max 15 neural 3D generations per IP/minute
MAX_TOTAL_REQUESTS_PER_MINUTE = 120        # Max 120 total requests per IP/minute
MAX_CONCURRENT_INFERENCES = 2              # Max 2 heavy concurrent reconstructions simultaneously

class SecurityManager:
    """Thread-safe rate limiter, concurrency controller, and disk cleaner."""
    def __init__(self):
        self._lock = threading.Lock()
        self._ip_api_timestamps = defaultdict(list)
        self._ip_total_timestamps = defaultdict(list)
        self._inference_semaphore = threading.Semaphore(MAX_CONCURRENT_INFERENCES)
        self._active_inferences = 0
        self._last_clean_time = time.time()

    def get_client_ip(self, handler: BaseHTTPRequestHandler) -> str:
        # Check standard reverse-proxy headers if behind Cloudflare / Nginx
        cf_ip = handler.headers.get("CF-Connecting-IP")
        if cf_ip:
            return cf_ip.strip()
        forwarded = handler.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return handler.client_address[0]

    def check_rate_limit(self, client_ip: str, is_heavy_api: bool = False) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            # Clean up old timestamps
            self._ip_total_timestamps[client_ip] = [
                t for t in self._ip_total_timestamps[client_ip] if now - t < RATE_LIMIT_WINDOW
            ]
            if len(self._ip_total_timestamps[client_ip]) >= MAX_TOTAL_REQUESTS_PER_MINUTE:
                retry_after = int(RATE_LIMIT_WINDOW - (now - self._ip_total_timestamps[client_ip][0])) + 1
                return False, retry_after
            self._ip_total_timestamps[client_ip].append(now)

            if is_heavy_api:
                self._ip_api_timestamps[client_ip] = [
                    t for t in self._ip_api_timestamps[client_ip] if now - t < RATE_LIMIT_WINDOW
                ]
                if len(self._ip_api_timestamps[client_ip]) >= MAX_RECONSTRUCT_PER_MINUTE:
                    retry_after = int(RATE_LIMIT_WINDOW - (now - self._ip_api_timestamps[client_ip][0])) + 1
                    return False, retry_after
                self._ip_api_timestamps[client_ip].append(now)

        return True, 0

    def try_acquire_inference(self) -> bool:
        """Non-blocking semaphore acquire to prevent server queue exhaustion."""
        acquired = self._inference_semaphore.acquire(blocking=False)
        if acquired:
            with self._lock:
                self._active_inferences += 1
        return acquired

    def release_inference(self):
        with self._lock:
            if self._active_inferences > 0:
                self._active_inferences -= 1
        self._inference_semaphore.release()

    def auto_prune_old_reconstructions(self, max_age_seconds: int = 3600):
        """Clean up generated files older than 1 hour to prevent disk fill-up."""
        now = time.time()
        if now - self._last_clean_time < 300:
            return
        self._last_clean_time = now
        try:
            for f in glob.glob(os.path.join(RECONSTRUCTIONS_DIR, "*.*")):
                if os.path.isfile(f) and (now - os.path.getmtime(f) > max_age_seconds):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
        except Exception:
            pass

SECURITY = SecurityManager()

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
        if not self.available:
            raise RuntimeError("TripoSR local engine is not available or failed to initialize.")

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
        client = Client("TencentARC/InstantMesh", token=token)

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
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
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
        client = Client("microsoft/TRELLIS.2", token=token)

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
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
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
# 4. Cloud TripoSG Engine (VAST-AI DiT Diffusion + 3D VAE)
# -------------------------------------------------------------
class TripoSGEngine:
    """VAST-AI TripoSG: Diffusion Transformer with 3D VAE geometry synthesis."""
    def predict_mesh(self, temp_img_path: str, hf_token: Optional[str] = None) -> Dict[str, Any]:
        from gradio_client import Client, handle_file

        t0 = time.perf_counter()
        token = hf_token or os.environ.get("HF_TOKEN")
        client = Client("VAST-AI/TripoSG", token=token, httpx_kwargs={"timeout": 120.0})

        try:
            client.predict(api_name='/start_session')
        except Exception:
            pass

        res = client.predict(
            image=handle_file(temp_img_path),
            api_name='/image_to_3d'
        )

        glb_source = res if isinstance(res, str) else (res.get("path") if isinstance(res, dict) else res[0])

        ts = int(time.time() * 1000)
        obj_filename = f"triposg_{ts}.obj"
        glb_filename = f"triposg_{ts}.glb"
        obj_path = os.path.join(RECONSTRUCTIONS_DIR, obj_filename)
        glb_path = os.path.join(RECONSTRUCTIONS_DIR, glb_filename)

        shutil.copy(glb_source, glb_path)

        mesh = trimesh.load(glb_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])

        mesh.export(obj_path)
        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "success": True,
            "engine": "TripoSG (VAST-AI DiT Diffusion + 3D VAE)",
            "obj_url": f"/reconstructions/{obj_filename}",
            "glb_url": f"/reconstructions/{glb_filename}",
            "inference_ms": round(total_ms, 1),
            "total_latency_ms": round(total_ms, 1),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "has_colors": True
        }

# -------------------------------------------------------------
# 5. Cloud SPAR3D Engine (Spatial-Aware 3D Mesh)
# -------------------------------------------------------------
class SPAR3DEngine:
    """SPAR3D: Spatial-Aware Reconstruction 3D (Fast geometry + texture tradeoff)."""
    def predict_mesh(self, temp_img_path: str, hf_token: Optional[str] = None) -> Dict[str, Any]:
        from gradio_client import Client, handle_file

        t0 = time.perf_counter()
        token = hf_token or os.environ.get("HF_TOKEN")
        client = Client("Neha03/spar-3d-mesh-generator", token=token, httpx_kwargs={"timeout": 120.0})

        res = client.predict(
            image=handle_file(temp_img_path),
            api_name='/predict'
        )

        glb_source = res if isinstance(res, str) else (res.get("path") if isinstance(res, dict) else res[0])

        ts = int(time.time() * 1000)
        obj_filename = f"spar3d_{ts}.obj"
        glb_filename = f"spar3d_{ts}.glb"
        obj_path = os.path.join(RECONSTRUCTIONS_DIR, obj_filename)
        glb_path = os.path.join(RECONSTRUCTIONS_DIR, glb_filename)

        shutil.copy(glb_source, glb_path)

        mesh = trimesh.load(glb_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])

        mesh.export(obj_path)
        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "success": True,
            "engine": "SPAR3D (Spatial-Aware 3D Reconstruction)",
            "obj_url": f"/reconstructions/{obj_filename}",
            "glb_url": f"/reconstructions/{glb_filename}",
            "inference_ms": round(total_ms, 1),
            "total_latency_ms": round(total_ms, 1),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "has_colors": True
        }

# -------------------------------------------------------------
# 6. Custom Voxel Baseline
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
CLOUD_TRIPOSG = TripoSGEngine()
CLOUD_SPAR3D = SPAR3DEngine()
CUSTOM_VOXEL = CustomVoxelEngine()

# -------------------------------------------------------------
# HTTP Request Handler (Hardened with Security Guards)
# -------------------------------------------------------------
class WebViewerHandler(BaseHTTPRequestHandler):
    server_version = "3D-Vision-Lab/1.0"
    sys_version = ""

    def apply_security_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header("X-XSS-Protection", "1; mode=block")
        self.send_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin-allow-popups")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")

        # Tailored CSP for WebGPU, WASM Shaders, Three.js, and Hugging Face CDN
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-eval' 'wasm-unsafe-eval' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "img-src 'self' data: blob: https:; "
            "connect-src 'self' blob: data: https://huggingface.co https://*.huggingface.co https://cdn-lfs.huggingface.co https://cdn-lfs-us-1.huggingface.co https://cdn.jsdelivr.net; "
            "worker-src 'self' blob:; "
            "media-src 'self' blob:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'self';"
        )
        self.send_header("Content-Security-Policy", csp_policy)

    def send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.apply_security_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Forwarded-For")
        self.apply_security_headers()
        self.end_headers()

    def do_HEAD(self):
        self.handle_get_or_head(is_head=True)

    def do_GET(self):
        self.handle_get_or_head(is_head=False)

    def serve_404(self, is_head: bool = False):
        custom_404 = os.path.join(WEB_DIR, "404.html")
        if os.path.exists(custom_404):
            self.serve_file(custom_404, status=404, is_head=is_head)
        else:
            self.send_error(404, "Page Not Found")

    def handle_get_or_head(self, is_head: bool = False):
        client_ip = SECURITY.get_client_ip(self)
        allowed, retry_after = SECURITY.check_rate_limit(client_ip, is_heavy_api=False)
        if not allowed:
            self.send_response(429)
            self.send_header("Retry-After", str(retry_after))
            self.send_json({"error": f"Too many requests. Please wait {retry_after}s."}, status=429)
            return

        parsed = urlparse(self.path)
        path = parsed.path

        # Path Traversal Guard
        if ".." in path or "\x00" in path:
            self.serve_404(is_head=is_head)
            return

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
            rel_path = path[len("/reconstructions/"):].lstrip("/")
            file_path = os.path.normpath(os.path.join(RECONSTRUCTIONS_DIR, rel_path))
            if file_path.startswith(os.path.normpath(RECONSTRUCTIONS_DIR)) and os.path.exists(file_path):
                query = parse_qs(parsed.query)
                is_download = "download" in query or "download" in parsed.query
                self.serve_file(file_path, as_attachment=is_download, is_head=is_head)
            else:
                self.serve_404(is_head=is_head)
            return

        if path.startswith("/my_dataset/"):
            rel_path = path.lstrip("/")
            file_path = os.path.normpath(os.path.join(PROJECT_ROOT, rel_path))
            if file_path.startswith(os.path.normpath(PROJECT_ROOT)) and os.path.exists(file_path):
                self.serve_file(file_path, is_head=is_head)
            else:
                self.serve_404(is_head=is_head)
            return

        # Allowed static extensions
        ALLOWED_EXTENSIONS = {
            ".html", ".css", ".js", ".json", ".svg", ".png",
            ".jpg", ".jpeg", ".webp", ".ico", ".obj", ".glb",
            ".xml", ".txt", ".webmanifest"
        }

        if path == "/" or path == "/index.html":
            file_path = os.path.join(WEB_DIR, "index.html")
        else:
            file_path = os.path.normpath(os.path.join(WEB_DIR, path.lstrip("/")))

        _, ext = os.path.splitext(file_path)
        is_known_asset = (ext.lower() in ALLOWED_EXTENSIONS) or (os.path.basename(file_path) in ("robots.txt", "sitemap.xml", "site.webmanifest"))

        if file_path.startswith(os.path.normpath(WEB_DIR)) and os.path.exists(file_path) and os.path.isfile(file_path) and is_known_asset:
            self.serve_file(file_path, is_head=is_head)
        else:
            self.serve_404(is_head=is_head)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/reconstruct":
            client_ip = SECURITY.get_client_ip(self)
            allowed, retry_after = SECURITY.check_rate_limit(client_ip, is_heavy_api=True)
            if not allowed:
                self.send_response(429)
                self.send_header("Retry-After", str(retry_after))
                self.send_json({
                    "error": f"Rate limit exceeded (Max {MAX_RECONSTRUCT_PER_MINUTE} reconstructions/min). Please wait {retry_after}s before submitting again."
                }, status=429)
                return

            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > MAX_UPLOAD_SIZE:
                self.send_json({
                    "error": f"Payload too large ({content_length // (1024*1024)}MB). Maximum allowed upload is {MAX_UPLOAD_SIZE // (1024*1024)}MB."
                }, status=413)
                return

            if not SECURITY.try_acquire_inference():
                self.send_json({
                    "error": "Server is currently at maximum capacity processing other 3D reconstructions. Please wait a few seconds and try again."
                }, status=503)
                return

            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                engine_type = data.get("engine", "trellis")
                resolution = int(data.get("resolution", 192))
                remove_bg = bool(data.get("remove_bg", True))
                threshold = float(data.get("threshold", 0.45))
                
                # Check user-provided token, falling back to server-configured environment variable
                server_hf_token = os.environ.get("HF_TOKEN", "").strip() or None
                hf_token = data.get("hf_token", "").strip() or server_hf_token

                temp_path = os.path.join(RECONSTRUCTIONS_DIR, f"temp_input_{int(time.time()*1000)}.png")

                if "image_base64" in data:
                    raw_b64 = data["image_base64"]
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",", 1)[1]
                    img_bytes = base64.b64decode(raw_b64)
                    
                    # Verify image integrity and guard against decompression bombs
                    img_check = Image.open(io.BytesIO(img_bytes))
                    img_check.verify()
                    img = Image.open(io.BytesIO(img_bytes))
                    img.save(temp_path)
                elif "image_url" in data:
                    img_url = data["image_url"].lstrip("/")
                    # Path traversal guard for sample URLs
                    if ".." in img_url or "\x00" in img_url:
                        self.send_json({"error": "Invalid image URL"}, status=400)
                        return
                    local_path = os.path.normpath(os.path.join(PROJECT_ROOT, img_url))
                    if not (local_path.startswith(os.path.normpath(PROJECT_ROOT)) and os.path.exists(local_path)):
                        self.send_json({"error": f"Image file not found: {img_url}"}, status=400)
                        return
                    img = Image.open(local_path)
                    shutil.copy(local_path, temp_path)
                else:
                    self.send_json({"error": "No image provided"}, status=400)
                    return

                # Route to selected engine
                if engine_type == "trellis":
                    result = CLOUD_TRELLIS.predict_mesh(temp_path, hf_token=hf_token)
                elif engine_type == "triposg":
                    result = CLOUD_TRIPOSG.predict_mesh(temp_path, hf_token=hf_token)
                elif engine_type == "spar3d":
                    result = CLOUD_SPAR3D.predict_mesh(temp_path, hf_token=hf_token)
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
                # Strip internal project root path to avoid exposing server directory hierarchy
                err_str = err_str.replace(PROJECT_ROOT, "[ROOT]")
                if "ZeroGPU quota" in err_str:
                    err_str = "ZeroGPU free anonymous quota reached. Please paste your free Hugging Face token into the 'Hugging Face Token' box (or open the official TRELLIS space directly)."
                elif "CUDA out of memory" in err_str:
                    err_str = "Host GPU ran out of memory. Please lower resolution or try the client WebGPU engine."
                self.send_json({"error": f"Reconstruction failed: {err_str}"}, status=500)
            finally:
                SECURITY.release_inference()
                SECURITY.auto_prune_old_reconstructions()
        else:
            self.send_json({"error": "API endpoint not found"}, status=404)

    def serve_file(self, file_path: str, status: int = 200, as_attachment: bool = False, is_head: bool = False):
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            if file_path.endswith(".obj"):
                mime_type = "model/obj"
            elif file_path.endswith(".glb"):
                mime_type = "model/gltf-binary"
            elif file_path.endswith(".webmanifest"):
                mime_type = "application/manifest+json"
            elif file_path.endswith(".xml"):
                mime_type = "application/xml"
            elif file_path.endswith(".svg"):
                mime_type = "image/svg+xml"
            elif file_path.endswith(".ico"):
                mime_type = "image/x-icon"
            else:
                mime_type = "application/octet-stream"

        try:
            file_size = os.path.getsize(file_path)
            self.send_response(status)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Access-Control-Allow-Origin", "*")

            # Caching Headers: Immutable cache for static assets, no-store for HTML
            if file_path.endswith((".css", ".js", ".svg", ".png", ".jpg", ".webp", ".ico", ".obj", ".glb", ".webmanifest")):
                self.send_header("Cache-Control", "public, max-age=86400")
            else:
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")

            self.apply_security_headers()
            if as_attachment:
                filename = os.path.basename(file_path)
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            if not is_head:
                with open(file_path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

def run_server(port: int = None):
    # Support Railway dynamic $PORT and cloud environments
    env_port = os.environ.get("PORT")
    final_port = int(env_port) if env_port else (port or 8080)
    # Bind to 0.0.0.0 for Railway/Cloud so edge proxy routes incoming traffic
    host = "0.0.0.0" if (env_port or os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("DOCKER")) else "127.0.0.1"
    
    server_address = (host, final_port)
    httpd = ThreadingHTTPServer(server_address, WebViewerHandler)
    url = f"http://localhost:{final_port}"

    print("\n" + "=" * 65)
    print("      🌐 3D VISION LAB — MULTI-ENGINE 3D VIEWER")
    print("=" * 65)
    print(f"  Server URL:       {url} (Host: {host})")
    print(f"  Engines:          WebGPU | TRELLIS.2 | InstantMesh | TripoSR")
    print(f"  Security:         Active (Rate Limiting + Concurrency Semaphore)")
    print(f"  Deployment:       {'Cloud / Railway Ready' if host == '0.0.0.0' else 'Local Development'}")
    print("=" * 65 + "\n", flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_server(port=port)
