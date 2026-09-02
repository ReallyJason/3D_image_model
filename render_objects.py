"""
Multi-Angle 3D Object Renderer using Headless ModernGL on Apple Silicon GPU.

Renders high-quality multi-view images from orbit cameras around each 3D object:

object_00001/
├── model.glb
├── view_000.png
├── view_001.png
├── view_002.png
├── ...
└── metadata.json

Each rendered view includes exact camera intrinsics/extrinsics recorded in metadata.json.
"""

import os
import math
import json
import argparse
import numpy as np
import trimesh
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple

import moderngl

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_DIR, "my_dataset")

# -------------------------------------------------------------
# Math / Camera Matrix Utilities
# -------------------------------------------------------------

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Computes a 4x4 View matrix (World to Camera)."""
    f = target - eye
    fnorm = np.linalg.norm(f)
    f = f / (fnorm if fnorm > 1e-8 else 1.0)

    unorm = np.linalg.norm(up)
    u = up / (unorm if unorm > 1e-8 else 1.0)

    s = np.cross(f, u)
    snorm = np.linalg.norm(s)
    s = s / (snorm if snorm > 1e-8 else 1.0)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m

def perspective(fovy_deg: float, aspect: float, znear: float = 0.1, zfar: float = 100.0) -> np.ndarray:
    """Computes a 4x4 Perspective Projection matrix."""
    tan_half_fovy = math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 1.0 / (aspect * tan_half_fovy)
    m[1, 1] = 1.0 / tan_half_fovy
    m[2, 2] = -(zfar + znear) / (zfar - znear)
    m[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    m[3, 2] = -1.0
    return m

def spherical_to_cartesian(azimuth_deg: float, elevation_deg: float, radius: float) -> np.ndarray:
    """Converts spherical orbit coords to 3D Cartesian coordinates."""
    az_rad = math.radians(azimuth_deg)
    el_rad = math.radians(elevation_deg)
    x = radius * math.cos(el_rad) * math.sin(az_rad)
    y = radius * math.sin(el_rad)
    z = radius * math.cos(el_rad) * math.cos(az_rad)
    return np.array([x, y, z], dtype=np.float32)

# -------------------------------------------------------------
# GLSL Shaders
# -------------------------------------------------------------

VERTEX_SHADER = """
#version 330
uniform mat4 u_mvp;
uniform mat4 u_model;

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
in vec2 in_uv;

out vec3 v_normal;
out vec3 v_position;
out vec3 v_color;
out vec2 v_uv;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_position = vec3(u_model * vec4(in_position, 1.0));
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_color = in_color;
    v_uv = in_uv;
}
"""

FRAGMENT_SHADER = """
#version 330
uniform vec3 u_light_pos;
uniform vec3 u_camera_pos;
uniform bool u_use_texture;
uniform sampler2D u_texture;

in vec3 v_normal;
in vec3 v_position;
in vec3 v_color;
in vec2 v_uv;

out vec4 fragColor;

void main() {
    vec3 N = normalize(v_normal);
    if (!gl_FrontFacing) N = -N;

    vec3 L1 = normalize(u_light_pos - v_position);
    vec3 V = normalize(u_camera_pos - v_position);
    vec3 H1 = normalize(L1 + V);

    // Key Light
    float ambient = 0.38;
    float diff1 = max(dot(N, L1), 0.0) * 0.55;
    float spec1 = pow(max(dot(N, H1), 0.0), 32.0) * 0.12;

    // Fill Light from opposing angle
    vec3 L2 = normalize(vec3(-L1.x, 0.4, -L1.z));
    float diff2 = max(dot(N, L2), 0.0) * 0.20;

    // Base color from texture or vertex color or neutral studio gray
    vec4 base = vec4(v_color, 1.0);
    if (u_use_texture) {
        vec4 tex = texture(u_texture, v_uv);
        base = tex;
    }

    vec3 rgb = base.rgb * (ambient + diff1 + diff2) + vec3(spec1);
    fragColor = vec4(rgb, base.a);
}
"""

# -------------------------------------------------------------
# ModernGL Renderer Class
# -------------------------------------------------------------

class MultiViewRenderer:
    def __init__(self, resolution: int = 512):
        self.resolution = resolution
        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)  # Double-sided rendering

        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )

        # Offscreen Framebuffer
        self.color_tex = self.ctx.texture((resolution, resolution), 4)
        self.depth_tex = self.ctx.depth_texture((resolution, resolution))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_tex],
            depth_attachment=self.depth_tex
        )

    def prepare_mesh(self, glb_path: str) -> Optional[Tuple[moderngl.VertexArray, int, Optional[moderngl.Texture]]]:
        """Loads and normalizes a 3D model into GPU buffers."""
        try:
            scene = trimesh.load(glb_path, force='scene')
            geom = scene.to_geometry()
            if not isinstance(geom, trimesh.Trimesh):
                mesh_list = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if not mesh_list:
                    return None
                geom = trimesh.util.concatenate(mesh_list)
        except Exception as e:
            print(f"Error loading {glb_path}: {e}")
            return None

        if len(geom.vertices) == 0 or len(geom.faces) == 0:
            return None

        # 1. Normalize object geometry: center at (0, 0, 0) and scale into unit sphere
        center = geom.bounding_box.centroid
        max_extent = float(np.max(geom.extents))
        scale = 1.6 / max(max_extent, 1e-6)

        vertices = (geom.vertices - center) * scale
        normals = geom.vertex_normals
        faces = geom.faces.astype(np.uint32)

        # 2. Extract UVs
        has_uv = False
        uvs = np.zeros((len(vertices), 2), dtype=np.float32)
        visual = getattr(geom, 'visual', None)
        if visual is not None and hasattr(visual, 'uv') and visual.uv is not None and len(visual.uv) == len(vertices):
            uvs = np.asarray(visual.uv, dtype=np.float32)
            has_uv = True

        # 3. Extract Vertex Colors
        has_vc = False
        colors = np.ones((len(vertices), 3), dtype=np.float32) * 0.85  # Clean clay gray default
        if visual is not None and hasattr(visual, 'vertex_colors') and visual.vertex_colors is not None and len(visual.vertex_colors) == len(vertices):
            vc = visual.vertex_colors
            if vc.shape[1] >= 3:
                colors = (vc[:, :3] / 255.0).astype(np.float32)
                has_vc = True

        # 4. Extract Diffuse Texture (if any)
        gl_texture = None
        if visual is not None and hasattr(visual, 'material') and visual.material is not None:
            mat = visual.material
            tex_img = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
            if tex_img is not None and isinstance(tex_img, Image.Image):
                try:
                    tex_img_rgba = tex_img.convert("RGBA")
                    # Flip vertical for OpenGL UV coordinates
                    tex_img_rgba = tex_img_rgba.transpose(Image.FLIP_TOP_BOTTOM)
                    gl_texture = self.ctx.texture(tex_img_rgba.size, 4, tex_img_rgba.tobytes())
                    gl_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    gl_texture.build_mipmaps()
                except Exception as e:
                    gl_texture = None

        # Pack vertex data: position (3f), normal (3f), color (3f), uv (2f) = 11 floats
        v_data = np.hstack([
            vertices.astype(np.float32),
            normals.astype(np.float32),
            colors.astype(np.float32),
            uvs.astype(np.float32)
        ]).flatten()

        vbo = self.ctx.buffer(v_data.tobytes())
        ibo = self.ctx.buffer(faces.tobytes())

        vao = self.ctx.vertex_array(
            self.prog,
            [(vbo, '3f 3f 3f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv')],
            index_buffer=ibo,
            index_element_size=4
        )

        return vao, len(faces) * 3, gl_texture

    def render_views(
        self,
        glb_path: str,
        output_dir: str,
        num_views: int = 12,
        elevations: List[float] = [20.0],
        distance: float = 2.4,
        fovy: float = 45.0,
        bg_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.0)  # RGBA (transparent bg)
    ) -> List[Dict[str, Any]]:
        """
        Renders multi-view images around the object and returns list of camera poses.
        """
        mesh_data = self.prepare_mesh(glb_path)
        if mesh_data is None:
            return []

        vao, index_count, gl_texture = mesh_data
        os.makedirs(output_dir, exist_ok=True)

        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        proj = perspective(fovy, 1.0, 0.1, 50.0)

        # Setup texture binding
        use_texture = gl_texture is not None
        self.prog['u_use_texture'].value = use_texture
        if use_texture:
            gl_texture.use(location=0)
            self.prog['u_texture'].value = 0

        self.fbo.use()
        views_metadata = []

        azimuth_step = 360.0 / num_views
        view_idx = 0

        for elevation in elevations:
            for i in range(num_views):
                azimuth = i * azimuth_step
                eye = spherical_to_cartesian(azimuth, elevation, distance)

                view = look_at(eye, target, up)
                mvp = proj @ view

                # Pass uniforms
                self.prog['u_mvp'].write(mvp.T.tobytes())
                self.prog['u_model'].write(np.eye(4, dtype=np.float32).tobytes())
                self.prog['u_camera_pos'].value = tuple(eye)
                # Light placed slightly offset from camera
                light_pos = eye + np.array([0.5, 1.5, 0.5], dtype=np.float32)
                self.prog['u_light_pos'].value = tuple(light_pos)

                # Clear and render
                self.ctx.clear(*bg_color)
                vao.render()

                # Read image buffer
                raw_bytes = self.fbo.read(components=4)
                img = Image.frombytes('RGBA', (self.resolution, self.resolution), raw_bytes)
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

                img_filename = f"view_{view_idx:03d}.png"
                img_path = os.path.join(output_dir, img_filename)
                img.save(img_path, format="PNG")

                # Compute camera-to-world (extrinsic inverse)
                c2w = np.linalg.inv(view)

                view_info = {
                    "index": view_idx,
                    "file": img_filename,
                    "azimuth_deg": round(azimuth, 2),
                    "elevation_deg": round(elevation, 2),
                    "distance": round(distance, 3),
                    "fov_deg": round(fovy, 1),
                    "camera_pos": [round(float(x), 4) for x in eye],
                    "camera_lookat": [0.0, 0.0, 0.0],
                    "camera_up": [0.0, 1.0, 0.0],
                    "world_to_camera_matrix": [[round(float(x), 5) for x in row] for row in view],
                    "camera_to_world_matrix": [[round(float(x), 5) for x in row] for row in c2w]
                }
                views_metadata.append(view_info)
                view_idx += 1

        # Release temporary GPU resources for this mesh
        vao.release()
        if gl_texture is not None:
            gl_texture.release()

        return views_metadata

def render_dataset(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    num_views: int = 12,
    elevations: List[float] = [20.0],
    resolution: int = 512,
    only_valid: bool = True,
    limit: Optional[int] = None
) -> None:
    """
    Renders multi-view images for all models in dataset_dir/objects/.
    """
    objects_dir = os.path.join(dataset_dir, "objects")
    master_metadata_path = os.path.join(dataset_dir, "metadata.json")

    if not os.path.exists(objects_dir):
        print(f"Error: Objects directory {objects_dir} not found. Run organize first.")
        return

    master_metadata: Dict[str, Any] = {}
    if os.path.exists(master_metadata_path):
        with open(master_metadata_path, "r", encoding="utf-8") as f:
            master_metadata = json.load(f)

    object_folders = sorted([
        f for f in os.listdir(objects_dir)
        if os.path.isdir(os.path.join(objects_dir, f)) and not f.startswith('.')
    ])

    if limit is not None:
        object_folders = object_folders[:limit]

    print(f"\n=======================================================")
    print(f"       MULTI-VIEW 3D OBJECT RENDERING (GPU)")
    print(f"=======================================================")
    print(f"Objects to process: {len(object_folders)}")
    print(f"Views per object:   {num_views * len(elevations)} ({num_views} azims x {len(elevations)} elevs)")
    print(f"Resolution:         {resolution}x{resolution}")
    print(f"Only Valid:         {only_valid}")
    print(f"=======================================================\n")

    renderer = MultiViewRenderer(resolution=resolution)

    rendered_count = 0
    skipped_count = 0

    for i, obj_id in enumerate(object_folders, start=1):
        obj_dir = os.path.join(objects_dir, obj_id)
        glb_path = os.path.join(obj_dir, "model.glb")
        obj_meta_path = os.path.join(obj_dir, "metadata.json")

        # Check validation status if available
        obj_meta: Dict[str, Any] = {}
        if os.path.exists(obj_meta_path):
            with open(obj_meta_path, "r", encoding="utf-8") as f:
                obj_meta = json.load(f)

        val_status = obj_meta.get("validation", {}).get("status")
        if only_valid and val_status == "FAIL":
            print(f"[{i}/{len(object_folders)}] ⏭️  Skipping {obj_id} (validation FAIL)")
            skipped_count += 1
            continue

        print(f"[{i}/{len(object_folders)}] 📸 Rendering {obj_id} ...", end="", flush=True)

        views = renderer.render_views(
            glb_path=glb_path,
            output_dir=obj_dir,
            num_views=num_views,
            elevations=elevations
        )

        if views:
            obj_meta["views"] = views
            obj_meta["num_rendered_views"] = len(views)
            with open(obj_meta_path, "w", encoding="utf-8") as f:
                json.dump(obj_meta, f, indent=2)

            if obj_id in master_metadata:
                master_metadata[obj_id]["num_rendered_views"] = len(views)

            rendered_count += 1
            print(f" Done ({len(views)} views saved)")
        else:
            print(f" Failed to render (mesh could not be processed)")
            skipped_count += 1

    # Update master metadata
    with open(master_metadata_path, "w", encoding="utf-8") as f:
        json.dump(master_metadata, f, indent=2)

    print(f"\n=======================================================")
    print(f"                 RENDERING COMPLETE")
    print(f"=======================================================")
    print(f"  Successfully Rendered: {rendered_count} objects")
    print(f"  Total Images Generated: {rendered_count * num_views * len(elevations)}")
    print(f"  Skipped / Failed:      {skipped_count} objects")
    print(f"  Output Directory:      {objects_dir}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render 3D objects from multiple angles")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Dataset directory")
    parser.add_argument("--num-views", type=int, default=12, help="Number of azimuth angles per elevation (default: 12)")
    parser.add_argument("--elevations", type=float, nargs="+", default=[20.0], help="Elevation angle(s) in degrees (default: 20.0)")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution in pixels (default: 512)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of objects to render")
    parser.add_argument("--all", action="store_true", help="Render all models including ones with validation warnings")
    args = parser.parse_args()

    render_dataset(
        dataset_dir=args.dataset_dir,
        num_views=args.num_views,
        elevations=args.elevations,
        resolution=args.resolution,
        only_valid=not args.all,
        limit=args.limit
    )
