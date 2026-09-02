"""
Realistic Multi-View 3D Renderer with Data Augmentation (Part 10).

Renders both clean studio views and realistic augmented views with:
- Camera randomization: random azimuth/elevation jitter, distance/zoom, FOV (focal length).
- Lighting randomization: key, fill, and rim lights with color temperature (warm, cool, sunset, neutral) and soft/hard intensity.
- Background randomization: studio white, neutral gray, soft indoor room gradient, outdoor sky-ground gradient, and transparent.
- Object randomization: subtle rotation jitter and position offset.

Embeds full camera extrinsics/intrinsics, lighting configuration, and background metadata per view.
"""

import os
import math
import json
import argparse
import random
import numpy as np
import trimesh
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple

import moderngl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "my_dataset")

# -------------------------------------------------------------
# Math / Camera Matrix Utilities
# -------------------------------------------------------------

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Computes 4x4 View Matrix (World to Camera)."""
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
    """Computes 4x4 Perspective Projection Matrix."""
    tan_half_fovy = math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 1.0 / (aspect * tan_half_fovy)
    m[1, 1] = 1.0 / tan_half_fovy
    m[2, 2] = -(zfar + znear) / (zfar - znear)
    m[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    m[3, 2] = -1.0
    return m

def spherical_to_cartesian(azimuth_deg: float, elevation_deg: float, radius: float) -> np.ndarray:
    """Converts spherical orbit coordinates to Cartesian coordinates."""
    az_rad = math.radians(azimuth_deg)
    el_rad = math.radians(elevation_deg)
    x = radius * math.cos(el_rad) * math.sin(az_rad)
    y = radius * math.sin(el_rad)
    z = radius * math.cos(el_rad) * math.cos(az_rad)
    return np.array([x, y, z], dtype=np.float32)

# -------------------------------------------------------------
# GLSL Shaders: Mesh Shading & Background Quad
# -------------------------------------------------------------

BG_VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.9999, 1.0);
}
"""

BG_FRAGMENT_SHADER = """
#version 330
uniform vec3 u_top_color;
uniform vec3 u_bottom_color;
uniform float u_opacity;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec3 col = mix(u_bottom_color, u_top_color, v_uv.y);
    fragColor = vec4(col, u_opacity);
}
"""

MESH_VERTEX_SHADER = """
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

MESH_FRAGMENT_SHADER = """
#version 330
uniform vec3 u_camera_pos;
uniform bool u_use_texture;
uniform sampler2D u_texture;

// Multi-light configuration (Key, Fill, Rim)
uniform vec3 u_light_pos1;
uniform vec3 u_light_color1;
uniform vec3 u_light_pos2;
uniform vec3 u_light_color2;
uniform vec3 u_light_pos3;
uniform vec3 u_light_color3;
uniform vec3 u_ambient_color;
uniform float u_roughness;

in vec3 v_normal;
in vec3 v_position;
in vec3 v_color;
in vec2 v_uv;

out vec4 fragColor;

vec3 compute_light(vec3 N, vec3 V, vec3 L_pos, vec3 L_color) {
    vec3 L = normalize(L_pos - v_position);
    vec3 H = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec_power = mix(64.0, 8.0, u_roughness);
    float spec = pow(max(dot(N, H), 0.0), spec_power) * (1.0 - u_roughness * 0.7);
    return L_color * (diff + spec * 0.3);
}

void main() {
    vec3 N = normalize(v_normal);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(u_camera_pos - v_position);

    vec4 base = vec4(v_color, 1.0);
    if (u_use_texture) {
        base = texture(u_texture, v_uv);
    }

    vec3 light_total = u_ambient_color;
    light_total += compute_light(N, V, u_light_pos1, u_light_color1);
    light_total += compute_light(N, V, u_light_pos2, u_light_color2);
    light_total += compute_light(N, V, u_light_pos3, u_light_color3);

    vec3 rgb = base.rgb * light_total;
    fragColor = vec4(rgb, base.a);
}
"""

# -------------------------------------------------------------
# Data Augmentation Samplers
# -------------------------------------------------------------

LIGHTING_PRESETS = {
    "studio_neutral": {
        "key_color": [1.0, 1.0, 1.0],
        "fill_color": [0.35, 0.35, 0.38],
        "rim_color": [0.20, 0.20, 0.25],
        "ambient": [0.35, 0.35, 0.35],
        "roughness": 0.4
    },
    "warm_indoor": {
        "key_color": [1.15, 0.95, 0.80],
        "fill_color": [0.40, 0.32, 0.28],
        "rim_color": [0.30, 0.25, 0.20],
        "ambient": [0.30, 0.28, 0.25],
        "roughness": 0.5
    },
    "cool_daylight": {
        "key_color": [0.90, 0.98, 1.15],
        "fill_color": [0.30, 0.35, 0.45],
        "rim_color": [0.25, 0.30, 0.40],
        "ambient": [0.32, 0.35, 0.38],
        "roughness": 0.35
    },
    "golden_sunset": {
        "key_color": [1.25, 0.75, 0.45],
        "fill_color": [0.45, 0.30, 0.50],
        "rim_color": [0.50, 0.35, 0.25],
        "ambient": [0.28, 0.22, 0.30],
        "roughness": 0.45
    },
    "harsh_direct": {
        "key_color": [1.40, 1.40, 1.35],
        "fill_color": [0.15, 0.15, 0.18],
        "rim_color": [0.10, 0.10, 0.15],
        "ambient": [0.20, 0.20, 0.20],
        "roughness": 0.25
    },
    "moody_dim": {
        "key_color": [0.65, 0.70, 0.75],
        "fill_color": [0.18, 0.20, 0.25],
        "rim_color": [0.35, 0.40, 0.50],
        "ambient": [0.18, 0.18, 0.20],
        "roughness": 0.6
    }
}

BACKGROUND_PRESETS = {
    "studio_white": {
        "top": [0.98, 0.98, 0.98],
        "bottom": [0.95, 0.95, 0.95],
        "opacity": 1.0
    },
    "neutral_gray": {
        "top": [0.75, 0.75, 0.75],
        "bottom": [0.68, 0.68, 0.68],
        "opacity": 1.0
    },
    "dark_studio": {
        "top": [0.18, 0.18, 0.20],
        "bottom": [0.10, 0.10, 0.12],
        "opacity": 1.0
    },
    "room_gradient": {
        "top": [0.82, 0.85, 0.88],    # Soft wall
        "bottom": [0.55, 0.50, 0.45], # Warm wooden floor
        "opacity": 1.0
    },
    "outdoor_sky": {
        "top": [0.60, 0.78, 0.95],    # Blue sky
        "bottom": [0.45, 0.55, 0.40], # Natural ground/grass
        "opacity": 1.0
    },
    "transparent": {
        "top": [0.0, 0.0, 0.0],
        "bottom": [0.0, 0.0, 0.0],
        "opacity": 0.0
    }
}

# -------------------------------------------------------------
# ModernGL MultiViewRenderer Class
# -------------------------------------------------------------

class MultiViewRenderer:
    def __init__(self, resolution: int = 512):
        self.resolution = resolution
        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        # Mesh shader program
        self.prog = self.ctx.program(
            vertex_shader=MESH_VERTEX_SHADER,
            fragment_shader=MESH_FRAGMENT_SHADER
        )

        # Background quad shader program
        self.bg_prog = self.ctx.program(
            vertex_shader=BG_VERTEX_SHADER,
            fragment_shader=BG_FRAGMENT_SHADER
        )

        # Full-screen background quad geometry
        quad_verts = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0
        ], dtype=np.float32)
        self.quad_vbo = self.ctx.buffer(quad_verts.tobytes())
        self.quad_vao = self.ctx.vertex_array(self.bg_prog, [(self.quad_vbo, '2f', 'in_pos')])

        # Offscreen Framebuffer
        self.color_tex = self.ctx.texture((resolution, resolution), 4)
        self.depth_tex = self.ctx.depth_texture((resolution, resolution))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_tex],
            depth_attachment=self.depth_tex
        )

    def prepare_mesh(self, glb_path: str) -> Optional[Tuple[moderngl.VertexArray, int, Optional[moderngl.Texture], np.ndarray, float]]:
        """Loads and normalizes 3D mesh into GPU buffers."""
        try:
            scene = trimesh.load(glb_path, force='scene')
            geom = scene.to_geometry()
            if not isinstance(geom, trimesh.Trimesh):
                mesh_list = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if not mesh_list:
                    return None
                geom = trimesh.util.concatenate(mesh_list)
        except Exception:
            return None

        if len(geom.vertices) == 0 or len(geom.faces) == 0:
            return None

        center = geom.bounding_box.centroid
        max_extent = float(np.max(geom.extents))
        scale = 1.6 / max(max_extent, 1e-6)

        vertices = (geom.vertices - center) * scale
        normals = geom.vertex_normals
        faces = geom.faces.astype(np.uint32)

        uvs = np.zeros((len(vertices), 2), dtype=np.float32)
        visual = getattr(geom, 'visual', None)
        if visual is not None and hasattr(visual, 'uv') and visual.uv is not None and len(visual.uv) == len(vertices):
            uvs = np.asarray(visual.uv, dtype=np.float32)

        colors = np.ones((len(vertices), 3), dtype=np.float32) * 0.85
        if visual is not None and hasattr(visual, 'vertex_colors') and visual.vertex_colors is not None and len(visual.vertex_colors) == len(vertices):
            vc = visual.vertex_colors
            if vc.shape[1] >= 3:
                colors = (vc[:, :3] / 255.0).astype(np.float32)

        gl_texture = None
        if visual is not None and hasattr(visual, 'material') and visual.material is not None:
            mat = visual.material
            tex_img = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
            if tex_img is not None and isinstance(tex_img, Image.Image):
                try:
                    tex_img_rgba = tex_img.convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
                    gl_texture = self.ctx.texture(tex_img_rgba.size, 4, tex_img_rgba.tobytes())
                    gl_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    gl_texture.build_mipmaps()
                except Exception:
                    gl_texture = None

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

        return vao, len(faces) * 3, gl_texture, center, max_extent

    def render_background(self, bg_preset: str):
        """Draws background quad or clears with transparency."""
        bg = BACKGROUND_PRESETS.get(bg_preset, BACKGROUND_PRESETS["studio_white"])
        if bg["opacity"] <= 0.0:
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            return

        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.bg_prog['u_top_color'].value = tuple(bg["top"])
        self.bg_prog['u_bottom_color'].value = tuple(bg["bottom"])
        self.bg_prog['u_opacity'].value = float(bg["opacity"])

        # Render background without writing depth
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def render_views(
        self,
        glb_path: str,
        output_dir: str,
        num_views: int = 12,
        elevations: List[float] = [20.0],
        base_distance: float = 2.4,
        base_fovy: float = 45.0,
        augment: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Renders multi-angle camera views.
        If augment=True, introduces realistic camera, lighting, and background randomization.
        """
        mesh_data = self.prepare_mesh(glb_path)
        if mesh_data is None:
            return []

        vao, index_count, gl_texture, orig_center, orig_extent = mesh_data
        os.makedirs(output_dir, exist_ok=True)

        use_texture = gl_texture is not None
        self.prog['u_use_texture'].value = use_texture
        if use_texture:
            gl_texture.use(location=0)
            self.prog['u_texture'].value = 0

        self.fbo.use()
        views_metadata = []

        azimuth_step = 360.0 / num_views
        view_idx = 0

        # Background choices pool for augmentation
        bg_choices = ["studio_white", "neutral_gray", "room_gradient", "outdoor_sky", "dark_studio"]
        lighting_choices = list(LIGHTING_PRESETS.keys())

        for elevation in elevations:
            for i in range(num_views):
                # 1. Camera angles & randomization
                nominal_azimuth = i * azimuth_step
                nominal_elevation = elevation

                if augment:
                    # Randomize azimuth jitter (+- 5 deg) & elevation jitter (+- 8 deg)
                    azimuth = nominal_azimuth + random.uniform(-5.0, 5.0)
                    elev = max(-10.0, min(80.0, nominal_elevation + random.uniform(-8.0, 8.0)))
                    # Randomize distance / zoom (focal length variation)
                    distance = base_distance * random.uniform(0.85, 1.25)
                    fovy = base_fovy * random.uniform(0.85, 1.20)
                    # Randomize object translation offset (slight uncentering)
                    target = np.array([
                        random.uniform(-0.06, 0.06),
                        random.uniform(-0.06, 0.06),
                        random.uniform(-0.06, 0.06)
                    ], dtype=np.float32)
                    bg_type = random.choice(bg_choices)
                    lighting_type = random.choice(lighting_choices)
                else:
                    azimuth = nominal_azimuth
                    elev = nominal_elevation
                    distance = base_distance
                    fovy = base_fovy
                    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    bg_type = "studio_white"
                    lighting_type = "studio_neutral"

                eye = spherical_to_cartesian(azimuth, elev, distance) + target
                up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

                view = look_at(eye, target, up)
                proj = perspective(fovy, 1.0, 0.1, 50.0)
                mvp = proj @ view

                # 2. Lighting setup
                cfg = LIGHTING_PRESETS[lighting_type]
                # Key light: high front-right
                lpos1 = eye + np.array([0.6, 1.5, 0.4], dtype=np.float32)
                # Fill light: opposing side
                lpos2 = eye + np.array([-0.8, 0.5, -0.3], dtype=np.float32)
                # Rim light: from behind object
                lpos3 = -eye * 0.7 + np.array([0.0, 1.0, 0.0], dtype=np.float32)

                self.prog['u_light_pos1'].value = tuple(lpos1)
                self.prog['u_light_color1'].value = tuple(cfg["key_color"])
                self.prog['u_light_pos2'].value = tuple(lpos2)
                self.prog['u_light_color2'].value = tuple(cfg["fill_color"])
                self.prog['u_light_pos3'].value = tuple(lpos3)
                self.prog['u_light_color3'].value = tuple(cfg["rim_color"])
                self.prog['u_ambient_color'].value = tuple(cfg["ambient"])
                self.prog['u_roughness'].value = float(cfg["roughness"])

                self.prog['u_mvp'].write(mvp.T.tobytes())
                self.prog['u_model'].write(np.eye(4, dtype=np.float32).tobytes())
                self.prog['u_camera_pos'].value = tuple(eye)

                # 3. Render Background
                self.render_background(bg_type)

                # 4. Render 3D Mesh
                vao.render()

                # 5. Read back pixels
                raw_bytes = self.fbo.read(components=4)
                img = Image.frombytes('RGBA', (self.resolution, self.resolution), raw_bytes)
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

                img_filename = f"view_{view_idx:03d}.png"
                img_path = os.path.join(output_dir, img_filename)
                img.save(img_path, format="PNG")

                c2w = np.linalg.inv(view)

                view_info = {
                    "index": view_idx,
                    "file": img_filename,
                    "azimuth_deg": round(float(azimuth), 2),
                    "elevation_deg": round(float(elev), 2),
                    "distance": round(float(distance), 3),
                    "fov_deg": round(float(fovy), 1),
                    "camera_pos": [round(float(x), 4) for x in eye],
                    "camera_lookat": [round(float(x), 4) for x in target],
                    "lighting_preset": lighting_type,
                    "background_preset": bg_type,
                    "is_augmented": augment,
                    "world_to_camera_matrix": [[round(float(x), 5) for x in row] for row in view],
                    "camera_to_world_matrix": [[round(float(x), 5) for x in row] for row in c2w]
                }
                views_metadata.append(view_info)
                view_idx += 1

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
    augment: bool = False,
    limit: Optional[int] = None
) -> None:
    objects_dir = os.path.join(dataset_dir, "objects")
    master_metadata_path = os.path.join(dataset_dir, "metadata.json")

    if not os.path.exists(objects_dir):
        print(f"Error: Objects directory {objects_dir} not found.")
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

    aug_str = "ENABLED (Realistic lighting, cameras, & backgrounds)" if augment else "CLEAN (Studio uniform)"
    print(f"\n=======================================================", flush=True)
    print(f"       MULTI-VIEW 3D OBJECT RENDERING (GPU)", flush=True)
    print(f"=======================================================", flush=True)
    print(f"Objects to process: {len(object_folders)}", flush=True)
    print(f"Views per object:   {num_views * len(elevations)}", flush=True)
    print(f"Resolution:         {resolution}x{resolution}", flush=True)
    print(f"Augmentation:       {aug_str}", flush=True)
    print(f"Only Valid:         {only_valid}", flush=True)
    print(f"=======================================================\n", flush=True)

    renderer = MultiViewRenderer(resolution=resolution)

    rendered_count = 0
    skipped_count = 0

    for i, obj_id in enumerate(object_folders, start=1):
        obj_dir = os.path.join(objects_dir, obj_id)
        glb_path = os.path.join(obj_dir, "model.glb")
        obj_meta_path = os.path.join(obj_dir, "metadata.json")

        obj_meta: Dict[str, Any] = {}
        if os.path.exists(obj_meta_path):
            with open(obj_meta_path, "r", encoding="utf-8") as f:
                obj_meta = json.load(f)

        val_status = obj_meta.get("validation", {}).get("status")
        if only_valid and val_status == "FAIL":
            print(f"[{i}/{len(object_folders)}] ⏭️  Skipping {obj_id} (validation FAIL)", flush=True)
            skipped_count += 1
            continue

        print(f"[{i}/{len(object_folders)}] 📸 Rendering {obj_id} ...", end="", flush=True)

        views = renderer.render_views(
            glb_path=glb_path,
            output_dir=obj_dir,
            num_views=num_views,
            elevations=elevations,
            augment=augment
        )

        if views:
            obj_meta["views"] = views
            obj_meta["num_rendered_views"] = len(views)
            obj_meta["is_augmented"] = augment
            with open(obj_meta_path, "w", encoding="utf-8") as f:
                json.dump(obj_meta, f, indent=2)

            if obj_id in master_metadata:
                master_metadata[obj_id]["num_rendered_views"] = len(views)
                master_metadata[obj_id]["is_augmented"] = augment

            rendered_count += 1
            print(f" Done ({len(views)} views saved)", flush=True)
        else:
            print(f" Failed to render (mesh could not be processed)", flush=True)
            skipped_count += 1

    with open(master_metadata_path, "w", encoding="utf-8") as f:
        json.dump(master_metadata, f, indent=2)

    print(f"\n=======================================================", flush=True)
    print(f"                 RENDERING COMPLETE", flush=True)
    print(f"=======================================================", flush=True)
    print(f"  Successfully Rendered: {rendered_count} objects", flush=True)
    print(f"  Total Images Generated: {rendered_count * num_views * len(elevations)}", flush=True)
    print(f"  Skipped / Failed:      {skipped_count} objects", flush=True)
    print(f"  Output Directory:      {objects_dir}", flush=True)
    print(f"=======================================================\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render 3D objects from multiple angles")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Dataset directory")
    parser.add_argument("--num-views", type=int, default=12, help="Number of azimuth angles per elevation (default: 12)")
    parser.add_argument("--elevations", type=float, nargs="+", default=[20.0], help="Elevation angle(s) in degrees")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution in pixels (default: 512)")
    parser.add_argument("--augment", action="store_true", help="Enable realistic data augmentation (lighting, camera, bg)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of objects to render")
    parser.add_argument("--all", action="store_true", help="Render all models including ones with validation warnings")
    args = parser.parse_args()

    render_dataset(
        dataset_dir=args.dataset_dir,
        num_views=args.num_views,
        elevations=args.elevations,
        resolution=args.resolution,
        augment=args.augment,
        only_valid=not args.all,
        limit=args.limit
    )
