/**
 * 3D Vision Lab — Interactive Web Viewer Client with Foundation Model Support.
 * Renders Photorealistic Colored GLB & OBJ Meshes via Three.js.
 */

// -------------------------------------------------------------
// State Management
// -------------------------------------------------------------
let currentImageBase64 = null;
let currentImageUrl = null;
let currentMesh = null;
let autoRotate = true;
let isWireframe = false;

// -------------------------------------------------------------
// DOM Elements
// -------------------------------------------------------------
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const dropzoneEmpty = document.getElementById('dropzoneEmpty');
const dropzonePreview = document.getElementById('dropzonePreview');
const previewImage = document.getElementById('previewImage');
const clearImageBtn = document.getElementById('clearImageBtn');
const samplesList = document.getElementById('samplesList');

const engineSelect = document.getElementById('engineSelect');
const triposrControls = document.getElementById('triposrControls');
const voxelControls = document.getElementById('voxelControls');
const resolutionSelect = document.getElementById('resolutionSelect');
const removeBgCheck = document.getElementById('removeBgCheck');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');

const generateBtn = document.getElementById('generateBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingStatusText = document.getElementById('loadingStatusText');
const meshStatus = document.getElementById('meshStatus');
const activeModelBadge = document.getElementById('activeModelBadge');

const rotateBtn = document.getElementById('rotateBtn');
const wireframeBtn = document.getElementById('wireframeBtn');
const resetCamBtn = document.getElementById('resetCamBtn');
const downloadBar = document.getElementById('downloadBar');
const downloadObjBtn = document.getElementById('downloadObjBtn');
const downloadGlbBtn = document.getElementById('downloadGlbBtn');

// Telemetry Elements
const inferenceMetric = document.getElementById('inferenceMetric');
const mcMetric = document.getElementById('mcMetric');
const verticesMetric = document.getElementById('verticesMetric');
const facesMetric = document.getElementById('facesMetric');

const hfTokenRow = document.getElementById('hfTokenRow');
const hfTokenInput = document.getElementById('hfTokenInput');

const webgpuControls = document.getElementById('webgpuControls');
const webgpuDepthSlider = document.getElementById('webgpuDepthSlider');
const webgpuDepthValue = document.getElementById('webgpuDepthValue');
const webgpuResSelect = document.getElementById('webgpuResSelect');

// Warning Modal Elements
const warningModal = document.getElementById('warningModal');
const dontRemindCheck = document.getElementById('dontRemindCheck');
const cancelModalBtn = document.getElementById('cancelModalBtn');
const confirmModalBtn = document.getElementById('confirmModalBtn');

// -------------------------------------------------------------
// Accessibility & Toast Notification System
// -------------------------------------------------------------
function announceA11y(text) {
  const announcer = document.getElementById('a11yAnnouncer');
  if (announcer) {
    announcer.textContent = text;
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}

function showToast(title, message, type = 'info', duration = 4500) {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const icons = {
    success: '✅',
    error: '❌',
    warning: '⚠️',
    info: 'ℹ️'
  };

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.setAttribute('role', 'alert');

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
    <div class="toast-body">
      <div class="toast-title">${escapeHtml(title)}</div>
      <div class="toast-msg">${escapeHtml(message)}</div>
    </div>
    <button class="toast-close" aria-label="Close notification">&times;</button>
  `;

  const closeBtn = toast.querySelector('.toast-close');
  const removeToast = () => {
    toast.classList.add('toast-exit');
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 260);
  };

  closeBtn.addEventListener('click', removeToast);
  container.appendChild(toast);

  if (duration > 0) {
    setTimeout(removeToast, duration);
  }
}

if (webgpuDepthSlider && webgpuDepthValue) {
  webgpuDepthSlider.addEventListener('input', (e) => {
    webgpuDepthValue.textContent = Number(e.target.value).toFixed(2);
  });
}

// -------------------------------------------------------------
// Engine Switcher UI Logic
// -------------------------------------------------------------
engineSelect.addEventListener('change', () => {
  const engine = engineSelect.value;
  if (engine === 'client_webgpu') {
    if (webgpuControls) webgpuControls.classList.remove('hidden');
    if (hfTokenRow) hfTokenRow.classList.add('hidden');
    triposrControls.classList.add('hidden');
    voxelControls.classList.add('hidden');
    activeModelBadge.textContent = 'Engine: In-Browser WebGPU (0 Host Cost)';
  } else if (engine === 'trellis') {
    if (webgpuControls) webgpuControls.classList.add('hidden');
    if (hfTokenRow) hfTokenRow.classList.remove('hidden');
    triposrControls.classList.add('hidden');
    voxelControls.classList.add('hidden');
    activeModelBadge.textContent = 'Engine: Microsoft TRELLIS.2 (SOTA Cloud)';
  } else if (engine === 'instantmesh') {
    if (webgpuControls) webgpuControls.classList.add('hidden');
    if (hfTokenRow) hfTokenRow.classList.remove('hidden');
    triposrControls.classList.add('hidden');
    voxelControls.classList.add('hidden');
    activeModelBadge.textContent = 'Engine: Tencent InstantMesh';
  } else if (engine === 'triposr') {
    if (webgpuControls) webgpuControls.classList.add('hidden');
    if (hfTokenRow) hfTokenRow.classList.add('hidden');
    triposrControls.classList.remove('hidden');
    voxelControls.classList.add('hidden');
    activeModelBadge.textContent = 'Engine: TripoSR (Local Mac GPU)';
  } else {
    if (webgpuControls) webgpuControls.classList.add('hidden');
    if (hfTokenRow) hfTokenRow.classList.add('hidden');
    triposrControls.classList.add('hidden');
    voxelControls.classList.remove('hidden');
    activeModelBadge.textContent = 'Engine: TinyImageToVoxelNet (Custom Baseline)';
  }
  if (currentImageBase64 || currentImageUrl) {
    checkAndTriggerReconstruction();
  }
});

// -------------------------------------------------------------
// Three.js 3D Viewport Setup
// -------------------------------------------------------------
const canvas = document.getElementById('canvas3d');
const container = document.getElementById('viewportContainer');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x08090d);

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.05, 100);
camera.position.set(1.8, 1.4, 2.2);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.25;
renderer.outputEncoding = THREE.sRGBEncoding;

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = autoRotate;
controls.autoRotateSpeed = 2.0;

// Studio 3-Point Lighting Rig
const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0xfffaed, 1.4);
keyLight.position.set(4, 6, 3);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x8cb4ff, 0.7);
fillLight.position.set(-4, 3, -3);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0x6366f1, 0.9);
rimLight.position.set(0, -4, -4);
scene.add(rimLight);

// Circular Ground Grid
const gridHelper = new THREE.GridHelper(4, 24, 0x6366f1, 0x1a2133);
gridHelper.position.y = -0.5;
scene.add(gridHelper);

// Placeholder Geometry
const placeholderGeom = new THREE.IcosahedronGeometry(0.45, 1);
const placeholderMat = new THREE.MeshStandardMaterial({
  color: 0x272f44,
  roughness: 0.4,
  metalness: 0.2,
  wireframe: true
});
const placeholderMesh = new THREE.Mesh(placeholderGeom, placeholderMat);
scene.add(placeholderMesh);

// -------------------------------------------------------------
// Render Loop
// -------------------------------------------------------------
function animate() {
  requestAnimationFrame(animate);
  controls.autoRotate = autoRotate;
  controls.update();

  if (placeholderMesh && placeholderMesh.visible) {
    placeholderMesh.rotation.y += 0.008;
  }

  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

// -------------------------------------------------------------
// Image Input Handling
// -------------------------------------------------------------
dropzone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFile(e.target.files[0]);
  }
});

['dragenter', 'dragover'].forEach(eventName => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(eventName => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', (e) => {
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleFile(e.dataTransfer.files[0]);
  }
});

window.addEventListener('paste', (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (let item of items) {
    if (item.type.startsWith('image/')) {
      handleFile(item.getAsFile());
      break;
    }
  }
});

function handleFile(file) {
  if (!file) return;

  // Maximum upload size validation (12 MB)
  const maxBytes = 12 * 1024 * 1024;
  if (file.size > maxBytes) {
    showToast('File Too Large', `Image is ${(file.size / (1024 * 1024)).toFixed(1)}MB. Maximum allowed is 12MB.`, 'error', 5000);
    announceA11y('Error: Selected image exceeds 12MB limit.');
    return;
  }

  const validMimes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
  if (!validMimes.includes(file.type.toLowerCase()) && !file.type.startsWith('image/')) {
    showToast('Unsupported Format', 'Please upload a PNG, JPG, or WEBP image.', 'warning', 4000);
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    setImageSource(e.target.result, null);
    showToast('Image Loaded', 'Ready for 3D neural reconstruction.', 'success', 2500);
    announceA11y('Image loaded successfully.');
  };
  reader.onerror = () => {
    showToast('Read Error', 'Could not read the selected image file.', 'error');
  };
  reader.readAsDataURL(file);
}

function setImageSource(base64Data, url) {
  currentImageBase64 = base64Data;
  currentImageUrl = url;

  previewImage.src = base64Data || url;
  dropzoneEmpty.classList.add('hidden');
  dropzonePreview.classList.remove('hidden');
  generateBtn.disabled = false;

  meshStatus.textContent = 'Image loaded. Ready to reconstruct.';
  checkAndTriggerReconstruction();
}

clearImageBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  currentImageBase64 = null;
  currentImageUrl = null;
  fileInput.value = '';
  previewImage.src = '';
  dropzoneEmpty.classList.remove('hidden');
  dropzonePreview.classList.add('hidden');
  generateBtn.disabled = true;
  document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('active'));
});

// -------------------------------------------------------------
// Sample Gallery
// -------------------------------------------------------------
async function loadSampleGallery() {
  try {
    const res = await fetch('/api/samples');
    const samples = await res.json();
    samplesList.innerHTML = '';

    samples.forEach(sample => {
      const item = document.createElement('div');
      item.className = 'sample-item';
      item.title = `${sample.name} (${sample.category})`;

      const img = document.createElement('img');
      img.src = sample.url;
      img.alt = sample.name;

      const name = document.createElement('span');
      name.className = 'sample-name';
      name.textContent = sample.category;

      item.appendChild(img);
      item.appendChild(name);

      item.addEventListener('click', () => {
        document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('active'));
        item.classList.add('active');
        setImageSource(null, sample.url);
      });

      samplesList.appendChild(item);
    });
  } catch (err) {
    console.warn('Could not fetch sample gallery:', err);
  }
}
loadSampleGallery();

thresholdSlider.addEventListener('input', (e) => {
  thresholdValue.textContent = Number(e.target.value).toFixed(2);
});

// -------------------------------------------------------------
// Warning Modal & Reconstruction Trigger Logic
// -------------------------------------------------------------
function checkAndTriggerReconstruction() {
  if (!currentImageBase64 && !currentImageUrl) return;

  const engine = engineSelect.value;
  const isSuppressed = localStorage.getItem('suppress_webgpu_warning') === 'true';

  if (engine === 'client_webgpu' && !isSuppressed && warningModal) {
    warningModal.classList.remove('hidden');
  } else {
    triggerReconstruction();
  }
}

if (cancelModalBtn) {
  cancelModalBtn.addEventListener('click', () => {
    warningModal.classList.add('hidden');
  });
}

if (confirmModalBtn) {
  confirmModalBtn.addEventListener('click', () => {
    if (dontRemindCheck && dontRemindCheck.checked) {
      localStorage.setItem('suppress_webgpu_warning', 'true');
    }
    warningModal.classList.add('hidden');
    triggerReconstruction();
  });
}

generateBtn.addEventListener('click', checkAndTriggerReconstruction);

async function triggerReconstruction() {
  if (!currentImageBase64 && !currentImageUrl) return;

  const engine = engineSelect.value;
  if (engine === 'client_webgpu') {
    await executeInBrowserWebGPU();
    return;
  }

  if (engine === 'trellis') {
    loadingStatusText.textContent = 'Microsoft TRELLIS.2 Synthesizing 3D Flow Model...';
  } else if (engine === 'instantmesh') {
    loadingStatusText.textContent = 'InstantMesh Multi-View Diffusion & FlexiCubes...';
  } else if (engine === 'triposr') {
    loadingStatusText.textContent = 'TripoSR Local Neural Engine Inferring on Apple Silicon...';
  } else {
    loadingStatusText.textContent = 'TinyImageToVoxelNet Predicting Voxel Occupancy...';
  }

  loadingOverlay.classList.remove('hidden');
  generateBtn.disabled = true;
  meshStatus.textContent = `Running ${engineSelect.options[engineSelect.selectedIndex].text}...`;

  const payload = {
    engine: engine,
    hf_token: hfTokenInput ? hfTokenInput.value.trim() : '',
    resolution: parseInt(resolutionSelect.value) || 192,
    remove_bg: removeBgCheck.checked,
    threshold: parseFloat(thresholdSlider.value)
  };

  if (currentImageBase64) {
    payload.image_base64 = currentImageBase64;
  } else if (currentImageUrl) {
    payload.image_url = currentImageUrl;
  }

  try {
    const res = await fetch('/api/reconstruct', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (!data.success) {
      throw new Error(data.error || 'Reconstruction failed');
    }

    // Telemetry Update
    inferenceMetric.textContent = `${data.inference_ms} ms`;
    mcMetric.textContent = `${data.total_latency_ms} ms`;
    verticesMetric.textContent = Number(data.vertices).toLocaleString();
    facesMetric.textContent = Number(data.faces).toLocaleString();

    meshStatus.textContent = `${data.engine} • ${data.vertices.toLocaleString()} verts, ${data.faces.toLocaleString()} faces (${data.total_latency_ms} ms)`;

    // Prefer Colored GLB if available, else OBJ
    if (data.glb_url) {
      loadGlbMesh(data.glb_url, data.obj_url);
    } else {
      loadObjMesh(data.obj_url);
    }

    // Download Links
    downloadObjBtn.href = `${data.obj_url}?download=1`;
    if (data.glb_url) {
      downloadGlbBtn.href = `${data.glb_url}?download=1`;
      downloadGlbBtn.classList.remove('hidden');
    } else {
      downloadGlbBtn.classList.add('hidden');
    }
    downloadBar.classList.remove('hidden');

  } catch (err) {
    console.error('Reconstruction error:', err);
    meshStatus.textContent = `Error: ${err.message}`;
    showToast('Reconstruction Error', err.message, 'error', 6000);
    announceA11y(`Reconstruction failed: ${err.message}`);
  } finally {
    loadingOverlay.classList.add('hidden');
    generateBtn.disabled = false;
  }
}

// -------------------------------------------------------------
// In-Browser WebGPU AI Pipeline (100% Client-Side, 0 Server Cost)
// -------------------------------------------------------------
let clientDepthPipeline = null;

async function executeInBrowserWebGPU() {
  const t0 = performance.now();
  loadingStatusText.textContent = 'Initializing In-Browser Neural Model (WebGPU)...';
  loadingOverlay.classList.remove('hidden');
  generateBtn.disabled = true;
  meshStatus.textContent = 'Computing 3D geometry locally on your device GPU (0 Host Resource)...';

  try {
    if (!clientDepthPipeline) {
      loadingStatusText.textContent = 'Downloading ONNX WebGPU Model (~25MB, cached in browser)...';
      const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3');
      env.allowLocalModels = false;
      try {
        clientDepthPipeline = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small', {
          device: 'webgpu',
          dtype: 'q4'
        });
      } catch (gpuErr) {
        console.warn('WebGPU fallback to wasm/cpu:', gpuErr);
        clientDepthPipeline = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small', {
          device: 'wasm',
          dtype: 'q4'
        });
      }
    }

    loadingStatusText.textContent = 'Inferring 3D Depth Map on your Device GPU...';
    const inputSrc = previewImage.src;
    const output = await clientDepthPipeline(inputSrc);
    const inferMs = performance.now() - t0;

    loadingStatusText.textContent = 'Generating 3D Solid Watertight Geometry...';
    const meshRes = parseInt(webgpuResSelect ? webgpuResSelect.value : 144) || 144;
    const depthScale = parseFloat(webgpuDepthSlider ? webgpuDepthSlider.value : 0.45) || 0.45;

    const meshResult = createSolid3DMeshFromDepth(output.depth, previewImage, meshRes, depthScale);

    applyMeshToScene(meshResult.mesh, true);

    const totalMs = performance.now() - t0;

    // Telemetry Update
    inferenceMetric.textContent = `${Math.round(inferMs)} ms`;
    mcMetric.textContent = `${Math.round(totalMs)} ms`;
    verticesMetric.textContent = meshResult.vertices.toLocaleString();
    facesMetric.textContent = meshResult.faces.toLocaleString();

    meshStatus.textContent = `In-Browser WebGPU • ${meshResult.vertices.toLocaleString()} verts, ${meshResult.faces.toLocaleString()} faces (0 Server Resource Used)`;

    // Export client-side OBJ directly as Blob with 0 server bandwidth
    const objBlobUrl = exportMeshToOBJBlob(meshResult.geometry);
    downloadObjBtn.href = objBlobUrl;
    downloadObjBtn.download = `model_client_${Date.now()}.obj`;
    downloadGlbBtn.classList.add('hidden');
    downloadBar.classList.remove('hidden');

  } catch (err) {
    console.error('Client WebGPU Error:', err);
    meshStatus.textContent = `Client Error: ${err.message}`;
    showToast('In-Browser WebGPU Error', err.message, 'error', 6500);
    announceA11y(`In-browser WebGPU generation error: ${err.message}`);
  } finally {
    loadingOverlay.classList.add('hidden');
    generateBtn.disabled = false;
  }
}

function createSolid3DMeshFromDepth(rawDepth, imgElement, gridResolution, depthScale) {
  const depthW = rawDepth.width;
  const depthH = rawDepth.height;
  const depthData = rawDepth.data;

  const aspect = (imgElement.naturalHeight || 1) / (imgElement.naturalWidth || 1);
  const cols = gridResolution;
  const rows = Math.round(gridResolution * aspect);

  let minD = Infinity, maxD = -Infinity;
  for (let i = 0; i < depthData.length; i++) {
    if (depthData[i] < minD) minD = depthData[i];
    if (depthData[i] > maxD) maxD = depthData[i];
  }
  const rangeD = (maxD - minD) || 1.0;

  function getDepth(u, v) {
    const px = Math.min(depthW - 1, Math.max(0, Math.floor(u * depthW)));
    const py = Math.min(depthH - 1, Math.max(0, Math.floor(v * depthH)));
    const val = depthData[py * depthW + px];
    return (val - minD) / rangeD;
  }

  const positions = [];
  const uvs = [];
  const indices = [];
  const stride = cols + 1;

  // Front displaced 3D surface
  for (let r = 0; r <= rows; r++) {
    const v = r / rows;
    const y = (0.5 - v) * (2 * aspect);
    for (let c = 0; c <= cols; c++) {
      const u = c / cols;
      const x = (u - 0.5) * 2;
      const z = getDepth(u, v) * depthScale;
      positions.push(x, y, z);
      uvs.push(u, 1 - v);
    }
  }

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const a = r * stride + c;
      const b = (r + 1) * stride + c;
      const d = r * stride + (c + 1);
      const e = (r + 1) * stride + (c + 1);
      indices.push(a, b, d);
      indices.push(d, b, e);
    }
  }

  // Back plate (solid watertight base)
  const backBaseIndex = positions.length / 3;
  const backZ = -0.06;
  for (let r = 0; r <= rows; r++) {
    const v = r / rows;
    const y = (0.5 - v) * (2 * aspect);
    for (let c = 0; c <= cols; c++) {
      const u = c / cols;
      const x = (u - 0.5) * 2;
      positions.push(x, y, backZ);
      uvs.push(u, 1 - v);
    }
  }

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const a = backBaseIndex + r * stride + c;
      const b = backBaseIndex + (r + 1) * stride + c;
      const d = backBaseIndex + r * stride + (c + 1);
      const e = backBaseIndex + (r + 1) * stride + (c + 1);
      indices.push(a, d, b);
      indices.push(d, e, b);
    }
  }

  // Side bevel walls
  // Top
  for (let c = 0; c < cols; c++) {
    const f1 = c, f2 = c + 1;
    const b1 = backBaseIndex + c, b2 = backBaseIndex + c + 1;
    indices.push(f1, f2, b1);
    indices.push(f2, b2, b1);
  }
  // Bottom
  for (let c = 0; c < cols; c++) {
    const f1 = rows * stride + c, f2 = rows * stride + c + 1;
    const b1 = backBaseIndex + rows * stride + c, b2 = backBaseIndex + rows * stride + c + 1;
    indices.push(f1, b1, f2);
    indices.push(f2, b1, b2);
  }
  // Left
  for (let r = 0; r < rows; r++) {
    const f1 = r * stride, f2 = (r + 1) * stride;
    const b1 = backBaseIndex + r * stride, b2 = backBaseIndex + (r + 1) * stride;
    indices.push(f1, b1, f2);
    indices.push(f2, b1, b2);
  }
  // Right
  for (let r = 0; r < rows; r++) {
    const f1 = r * stride + cols, f2 = (r + 1) * stride + cols;
    const b1 = backBaseIndex + r * stride + cols, b2 = backBaseIndex + (r + 1) * stride + cols;
    indices.push(f1, f2, b1);
    indices.push(f2, b2, b1);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  const texture = new THREE.Texture(imgElement);
  texture.needsUpdate = true;
  texture.encoding = THREE.sRGBEncoding;

  const material = new THREE.MeshStandardMaterial({
    map: texture,
    roughness: 0.4,
    metalness: 0.1,
    side: THREE.DoubleSide
  });

  const mesh = new THREE.Mesh(geometry, material);
  return {
    mesh,
    geometry,
    vertices: positions.length / 3,
    faces: indices.length / 3
  };
}

function exportMeshToOBJBlob(geometry) {
  const pos = geometry.attributes.position;
  const uvs = geometry.attributes.uv;
  let obj = "# 3D Vision Lab Client-Side WebGPU OBJ Export\n";
  for (let i = 0; i < pos.count; i++) {
    obj += `v ${pos.getX(i).toFixed(4)} ${pos.getY(i).toFixed(4)} ${pos.getZ(i).toFixed(4)}\n`;
  }
  if (uvs) {
    for (let i = 0; i < uvs.count; i++) {
      obj += `vt ${uvs.getX(i).toFixed(4)} ${uvs.getY(i).toFixed(4)}\n`;
    }
  }
  const index = geometry.index;
  if (index) {
    for (let i = 0; i < index.count; i += 3) {
      const a = index.getX(i) + 1;
      const b = index.getX(i + 1) + 1;
      const c = index.getX(i + 2) + 1;
      obj += `f ${a}/${a} ${b}/${b} ${c}/${c}\n`;
    }
  }
  const blob = new Blob([obj], { type: 'text/plain' });
  return URL.createObjectURL(blob);
}

// -------------------------------------------------------------
// Loaders: GLTF (Colored) & OBJ
// -------------------------------------------------------------
const gltfLoader = new THREE.GLTFLoader();
const objLoader = new THREE.OBJLoader();

function loadGlbMesh(url, fallbackObjUrl) {
  gltfLoader.load(url, (gltf) => {
    const object = gltf.scene;
    applyMeshToScene(object, true);
  }, undefined, (err) => {
    console.warn('GLB load failed, falling back to OBJ:', err);
    if (fallbackObjUrl) loadObjMesh(fallbackObjUrl);
  });
}

function loadObjMesh(url) {
  objLoader.load(url, (object) => {
    applyMeshToScene(object, false);
  });
}

function applyMeshToScene(object, isColored) {
  if (currentMesh) {
    currentMesh.traverse((child) => {
      if (child.isMesh) {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(m => m.dispose());
          } else {
            child.material.dispose();
          }
        }
      }
    });
    scene.remove(currentMesh);
  }
  if (placeholderMesh) {
    placeholderMesh.visible = false;
  }

  object.traverse((child) => {
    if (child.isMesh) {
      if (!isColored || !child.material) {
        child.material = new THREE.MeshStandardMaterial({
          color: 0x8898aa,
          roughness: 0.4,
          metalness: 0.15,
          wireframe: isWireframe,
          side: THREE.DoubleSide
        });
      } else {
        child.material.wireframe = isWireframe;
        child.material.side = THREE.DoubleSide;
        child.material.needsUpdate = true;
      }
      child.geometry.computeVertexNormals();
    }
  });

  // Center Mesh
  const box = new THREE.Box3().setFromObject(object);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());

  object.position.sub(center);
  object.position.y += size.y / 2 - 0.5;

  currentMesh = object;
  scene.add(currentMesh);

  resetCamera(size);
}

function resetCamera(size) {
  const maxDim = size ? Math.max(size.x, size.y, size.z) : 1.0;
  camera.position.set(maxDim * 1.5, maxDim * 1.2, maxDim * 1.8);
  controls.target.set(0, 0, 0);
  controls.update();
}

// -------------------------------------------------------------
// Viewport Tools & Accessibility Listeners
// -------------------------------------------------------------
rotateBtn.addEventListener('click', () => {
  autoRotate = !autoRotate;
  rotateBtn.classList.toggle('active', autoRotate);
  rotateBtn.setAttribute('aria-pressed', autoRotate);
  announceA11y(autoRotate ? 'Auto rotation enabled.' : 'Auto rotation paused.');
});

wireframeBtn.addEventListener('click', () => {
  isWireframe = !isWireframe;
  wireframeBtn.classList.toggle('active', isWireframe);
  wireframeBtn.setAttribute('aria-pressed', isWireframe);
  announceA11y(isWireframe ? 'Wireframe view activated.' : 'Solid mesh view activated.');

  if (currentMesh) {
    currentMesh.traverse((child) => {
      if (child.isMesh && child.material) {
        child.material.wireframe = isWireframe;
      }
    });
  }
});

resetCamBtn.addEventListener('click', () => {
  resetCamera();
  announceA11y('Camera view reset to default.');
});

// Keyboard support for dropzone and modal
if (dropzone) {
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });
}

window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && warningModal && !warningModal.classList.contains('hidden')) {
    warningModal.classList.add('hidden');
    announceA11y('Resource warning dialog dismissed.');
  }
});

// -------------------------------------------------------------
// WebGPU Compatibility Verification on Launch
// -------------------------------------------------------------
async function checkWebGPUAvailability() {
  const banner = document.getElementById('webgpuNoticeBanner');
  const dismissBtn = document.getElementById('dismissBannerBtn');
  const hardwareBadge = document.getElementById('hardwareBadge');

  if (dismissBtn && banner) {
    dismissBtn.addEventListener('click', () => banner.classList.add('hidden'));
  }

  if (!navigator.gpu) {
    console.info('WebGPU is not enabled or available in this browser. Defaulting to Cloud TRELLIS.2 engine.');
    if (banner) banner.classList.remove('hidden');
    if (hardwareBadge) {
      hardwareBadge.innerHTML = '<span class="pulse-dot"></span> Cloud AI Ready';
    }
    if (engineSelect && engineSelect.value === 'client_webgpu') {
      engineSelect.value = 'trellis';
      engineSelect.dispatchEvent(new Event('change'));
    }
  } else {
    console.info('WebGPU hardware adapter confirmed available.');
    if (hardwareBadge) {
      hardwareBadge.innerHTML = '<span class="pulse-dot"></span> WebGPU Active';
    }
  }
}
checkWebGPUAvailability();
