/**
 * 3D Vision Lab — Interactive Web Viewer Client (Option 2).
 * Three.js 3D Viewport + Real-Time Neural Mesh Inference.
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
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');
const generateBtn = document.getElementById('generateBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const meshStatus = document.getElementById('meshStatus');
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

// -------------------------------------------------------------
// Three.js 3D Viewport Setup
// -------------------------------------------------------------
const canvas = document.getElementById('canvas3d');
const container = document.getElementById('viewportContainer');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0c12);

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.05, 100);
camera.position.set(1.6, 1.2, 2.0);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = autoRotate;
controls.autoRotateSpeed = 2.0;

// Lighting Rig
const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0xfff5ea, 1.2);
keyLight.position.set(3, 5, 2);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x90b0ff, 0.6);
fillLight.position.set(-3, 2, -2);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0x6366f1, 0.8);
rimLight.position.set(0, -3, -3);
scene.add(rimLight);

// Floor Grid & Shadow Plane
const gridHelper = new THREE.GridHelper(4, 20, 0x6366f1, 0x1e2436);
gridHelper.position.y = -0.55;
scene.add(gridHelper);

// Default Placeholder Geometric Preview
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
// Render Loop & Resize
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
// Image Input Handling (Upload, Drag-and-Drop, Clipboard)
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

// Paste from clipboard
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
  if (!file.type.startsWith('image/')) {
    alert('Please upload an image file.');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    setImageSource(e.target.result, null);
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

  // Trigger reconstruction automatically on image selection for instant gratification!
  triggerReconstruction();
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
// Pre-Loaded Sample Gallery
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

// -------------------------------------------------------------
// Slider & Settings
// -------------------------------------------------------------
thresholdSlider.addEventListener('input', (e) => {
  thresholdValue.textContent = Number(e.target.value).toFixed(2);
});

// -------------------------------------------------------------
// Neural Reconstruction API Request
// -------------------------------------------------------------
generateBtn.addEventListener('click', triggerReconstruction);

async function triggerReconstruction() {
  if (!currentImageBase64 && !currentImageUrl) return;

  loadingOverlay.classList.remove('hidden');
  generateBtn.disabled = true;
  meshStatus.textContent = 'Neural network inferring 3D shape on Apple Silicon GPU...';

  const payload = {
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

    // Update Telemetry Metrics
    inferenceMetric.textContent = `${data.inference_ms} ms`;
    mcMetric.textContent = `${data.marching_cubes_ms} ms`;
    verticesMetric.textContent = Number(data.vertices).toLocaleString();
    facesMetric.textContent = Number(data.faces).toLocaleString();

    meshStatus.textContent = `Done in ${data.total_latency_ms} ms (${data.vertices.toLocaleString()} verts, ${data.faces.toLocaleString()} faces)`;

    // Load Reconstructed OBJ into 3D Viewport
    loadObjMesh(data.obj_url);

    // Setup Download Links
    downloadObjBtn.href = data.obj_url;
    if (data.glb_url) {
      downloadGlbBtn.href = data.glb_url;
      downloadGlbBtn.classList.remove('hidden');
    } else {
      downloadGlbBtn.classList.add('hidden');
    }
    downloadBar.classList.remove('hidden');

  } catch (err) {
    console.error('Reconstruction error:', err);
    meshStatus.textContent = `Error: ${err.message}`;
    alert(`Reconstruction error: ${err.message}`);
  } finally {
    loadingOverlay.classList.add('hidden');
    generateBtn.disabled = false;
  }
}

// -------------------------------------------------------------
// Three.js OBJ Mesh Loader
// -------------------------------------------------------------
const objLoader = new THREE.OBJLoader();

function loadObjMesh(url) {
  objLoader.load(url, (object) => {
    // Remove previous mesh
    if (currentMesh) {
      scene.remove(currentMesh);
    }
    if (placeholderMesh) {
      placeholderMesh.visible = false;
    }

    // Apply sleek modern ceramic material
    const material = new THREE.MeshStandardMaterial({
      color: 0x7c8ba1,
      roughness: 0.35,
      metalness: 0.2,
      wireframe: isWireframe,
      side: THREE.DoubleSide
    });

    object.traverse((child) => {
      if (child.isMesh) {
        child.material = material;
        child.geometry.computeVertexNormals();
      }
    });

    // Compute bounding box & center mesh perfectly
    const box = new THREE.Box3().setFromObject(object);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    object.position.sub(center);
    // Align base with ground grid
    object.position.y += size.y / 2 - 0.5;

    currentMesh = object;
    scene.add(currentMesh);

    // Smooth camera reset
    resetCamera(size);
  });
}

function resetCamera(size) {
  const maxDim = size ? Math.max(size.x, size.y, size.z) : 1.0;
  camera.position.set(maxDim * 1.5, maxDim * 1.2, maxDim * 1.8);
  controls.target.set(0, 0, 0);
  controls.update();
}

// -------------------------------------------------------------
// Viewport Tools
// -------------------------------------------------------------
rotateBtn.addEventListener('click', () => {
  autoRotate = !autoRotate;
  rotateBtn.classList.toggle('active', autoRotate);
});

wireframeBtn.addEventListener('click', () => {
  isWireframe = !isWireframe;
  wireframeBtn.classList.toggle('active', isWireframe);

  if (currentMesh) {
    currentMesh.traverse((child) => {
      if (child.isMesh) {
        child.material.wireframe = isWireframe;
      }
    });
  }
});

resetCamBtn.addEventListener('click', () => {
  resetCamera();
});
