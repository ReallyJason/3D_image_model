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

// -------------------------------------------------------------
// Engine Switcher UI Logic
// -------------------------------------------------------------
engineSelect.addEventListener('change', () => {
  const engine = engineSelect.value;
  if (engine === 'trellis') {
    hfTokenRow.classList.remove('hidden');
    triposrControls.classList.add('hidden');
    voxelControls.classList.add('hidden');
    activeModelBadge.textContent = 'Engine: Microsoft TRELLIS.2 (SOTA)';
  } else if (engine === 'instantmesh') {
    hfTokenRow.classList.remove('hidden');
    triposrControls.classList.add('hidden');
    voxelControls.classList.add('hidden');
    activeModelBadge.textContent = 'Engine: Tencent InstantMesh';
  } else if (engine === 'triposr') {
    hfTokenRow.classList.add('hidden');
    triposrControls.classList.remove('hidden');
    voxelControls.classList.add('hidden');
    activeModelBadge.textContent = 'Engine: TripoSR (Local Mac GPU)';
  } else {
    hfTokenRow.classList.add('hidden');
    triposrControls.classList.add('hidden');
    voxelControls.classList.remove('hidden');
    activeModelBadge.textContent = 'Engine: TinyImageToVoxelNet (Custom Baseline)';
  }
  if (currentImageBase64 || currentImageUrl) {
    triggerReconstruction();
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

  meshStatus.textContent = 'Image loaded. Generating 3D model...';
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
// Neural Reconstruction API
// -------------------------------------------------------------
generateBtn.addEventListener('click', triggerReconstruction);

async function triggerReconstruction() {
  if (!currentImageBase64 && !currentImageUrl) return;

  const engine = engineSelect.value;
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
    alert(`Reconstruction error: ${err.message}`);
  } finally {
    loadingOverlay.classList.add('hidden');
    generateBtn.disabled = false;
  }
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
      if (child.isMesh && child.material) {
        child.material.wireframe = isWireframe;
      }
    });
  }
});

resetCamBtn.addEventListener('click', () => {
  resetCamera();
});
