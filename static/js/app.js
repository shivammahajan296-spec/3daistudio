import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const form = document.querySelector("#promptForm");
const promptInput = document.querySelector("#promptInput");
const generateBtn = document.querySelector("#generateBtn");
const editBtn = document.querySelector("#editBtn");
const resetBtn = document.querySelector("#resetBtn");
const clearKeyBtn = document.querySelector("#clearKeyBtn");
const apiKeyInput = document.querySelector("#apiKeyInput");
const providerSelect = document.querySelector("#providerSelect");
const saveKeyBtn = document.querySelector("#saveKeyBtn");
const keyStatus = document.querySelector("#keyStatus");
const apiStatusBadge = document.querySelector("#apiStatusBadge");
const stepFileInput = document.querySelector("#stepFileInput");
const importStepBtn = document.querySelector("#importStepBtn");
const imageFileInput = document.querySelector("#imageFileInput");
const inferImageBtn = document.querySelector("#inferImageBtn");
const scrollTopBtn = document.querySelector("#scrollTopBtn");
const progressPanel = document.querySelector("#progressPanel");
const progressList = document.querySelector("#progressList");
const messages = document.querySelector("#messages");
const statusTitle = document.querySelector("#statusTitle");
const viewer = document.querySelector("#viewer");
const viewerEmpty = document.querySelector("#viewerEmpty");
const creationOverlay = document.querySelector("#creationOverlay");
const codeBlock = document.querySelector("#codeBlock");
const historyBtn = document.querySelector("#historyBtn");
const historyPanel = document.querySelector("#historyPanel");
const historyList = document.querySelector("#historyList");
const downloadStep = document.querySelector("#downloadStep");
const versionList = document.querySelector("#versionList");
const versionCount = document.querySelector("#versionCount");

let scene;
let camera;
let renderer;
let controls;
let currentMesh;
let animationFrame;
let currentResult = null;
let pendingBeforeImage = null;
let previewLoadToken = 0;
let versions = [];
let activeVersionId = null;
let progressTimer = null;
let progressIndex = 0;

initViewer();
loadSavedApiKey();
renderVersions();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitPrompt("generate");
});

editBtn.addEventListener("click", async () => {
  await submitPrompt("edit");
});

importStepBtn.addEventListener("click", async () => {
  await importStepFile();
});

inferImageBtn.addEventListener("click", async () => {
  await inferImageFile();
});

versionList.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-version-id]");
  if (!button) return;
  const version = versions.find((item) => item.id === button.dataset.versionId);
  if (!version) return;
  await restoreVersion(version);
});

scrollTopBtn.addEventListener("click", () => {
  messages.scrollTo({ top: 0, behavior: "smooth" });
});

clearKeyBtn.addEventListener("click", () => {
  apiKeyInput.value = "";
  localStorage.removeItem(providerStorageKey());
  keyStatus.textContent = `${providerLabel()} key cleared.`;
  setApiStatus("neutral", "No key");
});

providerSelect.addEventListener("change", () => {
  loadSavedApiKey();
});

async function importStepFile() {
  const file = stepFileInput.files && stepFileInput.files[0];
  if (!file) {
    addMessage("Choose a .step or .stp file first.", "error");
    return;
  }
  const lowerName = file.name.toLowerCase();
  if (!lowerName.endsWith(".step") && !lowerName.endsWith(".stp")) {
    addMessage("Please choose a .step or .stp file.", "error");
    return;
  }

  addMessage(`Import STEP: ${file.name}`, "user");
  addMessage("Importing the STEP file, validating geometry, and preparing it for prompt edits...", "assistant");
  startProgress("import");
  setBusy(true);
  setStatus("Importing STEP");
  clearDownload();

  try {
    const body = new FormData();
    body.append("file", file);
    const response = await fetch("/import-step", {
      method: "POST",
      body,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "STEP import failed.");
    }
    await renderResult(data, { recordVersion: true });
  } catch (error) {
    setStatus("Import Failed");
    addMessage(error.message, "error");
  } finally {
    finishProgress();
    setBusy(false);
  }
}

async function inferImageFile() {
  const file = imageFileInput.files && imageFileInput.files[0];
  if (!file) {
    addMessage("Choose an image first.", "error");
    return;
  }
  const lowerName = file.name.toLowerCase();
  if (![".png", ".jpg", ".jpeg", ".webp"].some((suffix) => lowerName.endsWith(suffix))) {
    addMessage("Please choose a PNG, JPG, JPEG, or WEBP image.", "error");
    return;
  }

  addImageMessage(file);
  addMessage("Image Inference Agent is reading the image and turning it into a CAD prompt...", "assistant");
  startProgress("image");
  setBusy(true);
  setStatus("Inferring Image");

  try {
    const body = new FormData();
    body.append("file", file);
    body.append("llm_provider", providerSelect.value);
    body.append("llm_api_key", apiKeyInput.value.trim() || "");
    const response = await fetch("/infer-image", {
      method: "POST",
      body,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Image inference failed.");
    }
    promptInput.value = data.prompt || "";
    renderLogs(data.logs || []);
    renderLlmStatus(data.llm);
    addMessage(data.message || "Image inferred into a CAD prompt.", "assistant");
    addMessage(data.prompt || "", "assistant");
    setStatus("Image Prompt Ready");
  } catch (error) {
    setStatus("Image Inference Failed");
    addMessage(error.message, "error");
  } finally {
    finishProgress();
    setBusy(false);
  }
}

async function submitPrompt(mode) {
  const prompt = promptInput.value.trim();
  if (!prompt) return;
  if (mode === "edit" && (!currentResult || !currentResult.code)) {
    addMessage("Generate a model first, then describe the edit you want.", "error");
    return;
  }
  if (mode === "edit") {
    pendingBeforeImage = captureViewerImage();
  }

  addMessage(mode === "edit" ? `Edit current model: ${prompt}` : prompt, "user");
  addMessage(
    mode === "edit"
      ? "Editing the current CadQuery model, executing it, and validating the new export..."
      : "Building the geometry plan, generating CadQuery, and validating the export...",
    "assistant"
  );
  startProgress(mode);
  const apiKey = apiKeyInput.value.trim();
  if (apiKey) {
    addMessage(`Sending ${providerLabel()} key ${fingerprintKey(apiKey)}.`, "assistant");
  } else {
    addMessage(`No ${providerLabel()} key is in the UI field; local fallback will be used.`, "assistant");
  }
  setBusy(true);
  setStatus("Generating");
  clearDownload();

  try {
    const body = {
      prompt,
      llm_provider: providerSelect.value,
      llm_api_key: apiKey || null,
      openrouter_api_key: providerSelect.value === "openrouter" ? apiKey || null : null,
    };
    if (mode === "edit") {
      body.previous_code = currentResult.code;
      body.previous_run_id = currentResult.run_id;
    }
    const response = await fetch(mode === "edit" ? "/edit" : "/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Generation failed.");
    }

    await renderResult(data, { recordVersion: true });
  } catch (error) {
    setStatus("Failed");
    addMessage(error.message, "error");
    pendingBeforeImage = null;
  } finally {
    finishProgress();
    setBusy(false);
  }
}

resetBtn.addEventListener("click", () => {
  messages.innerHTML = "";
  addMessage("Try: create a simple cosmetic jar with cylindrical body and lid", "assistant");
  promptInput.value = "create a simple cosmetic jar with cylindrical body and lid";
  codeBlock.textContent = "No code yet.";
  historyList.innerHTML = "";
  clearDownload();
  setStatus("Ready");
  clearMesh();
  currentResult = null;
  pendingBeforeImage = null;
  versions = [];
  activeVersionId = null;
  renderVersions();
  finishProgress();
  setApiStatus("neutral", "Not checked");
  stepFileInput.value = "";
  imageFileInput.value = "";
  editBtn.disabled = true;
});

historyBtn.addEventListener("click", () => {
  historyPanel.classList.toggle("hidden");
});

saveKeyBtn.addEventListener("click", () => {
  const key = apiKeyInput.value.trim();
  if (key) {
    localStorage.setItem(providerStorageKey(), key);
    localStorage.setItem("cadgen_llm_provider", providerSelect.value);
    keyStatus.textContent = `${providerLabel()} key saved in this browser.`;
    setApiStatus("neutral", "Saved");
  } else {
    localStorage.removeItem(providerStorageKey());
    keyStatus.textContent = `${providerLabel()} key cleared.`;
    setApiStatus("neutral", "No key");
  }
});

function loadSavedApiKey() {
  const savedProvider = localStorage.getItem("cadgen_llm_provider");
  if (savedProvider && providerSelect.querySelector(`option[value="${savedProvider}"]`)) {
    providerSelect.value = savedProvider;
  }
  const saved = localStorage.getItem(providerStorageKey()) || (providerSelect.value === "openrouter" ? localStorage.getItem("openrouter_api_key") : "");
  if (saved) {
    apiKeyInput.value = saved;
    keyStatus.textContent = `${providerLabel()} key loaded from this browser.`;
    setApiStatus("neutral", "Saved");
  } else {
    apiKeyInput.value = "";
    keyStatus.textContent = `Saved only in this browser for ${providerLabel()}.`;
    setApiStatus("neutral", "Not checked");
  }
}

function providerStorageKey() {
  return `cadgen_${providerSelect.value}_api_key`;
}

function providerLabel() {
  const option = providerSelect.selectedOptions[0];
  return option ? option.textContent : "LLM";
}

async function renderResult(data, options = {}) {
  codeBlock.textContent = data.code || "No code returned.";
  renderLogs(data.logs || []);
  renderLlmStatus(data.llm);

  if (data.status === "success") {
    currentResult = data;
    if (options.recordVersion) {
      addVersion(data);
    }
    editBtn.disabled = false;
    setStatus(`Ready after ${data.attempt_count} attempt${data.attempt_count === 1 ? "" : "s"}`);
    addMessage(data.message, "assistant");
    setDownload(data.files.step);
    setStatus(data.mode === "edit" ? "Loading Edited Preview" : "Loading Preview");
    await loadStl(data.files.stl, () => {
      if (data.mode === "edit" && pendingBeforeImage) {
        const afterImage = captureViewerImage();
        addBeforeAfterMessage(pendingBeforeImage, afterImage);
        pendingBeforeImage = null;
      }
    });
    setStatus(`Ready after ${data.attempt_count} attempt${data.attempt_count === 1 ? "" : "s"}`);
  } else {
    setStatus("Failed");
    addMessage(data.message || "Generation failed.", "error");
    pendingBeforeImage = null;
  }
}

async function restoreVersion(version) {
  currentResult = version.result;
  activeVersionId = version.id;
  codeBlock.textContent = currentResult.code || "No code returned.";
  renderLogs(currentResult.logs || []);
  renderLlmStatus(currentResult.llm);
  setDownload(currentResult.files.step);
  setStatus(`Restored ${version.label}`);
  renderVersions();
  await loadStl(currentResult.files.stl);
  editBtn.disabled = false;
  addMessage(`Restored ${version.label}.`, "assistant");
}

function addVersion(result) {
  const number = versions.length + 1;
  const version = {
    id: `${result.run_id}-${number}`,
    label: `v${number}`,
    result,
    title: versionTitle(result),
    meta: `${result.mode || "generate"} · ${result.attempt_count || 1} attempt${result.attempt_count === 1 ? "" : "s"}`,
  };
  versions.push(version);
  activeVersionId = version.id;
  renderVersions();
}

function versionTitle(result) {
  if (result.mode === "import") return `Imported STEP`;
  if (result.mode === "edit") return `Edited: ${truncate(result.prompt || "model", 42)}`;
  return `Generated: ${truncate(result.prompt || "model", 42)}`;
}

function renderVersions() {
  versionCount.textContent = String(versions.length);
  versionList.innerHTML = "";
  if (!versions.length) {
    const empty = document.createElement("li");
    empty.className = "version-item";
    empty.innerHTML = `<button type="button" disabled><span class="version-title">No versions yet</span><span class="version-meta">Generate, edit, or import to start</span></button>`;
    versionList.appendChild(empty);
    return;
  }
  for (const version of versions.slice().reverse()) {
    const item = document.createElement("li");
    item.className = `version-item ${version.id === activeVersionId ? "active" : ""}`;
    item.innerHTML = `
      <button type="button" data-version-id="${escapeHtml(version.id)}">
        <span class="version-title">${escapeHtml(version.label)} · ${escapeHtml(version.title)}</span>
        <span class="version-meta">${escapeHtml(version.meta)}</span>
      </button>
    `;
    versionList.appendChild(item);
  }
}

function truncate(text, max) {
  return text.length > max ? `${text.slice(0, max - 1)}...` : text;
}

function renderLogs(logs) {
  historyList.innerHTML = "";
  if (!logs.length) {
    const item = document.createElement("li");
    item.className = "agent-step";
    item.textContent = "No agent activity yet.";
    historyList.appendChild(item);
    return;
  }
  for (const log of logs) {
    const item = document.createElement("li");
    item.className = `agent-step ${log.status === "failed" ? "failed" : ""}`;
    item.innerHTML = agentLogHtml(log);
    historyList.appendChild(item);
  }
}

function agentLogHtml(log) {
  const detail = log.detail || {};
  const parts = [];
  parts.push(`<div class="agent-step-head"><strong>${agentName(log.node)}</strong><span>${escapeHtml(log.status || "done")}</span></div>`);
  parts.push(`<p>${escapeHtml(agentDescription(log.node, detail))}</p>`);

  const chips = [];
  if (detail.attempt) chips.push(`Attempt ${detail.attempt}`);
  if (detail.llm_source) chips.push(detail.llm_source === "fallback" ? "Local fallback" : detail.provider || label(detail.llm_source));
  if (detail.metadata && detail.metadata.volume) chips.push(`Volume ${Math.round(detail.metadata.volume)}`);
  if (detail.metadata && detail.metadata.step_size) chips.push(`STEP ${formatBytes(detail.metadata.step_size)}`);
  if (detail.non_empty_geometry) chips.push("Non-empty geometry");
  if (detail.step_exists) chips.push("STEP exists");
  if (detail.stl_exists) chips.push("Preview mesh exists");
  if (chips.length) {
    parts.push(`<div class="agent-chips">${chips.map((chip) => `<span>${escapeHtml(chip)}</span>`).join("")}</div>`);
  }

  const error = detail.llm_error || detail.error || (detail.message && log.status === "failed" ? detail.message : "");
  if (error) {
    parts.push(`<div class="agent-error">${escapeHtml(String(error))}</div>`);
  }
  return parts.join("");
}

function agentName(node) {
  const names = {
    understand_prompt: "Prompt Understanding Agent",
    import_step: "STEP Import Agent",
    image_inference: "Image Inference Agent",
    generate_code: "CadQuery Code Agent",
    execute_code: "Code Execution Agent",
    validate_output: "Geometry Validation Agent",
    repair_code: "Self-Repair Agent",
    package_response: "Export Packaging Agent",
  };
  return names[node] || label(node);
}

function agentDescription(node, detail) {
  const descriptions = {
    understand_prompt: detail.plan ? `Interpreted the prompt into a geometry plan: ${detail.plan}` : "Interpreted the product prompt into a geometry plan.",
    import_step: detail.file_name ? `Imported ${detail.file_name} and created editable CadQuery import code.` : "Imported a STEP file as the current model.",
    image_inference: detail.prompt ? `Inferred a CAD prompt from ${detail.file_name || "the image"}: ${detail.prompt}` : "Inferred the uploaded image into a CAD prompt.",
    generate_code: "Generated runnable CadQuery Python for the planned geometry.",
    execute_code: detail.ok ? "Executed the CadQuery script in a subprocess and exported model files." : "Executed the CadQuery script and captured the failure for repair.",
    validate_output: detail.ok ? "Validated that geometry is non-empty and output files exist." : "Checked geometry and output files; validation failed.",
    repair_code: "Analyzed the previous failure and generated repaired CadQuery code.",
    package_response: "Prepared the final response, file links, code, metadata, and preview paths.",
  };
  return descriptions[node] || "Completed an agent step.";
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  return `${Math.round(bytes / 1024)} KB`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderLlmStatus(llm) {
  if (!llm) return;
  const name = llm.provider_label || label(llm.provider || "provider");
  if (llm.used_provider && !llm.fallback_used) {
    setApiStatus("ok", `${name} connected`);
    return;
  }
  if (llm.used_provider && llm.fallback_used) {
    setApiStatus("warn", "Partial fallback");
    return;
  }
  if (llm.fallback_used) {
    setApiStatus(llm.key_received ? "bad" : "warn", llm.key_received ? "Auth failed" : "Fallback");
  }
}

function setApiStatus(kind, text) {
  apiStatusBadge.className = `status-badge ${kind}`;
  apiStatusBadge.textContent = text;
}

function startProgress(mode) {
  const steps = progressSteps(mode);
  progressIndex = 0;
  progressPanel.classList.remove("hidden");
  renderProgress(steps, progressIndex);
  clearInterval(progressTimer);
  progressTimer = setInterval(() => {
    progressIndex = Math.min(progressIndex + 1, steps.length - 1);
    renderProgress(steps, progressIndex);
  }, 1400);
}

function finishProgress() {
  clearInterval(progressTimer);
  progressTimer = null;
  const items = [...progressList.querySelectorAll("li")];
  for (const item of items) {
    item.classList.remove("active");
    item.classList.add("done");
  }
  if (items.length) {
    setTimeout(() => progressPanel.classList.add("hidden"), 900);
  }
}

function progressSteps(mode) {
  if (mode === "import") {
    return ["Uploading STEP", "Importing geometry", "Exporting preview", "Validating files"];
  }
  if (mode === "image") {
    return ["Uploading image", "Inferring shape", "Writing CAD prompt", "Ready to generate"];
  }
  if (mode === "edit") {
    return ["Planning edit", "Rewriting CadQuery", "Executing model", "Validating geometry", "Loading preview"];
  }
  return ["Understanding prompt", "Generating CadQuery", "Executing model", "Validating geometry", "Loading preview"];
}

function renderProgress(steps, activeIndex) {
  progressList.innerHTML = "";
  steps.forEach((step, index) => {
    const item = document.createElement("li");
    item.className = index < activeIndex ? "done" : index === activeIndex ? "active" : "";
    item.innerHTML = `<span class="progress-dot"></span><span>${escapeHtml(step)}</span>`;
    progressList.appendChild(item);
  });
}

function fingerprintKey(key) {
  if (key.length <= 14) return "entered in the UI";
  return `${key.slice(0, 8)}...${key.slice(-6)}`;
}

function addMessage(text, role) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  article.appendChild(bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function addBeforeAfterMessage(beforeSrc, afterSrc) {
  if (!beforeSrc || !afterSrc) return;
  const article = document.createElement("article");
  article.className = "message assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble comparison-bubble";
  bubble.innerHTML = `
    <strong>Before / After</strong>
    <div class="comparison-grid">
      <figure>
        <img src="${beforeSrc}" alt="Model before edit">
        <figcaption>Before</figcaption>
      </figure>
      <figure>
        <img src="${afterSrc}" alt="Model after edit">
        <figcaption>After</figcaption>
      </figure>
    </div>
  `;
  article.appendChild(bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function addImageMessage(file) {
  const url = URL.createObjectURL(file);
  const article = document.createElement("article");
  article.className = "message user";
  const bubble = document.createElement("div");
  bubble.className = "bubble image-bubble";
  bubble.innerHTML = `
    <strong>Infer from image</strong>
    <img src="${url}" alt="Uploaded reference image">
  `;
  article.appendChild(bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function setBusy(isBusy) {
  generateBtn.disabled = isBusy;
  editBtn.disabled = isBusy || !currentResult;
  importStepBtn.disabled = isBusy;
  inferImageBtn.disabled = isBusy;
  generateBtn.textContent = isBusy ? "Generating..." : "Generate";
  editBtn.textContent = isBusy ? "Working..." : "Edit Current";
  importStepBtn.textContent = isBusy ? "Working..." : "Import STEP";
  inferImageBtn.textContent = isBusy ? "Working..." : "Infer Image";
  creationOverlay.classList.toggle("hidden", !isBusy);
}

function setStatus(text) {
  statusTitle.textContent = text;
}

function setDownload(url) {
  if (!url) {
    clearDownload();
    return;
  }
  downloadStep.href = url;
  downloadStep.download = "generated-model.step";
  downloadStep.classList.remove("disabled");
  downloadStep.setAttribute("aria-disabled", "false");
}

function clearDownload() {
  downloadStep.href = "#";
  downloadStep.removeAttribute("download");
  downloadStep.classList.add("disabled");
  downloadStep.setAttribute("aria-disabled", "true");
}

function initViewer() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101417);

  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
  camera.position.set(110, 90, 120);

  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  viewer.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const hemi = new THREE.HemisphereLight(0xffffff, 0x28343a, 2.2);
  scene.add(hemi);

  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(80, 120, 90);
  scene.add(key);

  const grid = new THREE.GridHelper(180, 18, 0x49615e, 0x273331);
  grid.position.y = -1;
  scene.add(grid);

  window.addEventListener("resize", resizeViewer);
  resizeViewer();
  animate();
}

function resizeViewer() {
  const rect = viewer.getBoundingClientRect();
  const width = Math.max(rect.width, 320);
  const height = Math.max(rect.height, 320);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function animate() {
  animationFrame = requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function loadStl(url, onLoaded) {
  if (!url) return Promise.resolve();
  const token = ++previewLoadToken;
  const loader = new STLLoader();
  const previewUrl = `${url}${url.includes("?") ? "&" : "?"}preview=${Date.now()}`;
  return new Promise((resolve, reject) => {
    loader.load(
      previewUrl,
      (geometry) => {
        if (token !== previewLoadToken) {
          geometry.dispose();
          resolve();
          return;
        }
      clearMesh();
      geometry.computeVertexNormals();
      geometry.center();

      const material = new THREE.MeshStandardMaterial({
        color: 0x2bb3a3,
        roughness: 0.42,
        metalness: 0.08,
      });
      currentMesh = new THREE.Mesh(geometry, material);
      scene.add(currentMesh);

      const rawBox = new THREE.Box3().setFromObject(currentMesh);
      const size = rawBox.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      const scale = 100 / maxDim;
      currentMesh.scale.setScalar(scale);
      currentMesh.rotation.x = -Math.PI / 2;

      resizeViewer();
      frameMesh(currentMesh);
      viewerEmpty.classList.add("hidden");
      renderer.render(scene, camera);
      if (onLoaded) {
        requestAnimationFrame(() => {
          resizeViewer();
          controls.update();
          renderer.render(scene, camera);
          onLoaded();
          resolve();
        });
      } else {
        resolve();
      }
    },
    undefined,
    (error) => {
      addMessage(`Preview failed to load: ${error.message || "STL loader error"}`, "error");
      reject(error);
    }
    );
  });
}

function captureViewerImage() {
  if (!renderer || !currentMesh) return null;
  renderer.render(scene, camera);
  try {
    return renderer.domElement.toDataURL("image/png");
  } catch {
    return null;
  }
}

function frameMesh(mesh) {
  const box = new THREE.Box3().setFromObject(mesh);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 100;
  const distance = maxDim * 1.85;
  camera.position.set(center.x + distance, center.y + distance * 0.78, center.z + distance);
  camera.near = Math.max(distance / 100, 0.1);
  camera.far = distance * 12;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function clearMesh() {
  previewLoadToken += 1;
  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh.geometry.dispose();
    currentMesh.material.dispose();
    currentMesh = null;
  }
  viewerEmpty.classList.remove("hidden");
}

function label(node) {
  return String(node || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

window.addEventListener("beforeunload", () => {
  if (animationFrame) cancelAnimationFrame(animationFrame);
});
