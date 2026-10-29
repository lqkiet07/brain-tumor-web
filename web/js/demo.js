/* =========================================================
   Brain Tumor MRI Lab — Multi-Model Diagnostic Demo Studio
   Model Switching · Medical Semantic Styling · Interactive XAI
   ========================================================= */

(function () {
  const modelSelect = document.getElementById("model-select");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const predictBtn = document.getElementById("predict-btn");
  const useSampleBtn = document.getElementById("use-sample");

  const resultBox = document.getElementById("result-box");
  const emptyState = document.getElementById("empty-state");
  const demoError = document.getElementById("demo-error");

  const resultLabel = document.getElementById("result-label");
  const resultConfidence = document.getElementById("result-confidence");
  const resultProbs = document.getElementById("result-probs");
  const resultMeta = document.getElementById("result-meta");

  const heatmapOriginal = document.getElementById("heatmap-original");
  const heatmapOverlay = document.getElementById("heatmap-overlay");
  const heatmapSlider = document.getElementById("heatmap-slider");
  const sliderLabel = document.getElementById("slider-label");

  const DEC2 = new Intl.NumberFormat("vi-VN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  let currentFile = null;
  let currentPreviewUrl = null;

  function showError(msg) {
    demoError.textContent = msg;
    demoError.style.display = "block";
  }

  function clearError() {
    demoError.textContent = "";
    demoError.style.display = "none";
  }

  function setFile(file, url) {
    currentFile = file;
    currentPreviewUrl = url;
    predictBtn.disabled = false;
    predictBtn.textContent = "Phân tích ảnh";
    clearError();
  }

  function reset() {
    currentFile = null;
    currentPreviewUrl = null;
    predictBtn.disabled = true;
    predictBtn.textContent = "Phân tích ảnh";
    resultBox.hidden = true;
    emptyState.style.display = "block";
    clearError();
  }

  function handleFiles(files) {
    const file = files && files[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      showError("File đã chọn không phải ảnh. Hãy chọn tệp JPEG hoặc PNG.");
      return;
    }
    setFile(file, URL.createObjectURL(file));
  }

  // Dropzone events
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => handleFiles(fileInput.files));

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-over");
    })
  );
  dropzone.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));

  // Sample MRI button
  useSampleBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/assets/demo/sample_mri.png");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const blob = await res.blob();
      const file = new File([blob], "sample_mri.png", { type: "image/png" });
      setFile(file, "/assets/demo/sample_mri.png");
    } catch (err) {
      showError("Không tải được ảnh mẫu (" + err.message + ").");
    }
  });

  // Heatmap Opacity Slider
  heatmapSlider.addEventListener("input", () => {
    const val = Number(heatmapSlider.value);
    heatmapOverlay.style.opacity = String(val / 100);
    sliderLabel.textContent = `Bản đồ nhiệt (${val}%)`;
  });

  // Render Result with Medical Semantic Coloring
  function renderResult(data) {
    const isSafe = data.predicted_class === "notumor";

    // Set Label and Medical Color (Green for safe, Red for tumor)
    resultLabel.textContent = data.predicted_label;
    resultLabel.className = `diag-status-label ${isSafe ? "status--safe" : "status--danger"}`;

    resultConfidence.textContent = DEC2.format(data.confidence * 100) + "%";
    resultConfidence.className = `diag-confidence mono ${isSafe ? "status--safe" : "status--danger"}`;

    // Render 4 Probability Bars
    const topClass = data.predicted_class;
    resultProbs.innerHTML = data.probabilities
      .map((item) => {
        const isTop = item.class === topClass;
        const colorClass = item.class === "notumor" ? "is-safe" : "is-danger";
        return `
        <li class="diag-prob-row">
          <span style="color:${isTop ? "var(--lightbox-text)" : "var(--lightbox-muted)"}; font-weight:${isTop ? "600" : "400"};">
            ${item.label}
          </span>
          <div class="diag-prob-track">
            <div class="diag-prob-fill ${isTop ? colorClass : ""}" style="width: ${item.probability * 100}%"></div>
          </div>
          <span class="mono" style="text-align:right; color:${isTop ? "var(--lightbox-text)" : "var(--lightbox-muted)"};">
            ${DEC2.format(item.probability * 100)}%
          </span>
        </li>`;
      })
      .join("");

    // Lightbox Images
    heatmapOriginal.src = currentPreviewUrl || "/assets/demo/sample_mri.png";
    heatmapOverlay.src = "data:image/png;base64," + data.heatmap_png_base64;
    heatmapOverlay.style.opacity = String(Number(heatmapSlider.value) / 100);

    // Meta breakdown
    resultMeta.textContent =
      `${data.model.name} · Tầng XAI: ${data.heatmap_layer} · Suy luận CPU: ${DEC2.format(data.predict_ms)} ms ` +
      `· Sinh Heatmap: ${DEC2.format(data.heatmap_ms)} ms · ${data.model.total_params.toLocaleString("vi-VN")} tham số ` +
      `· ${DEC2.format(data.model.reported_latency_ms)} ms trên GPU Kaggle`;

    emptyState.style.display = "none";
    resultBox.hidden = false;
  }

  async function loadModelOptions() {
    try {
      const [modelsRes, healthRes] = await Promise.all([
        fetch("/api/models"),
        fetch("/api/health"),
      ]);
      if (!modelsRes.ok || !healthRes.ok) throw new Error("HTTP error");
      const models = await modelsRes.json();
      const health = await healthRes.json();

      modelSelect.innerHTML = models
        .map((m) => {
          const badgeText = m.badge ? ` [${m.badge}]` : "";
          return `<option value="${m.id}">${m.name} — ${DEC2.format(m.accuracy)}%${badgeText}</option>`;
        })
        .join("");

      const defaultId = health.default_model;
      if (models.some((m) => m.id === defaultId)) {
        modelSelect.value = defaultId;
      }
      modelSelect.disabled = false;
    } catch (err) {
      modelSelect.innerHTML = '<option value="">Không tải được danh sách mô hình</option>';
      modelSelect.disabled = true;
      showError("Không tải được danh sách mô hình từ server.");
    }
  }

  // Predict API call
  predictBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    if (modelSelect.disabled || !modelSelect.value) {
      showError("Danh sách mô hình chưa sẵn sàng. Vui lòng thử lại.");
      return;
    }
    predictBtn.disabled = true;
    predictBtn.textContent = "Đang phân tích…";
    clearError();

    const selectedModel = modelSelect.value || "vgg16_b4b5";
    const body = new FormData();
    body.append("file", currentFile, currentFile.name || "mri.png");

    try {
      const res = await fetch(`/api/predict?model=${encodeURIComponent(selectedModel)}`, {
        method: "POST",
        body,
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        showError(payload.detail || `Lỗi chẩn đoán (HTTP ${res.status}).`);
        return;
      }
      renderResult(payload);
    } catch (err) {
      showError("Không kết nối được tới server Python. Hãy chắc chắn server đang chạy (run_demo.bat).");
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = "Phân tích ảnh";
    }
  });

  reset();
  loadModelOptions();
})();
