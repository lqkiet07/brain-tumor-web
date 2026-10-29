/* =========================================================
   Brain Tumor MRI Lab — Application Logic & Workspace Manager
   Data-driven · Track Switching · Clean Tabular Rendering
   ========================================================= */

const NF = new Intl.NumberFormat("vi-VN");
const DEC2 = new Intl.NumberFormat("vi-VN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const DEC4 = new Intl.NumberFormat("vi-VN", { minimumFractionDigits: 4, maximumFractionDigits: 4 });

let PROJECT = null;
let ACTIVE_TRACK = "vgg";
let ACTIVE_FILTER = "all";

function byId(id) {
  return document.getElementById(id);
}

/* ---------------------------------------------------------- Leaderboard */
function renderLeaderboard(filter = "all") {
  ACTIVE_FILTER = filter;
  const tbody = byId("leaderboard-table").querySelector("tbody");
  const filtered = filter === "all"
    ? PROJECT.models
    : PROJECT.models.filter((m) => (m.family === filter || m.track === filter));

  const sorted = [...filtered].sort((a, b) => b.accuracy - a.accuracy);

  tbody.innerHTML = sorted
    .map((m) => {
      const isTopOverall = m.key === "densenet121";
      const isTopDemo = m.key === "vgg16_b4b5";
      const tagClass = `track-tag--${m.family || m.track}`;
      const familyLabel = m.family_name || (m.track === "vgg" ? "VGG" : m.track === "resnet" ? "ResNet" : "MobileNet");

      const f1Display = m.macro_f1 != null
        ? DEC4.format(m.macro_f1)
        : `<span class="null-val" title="Đo qua evaluate() Keras (Loss: 0.35–0.41), chưa trích xuất ma trận F1 đa lớp">—*</span>`;

      const latencyDisplay = m.latency_ms != null
        ? `${DEC2.format(m.latency_ms)} ms`
        : `<span class="null-val" title="Chạy trên môi trường Kaggle GPU độc lập">—</span>`;

      const sizeDisplay = m.file_mb != null
        ? `${DEC2.format(m.file_mb)} MB`
        : `<span class="null-val" title="Chưa nạp trọng số cục bộ">—</span>`;

      let badgeHtml = "";
      if (isTopOverall) {
        badgeHtml = '<span class="best-badge best-badge--top">Top 1 (97,29%)</span>';
      } else if (isTopDemo) {
        badgeHtml = '<span class="best-badge best-badge--demo">Top Demo</span>';
      }

      return `
        <tr class="${isTopOverall ? "is-best-row" : isTopDemo ? "is-demo-row" : ""}">
          <td>
            <strong>${m.name}</strong>
            ${badgeHtml}
          </td>
          <td><span class="track-tag ${tagClass}">${familyLabel}</span></td>
          <td class="num">${NF.format(m.params)}</td>
          <td class="num"><strong>${DEC2.format(m.accuracy)}%</strong></td>
          <td class="num">${f1Display}</td>
          <td class="num">${latencyDisplay}</td>
          <td class="num">${sizeDisplay}</td>
        </tr>`;
    })
    .join("");

  // Performance Bars
  const barsContainer = byId("accuracy-bars");
  barsContainer.innerHTML = sorted
    .map((m) => {
      const isTopOverall = m.key === "densenet121";
      return `
        <div class="perf-bar ${isTopOverall ? "perf-bar--best" : ""}">
          <span class="perf-bar__name" title="${m.name}">${m.name}</span>
          <div class="perf-bar__track">
            <div class="perf-bar__fill" style="width: ${m.accuracy}%"></div>
          </div>
          <span class="perf-bar__val mono">${DEC2.format(m.accuracy)}%</span>
        </div>`;
    })
    .join("");
}

/* ---------------------------------------------------------- Track Workspaces */
function renderTrackWorkspace(trackId) {
  ACTIVE_TRACK = trackId;
  const track = PROJECT.tracks[trackId];
  const container = byId("track-panel");

  // Track Header & Runs List
  let runsHtml = "";
  if (trackId === "vgg") {
    // VGG Staircase
    runsHtml = `
      <div class="staircase">
        ${track.runs.map((r, i) => `
          <div class="step step--${i + 1}">
            <div>
              ${r.gain ? `<span class="step__gain">${r.gain}</span>` : ""}
              <div class="step__acc">${DEC2.format(r.test_accuracy)}%</div>
              <div class="step__name">${r.name}</div>
            </div>
            <div class="step__note">${r.note} · ${NF.format(r.trainable_params)} params</div>
          </div>
        `).join("")}
      </div>`;
  } else {
    // Standard Card List
    runsHtml = `
      <div style="display:grid; gap:0.75rem; margin:1.25rem 0;">
        ${track.runs.map((r) => `
          <div style="background:#fff; border:1px solid var(--border); border-radius:var(--radius); padding:1rem; display:flex; justify-content:space-between; align-items:baseline; gap:1rem;">
            <div>
              <strong>${r.name}</strong>
              <div style="font-size:12px; color:var(--text-secondary); margin-top:0.2rem;">${r.note}</div>
            </div>
            <div class="mono" style="font-size:1.2rem; font-weight:600; color:${r.test_accuracy ? "var(--text-primary)" : "var(--text-muted)"}">
              ${r.test_accuracy ? DEC2.format(r.test_accuracy) + "%" : r.val_accuracy ? DEC2.format(r.val_accuracy) + "% (val)" : "—"}
            </div>
          </div>
        `).join("")}
      </div>`;
  }

  container.innerHTML = `
    <div class="track-left">
      <h3 style="font-size:1.25rem; margin-bottom:0.6rem;">${track.name}</h3>
      <p class="section-lead" style="margin-bottom:1.5rem;">${track.lead}</p>
      ${runsHtml}
    </div>

    <div class="track-right" style="display:grid; gap:1.25rem;">
      <div style="background:#fff; border:1px solid var(--border); border-radius:var(--radius); padding:1rem;">
        <span class="eyebrow" style="margin:0 0 0.5rem; display:block;">Ma trận nhầm lẫn</span>
        <img src="/${track.cm_image}" alt="Ma trận nhầm lẫn của ${track.name}" style="border-radius:var(--radius-sm); border:1px solid var(--border);" loading="lazy" />
      </div>

      <div style="background:var(--lightbox-canvas); border:1px solid var(--lightbox-border); border-radius:var(--radius); padding:1.25rem; color:var(--lightbox-text);">
        <span class="eyebrow" style="color:var(--lightbox-muted); margin:0 0 0.5rem; display:block;">Bản đồ nhiệt kích hoạt (XAI)</span>
        <img src="/${track.xai_image}" alt="Bản đồ nhiệt ${track.name}" style="border-radius:var(--radius-sm); max-height:260px; width:100%; object-fit:contain; background:#000;" loading="lazy" />
        <p style="font-size:12px; color:var(--lightbox-muted); margin:0.75rem 0 0; line-height:1.5;">${track.xai_text}</p>
      </div>
    </div>
  `;
}

/* ---------------------------------------------------------- Dataset Grid */
function renderDatasetStrip() {
  const aug = PROJECT.assets.filter((f) => f.startsWith("dataset/aug_")).sort();
  byId("aug-strip").innerHTML = aug
    .map((file, i) => `<img src="/assets/${file}" alt="Biến thể tăng cường ${i + 1}" loading="lazy" />`)
    .join("");
}

/* ---------------------------------------------------------- Init */
async function boot() {
  try {
    const res = await fetch("/data/project.json");
    if (!res.ok) throw new Error("HTTP " + res.status);
    PROJECT = await res.json();
  } catch (err) {
    document.body.insertAdjacentHTML(
      "afterbegin",
      `<div style="padding:1rem; background:#fef2f2; color:#b91c1c; border-bottom:1px solid #fecaca;">
         Không tải được dữ liệu (HTTP ${err.message}). Hãy chắc chắn server đang chạy.
       </div>`
    );
    return;
  }

  // Filter Buttons
  document.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      renderLeaderboard(btn.dataset.filter);
    });
  });

  // Track Tab Buttons
  document.querySelectorAll(".track-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".track-tab-btn").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      renderTrackWorkspace(btn.dataset.track);
    });
  });

  renderLeaderboard("all");
  renderTrackWorkspace("vgg");
  renderDatasetStrip();
}

document.addEventListener("DOMContentLoaded", boot);
