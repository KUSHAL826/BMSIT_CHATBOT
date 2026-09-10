// BMSIT College AI Agent - Admin Dashboard Controller

document.addEventListener("DOMContentLoaded", () => {
  initScraperControls();
  initFileUploadDropzone();
  initDeleteHandlers();
  initDocumentViewerModal();
  initSettingsModal();
  pollStatusIfRunning();
});

// Toast Notification
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container") || createToastContainer();
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  const icon = type === "success" ? "✅" : type === "error" ? "❌" : "ℹ️";
  toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

function createToastContainer() {
  const c = document.createElement("div");
  c.id = "toast-container";
  c.className = "toast-container";
  document.body.appendChild(c);
  return c;
}

// -------------------------------------------------------------
// Scraper Controls
// -------------------------------------------------------------
function initScraperControls() {
  const scrapeBtn = document.getElementById("btn-scrape-now");
  const saveUrlBtn = document.getElementById("btn-save-settings");
  
  if (scrapeBtn) {
    scrapeBtn.addEventListener("click", triggerScrapeNow);
  }

  if (saveUrlBtn) {
    saveUrlBtn.addEventListener("click", saveScraperSettings);
  }
}

async function triggerScrapeNow() {
  const btn = document.getElementById("btn-scrape-now");
  const statusBox = document.getElementById("scraper-status-box");
  const statusMsg = document.getElementById("scraper-status-msg");
  const fillBar = document.getElementById("scraper-progress-fill");

  btn.disabled = true;
  btn.innerHTML = `<span class="spinner">⏳</span> Scraping in Progress...`;
  statusBox.classList.add("active");
  statusMsg.textContent = "Starting BMSIT crawler...";
  fillBar.style.width = "15%";

  try {
    const res = await fetch("/api/admin/scrape", { method: "POST" });
    const data = await res.json();
    if (res.ok) {
      showToast("Web crawl initiated!", "success");
      pollScrapeProgress();
    } else {
      showToast(data.message || "Failed to trigger scrape", "error");
      btn.disabled = false;
      btn.innerHTML = "⚡ Scrape Now";
    }
  } catch (err) {
    showToast("Network error triggering scrape", "error");
    btn.disabled = false;
    btn.innerHTML = "⚡ Scrape Now";
  }
}

let pollInterval = null;
function pollScrapeProgress() {
  if (pollInterval) clearInterval(pollInterval);

  pollInterval = setInterval(async () => {
    try {
      const res = await fetch("/api/admin/scrape/status");
      const status = await res.json();

      const btn = document.getElementById("btn-scrape-now");
      const statusBox = document.getElementById("scraper-status-box");
      const statusMsg = document.getElementById("scraper-status-msg");
      const fillBar = document.getElementById("scraper-progress-fill");

      if (status.is_running) {
        statusMsg.textContent = status.status_message || "Processing...";
        let pct = 20;
        if (status.phase === "chunking") {
          pct = 85;
        } else if (status.phase === "indexing") {
          pct = 92;
        } else if (status.total_target > 0) {
          pct = Math.min(80, Math.max(10, Math.round((status.pages_scraped / status.total_target) * 80)));
        }
        fillBar.style.width = `${pct}%`;
      } else {
        clearInterval(pollInterval);
        fillBar.style.width = "100%";
        btn.disabled = false;
        btn.innerHTML = "⚡ Scrape Now";

        if (status.error) {
          statusMsg.textContent = `Completed with error: ${status.error}`;
          showToast(`Scrape failed: ${status.error}`, "error");
        } else {
          statusMsg.textContent = status.status_message || "Completed successfully!";
          showToast("Scraping and FAISS indexing completed!", "success");
          setTimeout(() => location.reload(), 1500);
        }
      }
    } catch (e) {
      console.error("Error polling scrape status:", e);
    }
  }, 1200);
}

async function pollStatusIfRunning() {
  try {
    const res = await fetch("/api/admin/scrape/status");
    const status = await res.json();
    if (status.is_running) {
      const btn = document.getElementById("btn-scrape-now");
      const statusBox = document.getElementById("scraper-status-box");
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner">⏳</span> Scraping in Progress...`;
      }
      if (statusBox) statusBox.classList.add("active");
      pollScrapeProgress();
    }
  } catch (e) {}
}

async function saveScraperSettings() {
  const urlInput = document.getElementById("target-url-input");
  const maxPages = document.getElementById("max-pages-input");
  const depth = document.getElementById("max-depth-input");

  const payload = {
    bmsit_url: urlInput.value.trim(),
    scrape_max_pages: parseInt(maxPages.value) || 30,
    scrape_depth: parseInt(depth.value) || 2
  };

  try {
    const res = await fetch("/api/admin/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      showToast("Scraper settings saved!", "success");
    } else {
      showToast("Error updating settings", "error");
    }
  } catch (err) {
    showToast("Network error saving settings", "error");
  }
}

// -------------------------------------------------------------
// File Upload Dropzone (PDF, DOC, DOCX, CSV)
// -------------------------------------------------------------
let selectedFiles = [];

function initFileUploadDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const uploadBtn = document.getElementById("btn-upload-files");
  const previewList = document.getElementById("file-preview-list");

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", () => fileInput.click());

  ["dragenter", "dragover"].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files.length) {
      handleNewFiles(dt.files);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      handleNewFiles(fileInput.files);
    }
  });

  if (uploadBtn) {
    uploadBtn.addEventListener("click", uploadSelectedFiles);
  }
}

function handleNewFiles(fileList) {
  const allowed = ["pdf", "docx", "doc", "csv"];
  for (const f of fileList) {
    const ext = f.name.split(".").pop().toLowerCase();
    if (allowed.includes(ext)) {
      selectedFiles.push(f);
    } else {
      showToast(`Unsupported format: ${f.name}. Only PDF, DOC, DOCX, CSV allowed.`, "error");
    }
  }
  renderFilePreviews();
}

function renderFilePreviews() {
  const container = document.getElementById("file-preview-list");
  const uploadBtn = document.getElementById("btn-upload-files");
  if (!container) return;

  container.innerHTML = "";
  if (selectedFiles.length === 0) {
    if (uploadBtn) uploadBtn.style.display = "none";
    return;
  }

  if (uploadBtn) {
    uploadBtn.style.display = "inline-flex";
    uploadBtn.innerHTML = `📤 Index ${selectedFiles.length} File${selectedFiles.length > 1 ? 's' : ''} into RAG`;
  }

  selectedFiles.forEach((file, idx) => {
    const item = document.createElement("div");
    item.className = "file-preview-item";
    const sizeKb = Math.round(file.size / 1024);
    item.innerHTML = `
      <div><strong>${file.name}</strong> <span style="color:#94a3b8; font-size:12px;">(${sizeKb} KB)</span></div>
      <button style="background:none; border:none; color:#f87171; cursor:pointer; font-weight:bold;" onclick="removeSelectedFile(${idx})">✕</button>
    `;
    container.appendChild(item);
  });
}

window.removeSelectedFile = function(idx) {
  selectedFiles.splice(idx, 1);
  renderFilePreviews();
};

async function uploadSelectedFiles() {
  if (selectedFiles.length === 0) return;
  const uploadBtn = document.getElementById("btn-upload-files");
  uploadBtn.disabled = true;
  uploadBtn.innerHTML = `⏳ Extracting, Chunking & Embedding...`;

  const formData = new FormData();
  selectedFiles.forEach(f => formData.append("files", f));

  try {
    const res = await fetch("/api/admin/upload", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (res.ok) {
      showToast(`Successfully indexed ${data.results.length} files!`, "success");
      selectedFiles = [];
      renderFilePreviews();
      setTimeout(() => location.reload(), 1200);
    } else {
      showToast(data.error || "Upload failed", "error");
      uploadBtn.disabled = false;
      renderFilePreviews();
    }
  } catch (err) {
    showToast("Network error during file upload", "error");
    uploadBtn.disabled = false;
    renderFilePreviews();
  }
}

// -------------------------------------------------------------
// Training Source Deletion Handlers
// -------------------------------------------------------------
function initDeleteHandlers() {
  document.querySelectorAll(".btn-delete-source").forEach(btn => {
    btn.addEventListener("click", async (e) => {
      const sourceId = btn.getAttribute("data-id");
      const sourceName = btn.getAttribute("data-name");

      if (!confirm(`Are you sure you want to delete "${sourceName}"?\nAll associated vectors will be immediately removed from the FAISS RAG knowledge base.`)) {
        return;
      }

      btn.disabled = true;
      btn.textContent = "Deleting...";

      try {
        const res = await fetch(`/api/admin/source/${sourceId}`, {
          method: "DELETE"
        });
        const data = await res.json();
        if (res.ok) {
          showToast(`Deleted ${sourceName} (${data.removed_chunks} vectors removed)`, "success");
          const row = document.getElementById(`history-row-${sourceId}`);
          if (row) {
            row.style.opacity = "0";
            setTimeout(() => row.remove(), 300);
          }
          // Refresh metrics
          setTimeout(() => location.reload(), 1000);
        } else {
          showToast(data.error || "Delete failed", "error");
          btn.disabled = false;
          btn.textContent = "Delete";
        }
      } catch (err) {
        showToast("Error deleting source", "error");
        btn.disabled = false;
        btn.textContent = "Delete";
      }
    });
  });
}

// -------------------------------------------------------------
// Settings Modal (API Key configuration)
// -------------------------------------------------------------
function initSettingsModal() {
  const modal = document.getElementById("settings-modal");
  const openBtn = document.getElementById("btn-open-settings");
  const closeBtn = document.getElementById("btn-close-settings");
  const saveKeyBtn = document.getElementById("btn-save-key");

  if (!modal) return;

  if (openBtn) openBtn.addEventListener("click", () => modal.style.display = "flex");
  if (closeBtn) closeBtn.addEventListener("click", () => modal.style.display = "none");

  if (saveKeyBtn) {
    saveKeyBtn.addEventListener("click", async () => {
      const keyInput = document.getElementById("gemini-api-key-input");
      const keyVal = keyInput.value.trim();
      if (!keyVal) {
        showToast("Please enter a valid API key", "error");
        return;
      }

      try {
        const res = await fetch("/api/admin/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ gemini_api_key: keyVal })
        });
        if (res.ok) {
          showToast("Gemini API Key saved and activated!", "success");
          modal.style.display = "none";
          keyInput.value = "";
          setTimeout(() => location.reload(), 1000);
        } else {
          showToast("Failed to save API key", "error");
        }
      } catch (e) {
        showToast("Network error saving key", "error");
      }
    });
  }
}

// -------------------------------------------------------------
// Document Preview Modal (PDF / Documents Viewer)
// -------------------------------------------------------------
function initDocumentViewerModal() {
  const modal = document.getElementById("doc-viewer-modal");
  const iframe = document.getElementById("doc-viewer-iframe");
  const textContainer = document.getElementById("doc-viewer-text-container");
  const modalTitle = document.getElementById("doc-modal-title");
  const externalLink = document.getElementById("doc-modal-external-link");
  const closeBtn = document.getElementById("btn-close-doc-modal");

  if (!modal || !iframe) return;

  function closeViewer() {
    modal.style.display = "none";
    iframe.src = "about:blank";
    if (textContainer) {
      textContainer.innerHTML = "";
      textContainer.style.display = "none";
    }
  }

  if (closeBtn) closeBtn.addEventListener("click", closeViewer);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeViewer();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.style.display === "flex") closeViewer();
  });

  document.querySelectorAll(".btn-view-doc").forEach(btn => {
    btn.addEventListener("click", async () => {
      const filename = btn.getAttribute("data-file") || "Knowledge Source";
      const sourceId = btn.getAttribute("data-id");
      const sourceType = btn.getAttribute("data-type") || (filename.includes(".") ? "document" : "website");

      modalTitle.textContent = filename;
      modal.style.display = "flex";

      if (sourceType === "website" && sourceId) {
        // Scraped Website Preview
        iframe.style.display = "none";
        iframe.src = "about:blank";
        externalLink.href = "https://bmsit.ac.in";
        externalLink.textContent = "🌐 Open BMSIT Website";
        if (textContainer) {
          textContainer.style.display = "block";
          textContainer.innerHTML = `<div style="text-align:center; padding: 40px; color: #94a3b8;"><span style="font-size:24px;">⏳</span><br><br>Loading scraped website knowledge preview...</div>`;

          try {
            const res = await fetch(`/api/admin/source-preview/${sourceId}`);
            const data = await res.json();
            if (res.ok && data.status === "success") {
              let html = `<div style="max-width: 860px; margin: 0 auto;">`;
              html += `<div style="margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.1);">
                <h2 style="font-size: 20px; font-weight: 700; color: #fff;">🌐 ${data.title}</h2>
                <div style="font-size: 13px; color: #94a3b8; margin-top: 6px; display: flex; gap: 16px; flex-wrap: wrap;">
                  <span>Indexed Date: <strong style="color:#e2e8f0;">${data.date}</strong></span>
                  <span>Total Chunks in FAISS: <strong style="color:#38bdf8;">${data.total_chunks}</strong></span>
                  <span>Pages Scraped: <strong style="color:#a855f7;">${data.pages_count}</strong></span>
                </div>
              </div>`;

              // Detected changes
              if (data.changes && data.changes.length > 0) {
                html += `<div style="margin-bottom: 24px;">
                  <h3 style="font-size: 15px; font-weight: 600; color: #f59e0b; margin-bottom: 10px;">🔍 Detected Page Changes (${data.changes.length})</h3>
                  <div style="display: flex; flex-direction: column; gap: 8px; max-height: 220px; overflow-y: auto; padding-right: 6px;">`;
                data.changes.forEach(c => {
                  const tagColor = c.change_type === 'NEW' ? '#10b981' : (c.change_type === 'MODIFIED' ? '#f59e0b' : '#ef4444');
                  html += `<div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 8px 12px; font-size: 12.5px; display: flex; align-items: center; gap: 10px;">
                    <span style="background: ${tagColor}22; color: ${tagColor}; border: 1px solid ${tagColor}44; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700;">${c.change_type}</span>
                    <a href="${c.url}" target="_blank" style="color: #60a5fa; text-decoration: none; word-break: break-all;">${c.url}</a>
                    <span style="color: #64748b; margin-left: auto; font-size: 11px;">${c.summary || ''}</span>
                  </div>`;
                });
                html += `</div></div>`;
              }

              // Sample Chunks
              if (data.sample_chunks && data.sample_chunks.length > 0) {
                html += `<div>
                  <h3 style="font-size: 15px; font-weight: 600; color: #38bdf8; margin-bottom: 12px;">📚 Sample Knowledge Base Chunks in RAG (${data.sample_chunks.length} shown)</h3>
                  <div style="display: flex; flex-direction: column; gap: 12px;">`;
                data.sample_chunks.forEach((c, idx) => {
                  html += `<div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 12px 16px;">
                    <div style="font-size: 11px; color: #64748b; margin-bottom: 6px; font-family: monospace;">Chunk ID: ${c.id}</div>
                    <div style="font-size: 13px; color: #cbd5e1; line-height: 1.5;">${c.text}</div>
                  </div>`;
                });
                html += `</div></div>`;
              }

              html += `</div>`;
              textContainer.innerHTML = html;
            } else {
              textContainer.innerHTML = `<div style="text-align:center; padding: 40px; color: #f87171;">⚠️ ${data.error || 'Failed to load website preview'}</div>`;
            }
          } catch (e) {
            textContainer.innerHTML = `<div style="text-align:center; padding: 40px; color: #f87171;">⚠️ Network error loading website preview.</div>`;
          }
        }
        return;
      }

      // Document Preview
      const docUrl = `/api/admin/document/${encodeURIComponent(filename)}`;
      externalLink.href = docUrl;
      externalLink.textContent = "↗ Open in New Tab";
      const ext = filename.split(".").pop().toLowerCase();

      if (ext === "pdf") {
        if (textContainer) textContainer.style.display = "none";
        iframe.style.display = "block";
        iframe.src = docUrl;
      } else {
        iframe.style.display = "none";
        iframe.src = "about:blank";
        if (textContainer) {
          textContainer.style.display = "block";
          textContainer.innerHTML = `<div style="text-align:center; padding: 40px; color: #94a3b8;"><span style="font-size:24px;">⏳</span><br><br>Extracting document preview...</div>`;

          try {
            const previewUrl = sourceId ? `/api/admin/source-preview/${sourceId}` : `/api/admin/document-content/${encodeURIComponent(filename)}`;
            const res = await fetch(previewUrl);
            const data = await res.json();
            if (res.ok && data.sections && data.sections.length > 0) {
              let html = `<div style="max-width: 800px; margin: 0 auto;">`;
              html += `<div style="margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.1);"><h2 style="font-size: 20px; font-weight: 700; color: #fff;">${data.filename || filename}</h2><span style="font-size: 12px; color: #38bdf8; text-transform: uppercase;">Format: ${data.type || ext}</span></div>`;

              data.sections.forEach((sec, idx) => {
                html += `<div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 18px 22px; margin-bottom: 18px;">`;
                html += `<h4 style="font-size: 14px; font-weight: 600; color: #60a5fa; margin-bottom: 10px;">${sec.title || 'Section ' + (idx + 1)}</h4>`;
                const contentHtml = sec.content
                  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
                  .replace(/\n\n+/g, "</p><p style='margin-bottom:10px;'>")
                  .replace(/\n/g, "<br>");
                html += `<div style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6;"><p style='margin-bottom:10px;'>${contentHtml}</p></div>`;
                html += `</div>`;
              });

              html += `</div>`;
              textContainer.innerHTML = html;
            } else {
              textContainer.innerHTML = `<div style="text-align:center; padding: 40px; color: #f87171;">⚠️ Could not preview this file: ${data.message || 'No content found'}</div>`;
            }
          } catch (err) {
            textContainer.innerHTML = `<div style="text-align:center; padding: 40px; color: #f87171;">⚠️ Network error loading document preview.</div>`;
          }
        }
      }
    });
  });
}

