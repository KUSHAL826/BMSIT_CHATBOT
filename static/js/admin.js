// BMSIT College AI Agent - Admin Dashboard Controller

document.addEventListener("DOMContentLoaded", () => {
  initSessionControl();
  initScraperControls();
  initFileUploadDropzone();
  initDeleteHandlers();
  initDocumentViewerModal();
  initSettingsModal();
  initWebsitePagination();
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
  const forceScrapeBtn = document.getElementById("btn-force-scrape-now");
  const saveUrlBtn = document.getElementById("btn-save-settings");
  
  if (scrapeBtn) {
    scrapeBtn.addEventListener("click", () => triggerScrapeNow(false));
  }

  if (forceScrapeBtn) {
    forceScrapeBtn.addEventListener("click", () => triggerScrapeNow(true));
  }

  if (saveUrlBtn) {
    saveUrlBtn.addEventListener("click", saveScraperSettings);
  }
}

async function triggerScrapeNow(force = false) {
  const btn = document.getElementById("btn-scrape-now");
  const forceBtn = document.getElementById("btn-force-scrape-now");
  const statusBox = document.getElementById("scraper-status-box");
  const statusMsg = document.getElementById("scraper-status-msg");
  const fillBar = document.getElementById("scraper-progress-fill");
  const statusBar = document.getElementById("scraper-status-bar");
  const liveBadge = document.getElementById("scraper-live-badge");

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner">⏳</span> Scraping in Progress...`;
  }
  if (forceBtn) {
    forceBtn.disabled = true;
    forceBtn.innerHTML = `<span class="spinner">⏳</span> In Progress...`;
  }
  if (statusBox) statusBox.classList.add("active");
  if (statusBar) statusBar.style.display = "block";
  if (liveBadge) {
    liveBadge.textContent = "⏳ Crawling Active";
    liveBadge.className = "status-badge";
    liveBadge.style.color = "#38bdf8";
  }
  statusMsg.textContent = force ? "Starting full re-crawl and force embedding all pages..." : "Starting BMSITM crawler...";
  fillBar.style.width = "15%";

  try {
    const url = force ? "/api/admin/scrape?force=true&reembed_only=true" : "/api/admin/scrape";
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: !!force, reembed_only: !!force, trigger: force ? "Force Re-embed All Pages" : "Manual run" })
    });
    const data = await res.json();
    if (res.ok) {
      showToast(force ? "Full site scrape & force embedding initiated!" : "Web crawl initiated!", "success");
      pollScrapeProgress();
    } else {
      showToast(data.message || "Failed to trigger scrape", "error");
      if (btn) { btn.disabled = false; btn.innerHTML = "⚡ Scrape Now"; }
      if (forceBtn) { forceBtn.disabled = false; forceBtn.innerHTML = "🔄 Force Re-embed All Pages"; }
    }
  } catch (err) {
    showToast("Network error triggering scrape", "error");
    if (btn) { btn.disabled = false; btn.innerHTML = "⚡ Scrape Now"; }
    if (forceBtn) { forceBtn.disabled = false; forceBtn.innerHTML = "🔄 Force Re-embed All Pages"; }
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
      const statusBar = document.getElementById("scraper-status-bar");
      const liveBadge = document.getElementById("scraper-live-badge");

      if (status.is_running) {
        if (statusBar) statusBar.style.display = "block";
        if (liveBadge) {
          liveBadge.textContent = "⏳ Crawling Active";
          liveBadge.className = "status-badge";
          liveBadge.style.color = "#38bdf8";
        }
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
        const forceBtn = document.getElementById("btn-force-scrape-now");
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = "⚡ Scrape Now";
        }
        if (forceBtn) {
          forceBtn.disabled = false;
          forceBtn.innerHTML = "🔄 Force Re-embed All Pages";
        }

        if (status.error) {
          if (liveBadge) {
            liveBadge.textContent = "❌ Failed";
            liveBadge.className = "status-badge status-failed";
          }
          statusMsg.textContent = `Completed with error: ${status.error}`;
          showToast(`Scrape failed: ${status.error}`, "error");
        } else {
          if (liveBadge) {
            liveBadge.textContent = "● Completed";
            liveBadge.className = "status-badge status-indexed";
          }
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
      const statusBar = document.getElementById("scraper-status-bar");
      const liveBadge = document.getElementById("scraper-live-badge");
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner">⏳</span> Scraping in Progress...`;
      }
      if (statusBox) statusBox.classList.add("active");
      if (statusBar) statusBar.style.display = "block";
      if (liveBadge) {
        liveBadge.textContent = "⏳ Crawling Active";
        liveBadge.className = "status-badge";
        liveBadge.style.color = "#38bdf8";
      }
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
        const res = await fetch(`/api/admin/source/${encodeURIComponent(sourceId)}`, {
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
      const pageUrl = btn.getAttribute("data-url") || "";

      modalTitle.textContent = filename;
      modal.style.display = "flex";

      if ((sourceType === "website" || sourceType === "website_page") && sourceId) {
        // Scraped Website / Webpage Preview
        iframe.style.display = "none";
        iframe.src = "about:blank";
        if (pageUrl) {
          externalLink.href = pageUrl;
          externalLink.textContent = "↗ Open Live Webpage";
        } else {
          externalLink.href = "https://bmsit.ac.in";
          externalLink.textContent = "🌐 Open BMSITM Website";
        }

        if (textContainer) {
          textContainer.style.display = "block";
          textContainer.innerHTML = `<div style="text-align:center; padding: 40px; color: #94a3b8;"><span style="font-size:24px;">⏳</span><br><br>Loading scraped webpage knowledge preview...</div>`;

          try {
            const res = await fetch(`/api/admin/source-preview/${encodeURIComponent(sourceId)}`);
            const data = await res.json();
            if (res.ok && data.status === "success") {
              if (data.source_type === "website_page") {
                let html = `<div style="max-width: 860px; margin: 0 auto;">`;
                html += `<div style="margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.1);">
                  <h2 style="font-size: 20px; font-weight: 700; color: #fff;">🌐 ${data.title}</h2>
                  <div style="font-size: 13px; color: #94a3b8; margin-top: 6px; display: flex; gap: 16px; flex-wrap: wrap;">
                    <span>URL: <a href="${data.url}" target="_blank" style="color: #60a5fa; text-decoration: none;">${data.url} ↗</a></span>
                    <span>Indexed Date: <strong style="color:#e2e8f0;">${data.date}</strong></span>
                    <span>Vector Chunks: <strong style="color:#38bdf8;">${data.total_chunks}</strong></span>
                  </div>
                </div>`;

                if (data.text) {
                  html += `<div style="margin-bottom: 24px;">
                    <h3 style="font-size: 15px; font-weight: 600; color: #38bdf8; margin-bottom: 10px;">📄 Extracted Webpage Text</h3>
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 16px 20px; font-size: 13px; color: #cbd5e1; line-height: 1.6; max-height: 320px; overflow-y: auto; white-space: pre-wrap;">${data.text.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>
                  </div>`;
                }

                if (data.sample_chunks && data.sample_chunks.length > 0) {
                  html += `<div>
                    <h3 style="font-size: 15px; font-weight: 600; color: #a855f7; margin-bottom: 12px;">🧩 Indexed Vector Chunks (${data.sample_chunks.length})</h3>
                    <div style="display: flex; flex-direction: column; gap: 12px;">`;
                  data.sample_chunks.forEach((c) => {
                    html += `<div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 12px 16px;">
                      <div style="font-size: 11px; color: #64748b; margin-bottom: 6px; font-family: monospace;">Chunk ID: ${c.id || 'N/A'}</div>
                      <div style="font-size: 13px; color: #cbd5e1; line-height: 1.5;">${(c.text || '').replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>
                    </div>`;
                  });
                  html += `</div></div>`;
                }

                html += `</div>`;
                textContainer.innerHTML = html;
                return;
              }

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

// -------------------------------------------------------------
// Website Scraping Direct Table Pagination (10 chunks/page)
// -------------------------------------------------------------
function initWebsitePagination() {
  const tbody = document.getElementById("website-flat-tbody");
  const paginationContainer = document.getElementById("website-pagination-container");
  const buttonsWrap = document.getElementById("pagination-buttons-wrap");
  const pagStart = document.getElementById("pag-start");
  const pagEnd = document.getElementById("pag-end");
  const pagTotal = document.getElementById("pag-total");
  const searchInput = document.getElementById("website-search-input");
  const clearSearchBtn = document.getElementById("btn-clear-search");

  if (!tbody) return;

  const allRows = Array.from(tbody.querySelectorAll("tr.website-item-row"));
  if (allRows.length === 0) {
    if (paginationContainer) paginationContainer.style.display = "none";
    return;
  }

  const PAGE_SIZE = 10;
  let currentPage = 1;
  let filteredRows = [...allRows];

  function filterRows() {
    const query = searchInput ? searchInput.value.trim().toLowerCase() : "";
    if (clearSearchBtn) {
      clearSearchBtn.style.display = query ? "block" : "none";
    }

    if (!query) {
      filteredRows = [...allRows];
    } else {
      filteredRows = allRows.filter(row => {
        const title = row.getAttribute("data-title") || "";
        const url = row.getAttribute("data-url") || "";
        return title.includes(query) || url.includes(query);
      });
    }

    currentPage = 1;
    render();
  }

  function getPaginationRange(current, total) {
    if (total <= 7) {
      return Array.from({ length: total }, (_, i) => i + 1);
    }
    const delta = 1;
    const range = [];
    const rangeWithDots = [];
    let l;

    for (let i = 1; i <= total; i++) {
      if (i === 1 || i === total || (i >= current - delta && i <= current + delta)) {
        range.push(i);
      }
    }

    for (const i of range) {
      if (l) {
        if (i - l === 2) {
          rangeWithDots.push(l + 1);
        } else if (i - l !== 1) {
          rangeWithDots.push("...");
        }
      }
      rangeWithDots.push(i);
      l = i;
    }
    return rangeWithDots;
  }

  function render() {
    const totalEntries = filteredRows.length;
    const totalPages = Math.max(1, Math.ceil(totalEntries / PAGE_SIZE));

    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;

    const startIndex = (currentPage - 1) * PAGE_SIZE;
    const endIndex = Math.min(startIndex + PAGE_SIZE, totalEntries);

    // Hide all rows first
    allRows.forEach(row => {
      row.style.display = "none";
    });

    // Handle empty results state
    let noResultsRow = document.getElementById("no-matching-website-pages");
    if (totalEntries === 0) {
      if (!noResultsRow) {
        noResultsRow = document.createElement("tr");
        noResultsRow.id = "no-matching-website-pages";
        noResultsRow.innerHTML = `<td colspan="6" style="text-align: center; color: #94a3b8; padding: 36px 16px;">
          🔍 No pages match your search query. <button type="button" id="btn-reset-search-link" style="background:none; border:none; color:#38bdf8; text-decoration:underline; cursor:pointer;">Clear search</button>
        </td>`;
        tbody.appendChild(noResultsRow);
        const resetBtn = document.getElementById("btn-reset-search-link");
        if (resetBtn) {
          resetBtn.addEventListener("click", () => {
            if (searchInput) searchInput.value = "";
            filterRows();
          });
        }
      } else {
        noResultsRow.style.display = "";
      }
    } else {
      if (noResultsRow) {
        noResultsRow.style.display = "none";
      }
      // Show rows for current page
      for (let i = startIndex; i < endIndex; i++) {
        filteredRows[i].style.display = "";
      }
    }

    // Update Showing X to Y of Z entries
    if (pagStart) pagStart.textContent = totalEntries > 0 ? (startIndex + 1) : 0;
    if (pagEnd) pagEnd.textContent = endIndex;
    if (pagTotal) pagTotal.textContent = totalEntries;

    // Render pagination buttons
    if (buttonsWrap) {
      buttonsWrap.innerHTML = "";

      if (totalPages <= 1) {
        if (paginationContainer) {
          paginationContainer.style.display = totalEntries === 0 ? "none" : "flex";
        }
        return;
      }

      if (paginationContainer) {
        paginationContainer.style.display = "flex";
      }

      // Prev Button
      const prevBtn = document.createElement("button");
      prevBtn.type = "button";
      prevBtn.className = "pag-btn pag-prev";
      prevBtn.innerHTML = "◀ Prev";
      prevBtn.disabled = (currentPage === 1);
      prevBtn.addEventListener("click", () => {
        if (currentPage > 1) {
          currentPage--;
          render();
        }
      });
      buttonsWrap.appendChild(prevBtn);

      // Numbered Page Buttons with intelligent ellipsis
      const pageNumbers = getPaginationRange(currentPage, totalPages);
      pageNumbers.forEach(p => {
        if (p === "...") {
          const ellipsis = document.createElement("span");
          ellipsis.className = "pag-ellipsis";
          ellipsis.textContent = "…";
          buttonsWrap.appendChild(ellipsis);
        } else {
          const pageBtn = document.createElement("button");
          pageBtn.type = "button";
          pageBtn.className = `pag-btn ${p === currentPage ? "active" : ""}`;
          pageBtn.textContent = p;
          pageBtn.addEventListener("click", () => {
            currentPage = p;
            render();
          });
          buttonsWrap.appendChild(pageBtn);
        }
      });

      // Next Button
      const nextBtn = document.createElement("button");
      nextBtn.type = "button";
      nextBtn.className = "pag-btn pag-next";
      nextBtn.innerHTML = "Next ▶";
      nextBtn.disabled = (currentPage === totalPages);
      nextBtn.addEventListener("click", () => {
        if (currentPage < totalPages) {
          currentPage++;
          render();
        }
      });
      buttonsWrap.appendChild(nextBtn);
    }
  }

  // Event Listeners
  if (searchInput) {
    searchInput.addEventListener("input", filterRows);
  }

  if (clearSearchBtn) {
    clearSearchBtn.addEventListener("click", () => {
      searchInput.value = "";
      clearSearchBtn.style.display = "none";
      filterRows();
      searchInput.focus();
    });
  }

  // Initial render
  render();
}

// -------------------------------------------------------------
// Admin Session Control & 20-minute Inactivity Auto-Logout
// -------------------------------------------------------------
const INACTIVITY_TIMEOUT_MS = 20 * 60 * 1000; // 20 minutes
let lastUserInteraction = Date.now();
let isLoggingOut = false;

function recordUserActivity() {
  lastUserInteraction = Date.now();
}

function initSessionControl() {
  const timerEl = document.getElementById("session-timer");

  // Track user activity across interactions (debounced to avoid performance cost)
  const activityEvents = ["mousedown", "mousemove", "keydown", "scroll", "touchstart", "click"];
  let lastEventTime = 0;
  activityEvents.forEach((evt) => {
    window.addEventListener(
      evt,
      () => {
        const now = Date.now();
        if (now - lastEventTime > 1000) {
          lastUserInteraction = now;
          lastEventTime = now;
        }
      },
      { passive: true }
    );
  });

  // Update timer display and enforce automatic logout upon 20 minutes of inactivity
  setInterval(() => {
    if (isLoggingOut) return;
    const elapsed = Date.now() - lastUserInteraction;
    const remainingMs = Math.max(0, INACTIVITY_TIMEOUT_MS - elapsed);

    if (timerEl) {
      const minutes = Math.floor(remainingMs / 60000);
      const seconds = Math.floor((remainingMs % 60000) / 1000);
      timerEl.textContent = `${minutes}:${seconds < 10 ? "0" : ""}${seconds}`;

      // Visual warning colors as time counts down
      if (remainingMs < 60000) {
        timerEl.style.color = "#f87171"; // red (< 1m)
      } else if (remainingMs < 300000) {
        timerEl.style.color = "#fbbf24"; // amber (< 5m)
      } else {
        timerEl.style.color = "#38bdf8"; // default blue
      }
    }

    if (remainingMs <= 0) {
      isLoggingOut = true;
      window.location.href = "/admin/logout?expired=1";
    }
  }, 1000);

  // Global fetch interceptor: detect 401 Unauthorized (session expired on server)
  const originalFetch = window.fetch;
  window.fetch = async function (...args) {
    recordUserActivity();
    try {
      const response = await originalFetch.apply(this, args);
      if (response && response.status === 401 && !isLoggingOut) {
        try {
          const clone = response.clone();
          const json = await clone.json().catch(() => ({}));
          if (json.session_expired || window.location.pathname.startsWith("/admin")) {
            isLoggingOut = true;
            window.location.href = "/admin/login?expired=1";
          }
        } catch (_) {
          isLoggingOut = true;
          window.location.href = "/admin/login?expired=1";
        }
      }
      return response;
    } catch (err) {
      throw err;
    }
  };
}

