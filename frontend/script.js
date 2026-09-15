/**
 * script.js — Frontend logic for the Smart Attendance System.
 *
 * Responsibilities:
 *  1. Webcam stream via getUserMedia
 *  2. Periodic frame capture → base64 → POST /verify
 *  3. Update liveness badge, blink dots, recognition cards
 *  4. Poll /attendance/records every 5s and refresh table
 *  5. Register face modal (capture → POST /register)
 *  6. Stat counters and toast notifications
 */

// ─── Configuration ────────────────────────────────────────────────────────────
const API_BASE        = 'http://localhost:8000';
const CAPTURE_INTERVAL_MS  = 600;   // how often we POST a frame (ms)
const RECORDS_POLL_MS      = 5000;  // how often we refresh attendance table
const SESSION_ID           = `session_${Date.now()}`;
const REQUIRED_BLINKS      = 2;

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const webcamVideo       = document.getElementById('webcam-video');
const captureCanvas     = document.getElementById('capture-canvas');
const idleOverlay       = document.getElementById('idle-overlay');
const scanLine          = document.getElementById('scan-line');
const btnStart          = document.getElementById('btn-start');
const btnStop           = document.getElementById('btn-stop');
const btnClear          = document.getElementById('btn-clear');
const btnRegisterToggle = document.getElementById('btn-register-toggle');

const livenessBadge     = document.getElementById('liveness-badge');
const livenessMessage   = document.getElementById('liveness-message');
const blinkDots         = [document.getElementById('dot-0'), document.getElementById('dot-1')];
const recognitionPanel  = document.getElementById('recognition-panel');
const recognitionResults= document.getElementById('recognition-results');

const attendanceTbody   = document.getElementById('attendance-tbody');
const emptyRow          = document.getElementById('empty-row');
const lastRefresh       = document.getElementById('last-refresh');

const statusDot         = document.getElementById('status-dot');
const statusText        = document.getElementById('status-text');

const statTotal         = document.getElementById('stat-total');
const statPresent       = document.getElementById('stat-present');
const statSpoof         = document.getElementById('stat-spoof');
const statKnown         = document.getElementById('stat-known');

const registerModal     = document.getElementById('register-modal');
const registerVideo     = document.getElementById('register-video');
const registerCanvas    = document.getElementById('register-canvas');
const registerName      = document.getElementById('register-name');
const registerMsg       = document.getElementById('register-msg');
const btnCaptureReg     = document.getElementById('btn-capture-register');
const btnModalClose     = document.getElementById('btn-modal-close');

const toastEl        = document.getElementById('toast');
const toastContent      = document.getElementById('toast-content');

// ─── State ────────────────────────────────────────────────────────────────────
let captureIntervalId  = null;
let pollIntervalId     = null;
let webcamStream       = null;
let registerStream     = null;
let isLive             = false;
let currentBlinkCount  = 0;
let toastTimeout       = null;
let attendanceCache    = [];

// ─── Utilities ────────────────────────────────────────────────────────────────
function frameToBase64(videoEl, canvasEl, quality = 0.8) {
  canvasEl.width  = videoEl.videoWidth  || 640;
  canvasEl.height = videoEl.videoHeight || 480;
  const ctx = canvasEl.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
  return canvasEl.toDataURL('image/jpeg', quality);
}

function showToast(msg, type = 'info') {
  const colors = {
    info:    'text-gray-200',
    success: 'text-green-400',
    error:   'text-red-400',
    warning: 'text-yellow-400',
  };
  toastContent.className = `glassmorphism border border-white/10 rounded-xl px-5 py-3 text-sm font-medium shadow-xl ${colors[type] ?? colors.info}`;
  toastContent.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toastEl.classList.remove('show'), 3500);
}

function setSystemStatus(online) {
  if (online) {
    statusDot.className  = 'w-2 h-2 rounded-full bg-live animate-pulse';
    statusText.textContent = 'System Online';
  } else {
    statusDot.className  = 'w-2 h-2 rounded-full bg-spoof animate-pulse';
    statusText.textContent = 'Backend Unreachable';
  }
}

// ─── Backend health check ─────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    setSystemStatus(true);
    statKnown.textContent = data.known_faces ?? '—';
  } catch {
    setSystemStatus(false);
  }
}

// ─── Webcam helpers ───────────────────────────────────────────────────────────
async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    webcamVideo.srcObject = webcamStream;
    await webcamVideo.play();
    idleOverlay.style.display = 'none';
    scanLine.classList.add('active');
    return true;
  } catch (err) {
    showToast(`Camera error: ${err.message}`, 'error');
    return false;
  }
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  webcamVideo.srcObject = null;
  idleOverlay.style.display = '';
  scanLine.classList.remove('active');
}

// ─── Liveness + Recognition UI updaters ──────────────────────────────────────
function updateLivenessUI(liveness) {
  const { is_live, blink_count, message } = liveness;

  livenessMessage.textContent = message || '—';
  currentBlinkCount = blink_count ?? currentBlinkCount;

  // Blink dots
  blinkDots.forEach((dot, i) => {
    dot.classList.toggle('done', i < currentBlinkCount);
  });

  if (is_live && !isLive) {
    isLive = true;
    livenessBadge.className = 'liveness-badge badge-live';
    livenessBadge.textContent = '✓ Live';
    showToast('Liveness confirmed! Identifying…', 'success');
  } else if (!is_live && blink_count > 0) {
    livenessBadge.className = 'liveness-badge badge-checking';
    livenessBadge.textContent = `Blink ${blink_count}/${REQUIRED_BLINKS}`;
  } else if (!is_live) {
    livenessBadge.className = 'liveness-badge badge-checking';
    livenessBadge.textContent = 'Verifying…';
  }
}

function updateRecognitionUI(results) {
  if (!results || results.length === 0) {
    recognitionPanel.classList.add('hidden');
    return;
  }

  recognitionPanel.classList.remove('hidden');
  recognitionResults.innerHTML = '';

  results.forEach(r => {
    const confidencePct = Math.round((r.confidence ?? 0) * 100);
    const matched = r.matched;
    const nameColor = matched ? 'text-live' : 'text-unknown';
    const icon = matched ? '✓' : '?';

    const card = document.createElement('div');
    card.className = 'rec-result-card';
    card.innerHTML = `
      <div class="flex items-center gap-2">
        <span class="w-7 h-7 rounded-full bg-gray-800 flex items-center justify-center text-xs font-bold ${nameColor}">${icon}</span>
        <div>
          <p class="font-semibold text-sm ${nameColor}">${escapeHtml(r.name)}</p>
          <p class="text-xs text-gray-500">Face detected</p>
        </div>
      </div>
      <div class="text-right">
        <p class="text-sm font-bold ${nameColor}">${confidencePct}%</p>
        <p class="text-xs text-gray-500">confidence</p>
      </div>
    `;
    recognitionResults.appendChild(card);
  });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ─── Frame capture & verify ───────────────────────────────────────────────────
async function captureAndVerify() {
  if (!webcamStream) return;

  const imageB64 = frameToBase64(webcamVideo, captureCanvas);

  try {
    const res = await fetch(`${API_BASE}/verify`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ image: imageB64, session_id: SESSION_ID }),
      signal:  AbortSignal.timeout(5000),
    });

    if (!res.ok) {
      console.warn('Verify error:', res.status);
      return;
    }

    const data = await res.json();
    updateLivenessUI(data.liveness);
    updateRecognitionUI(data.recognition);

    // Notify on attendance log
    if (data.logged && data.logged.length > 0) {
      showToast(`✅ Logged: ${data.logged.join(', ')}`, 'success');
    }

  } catch (err) {
    if (err.name !== 'AbortError') {
      console.error('Verify failed:', err);
    }
  }
}

// ─── Attendance table ─────────────────────────────────────────────────────────
async function refreshAttendance() {
  try {
    const res  = await fetch(`${API_BASE}/attendance/records?limit=50`, {
      signal: AbortSignal.timeout(4000),
    });
    const rows = await res.json();

    attendanceCache = rows;
    renderTable(rows);
    updateStats(rows);
    lastRefresh.textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch {
    /* silent fail — will retry */
  }
}

function renderTable(rows) {
  // Remove old data rows (keep empty-row)
  attendanceTbody.querySelectorAll('.data-row').forEach(r => r.remove());

  if (rows.length === 0) {
    emptyRow.style.display = '';
    return;
  }

  emptyRow.style.display = 'none';

  rows.forEach(rec => {
    const tr = document.createElement('tr');
    tr.className = 'data-row';

    const pillClass = {
      present: 'pill-present',
      unknown: 'pill-unknown',
      spoof:   'pill-spoof',
    }[rec.status] ?? 'pill-unknown';

    tr.innerHTML = `
      <td class="td-cell font-medium text-white">${escapeHtml(rec.name)}</td>
      <td class="td-cell text-gray-400 text-xs">${escapeHtml(rec.timestamp)}</td>
      <td class="td-cell text-gray-300">${(rec.confidence * 100).toFixed(1)}%</td>
      <td class="td-cell"><span class="status-pill ${pillClass}">${escapeHtml(rec.status)}</span></td>
    `;
    attendanceTbody.appendChild(tr);
  });
}

function updateStats(rows) {
  statTotal.textContent   = rows.length;
  statPresent.textContent = rows.filter(r => r.status === 'present').length;
  statSpoof.textContent   = rows.filter(r => r.status === 'spoof').length;
}

// ─── Clear records ────────────────────────────────────────────────────────────
btnClear.addEventListener('click', async () => {
  if (!confirm('Clear all attendance records?')) return;
  try {
    await fetch(`${API_BASE}/attendance/clear`, { method: 'DELETE' });
    showToast('Records cleared.', 'warning');
    await refreshAttendance();
  } catch {
    showToast('Failed to clear records.', 'error');
  }
});

// ─── Start / Stop ─────────────────────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  const ok = await startWebcam();
  if (!ok) return;

  // Reset liveness state
  isLive = false;
  currentBlinkCount = 0;
  livenessBadge.className = 'liveness-badge badge-checking';
  livenessBadge.textContent = 'Verifying…';
  livenessMessage.textContent = 'Please look at the camera and blink naturally.';
  blinkDots.forEach(d => d.classList.remove('done'));
  recognitionPanel.classList.add('hidden');

  // Reset backend session
  try {
    await fetch(`${API_BASE}/verify/reset-liveness`, { method: 'POST' });
  } catch { /* ignore */ }

  captureIntervalId = setInterval(captureAndVerify, CAPTURE_INTERVAL_MS);
  pollIntervalId    = setInterval(refreshAttendance, RECORDS_POLL_MS);
  await refreshAttendance();

  btnStart.classList.add('hidden');
  btnStop.classList.remove('hidden');
});

btnStop.addEventListener('click', () => {
  clearInterval(captureIntervalId);
  clearInterval(pollIntervalId);
  captureIntervalId = null;
  pollIntervalId    = null;
  stopWebcam();

  livenessBadge.className = 'liveness-badge badge-pending';
  livenessBadge.textContent = 'Waiting…';
  livenessMessage.textContent = 'Start the camera to begin.';
  blinkDots.forEach(d => d.classList.remove('done'));

  btnStop.classList.add('hidden');
  btnStart.classList.remove('hidden');
});

// ─── Register Modal ───────────────────────────────────────────────────────────
btnRegisterToggle.addEventListener('click', async () => {
  registerModal.classList.remove('hidden');
  registerMsg.textContent = '';
  registerName.value = '';
  try {
    registerStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    registerVideo.srcObject = registerStream;
    await registerVideo.play();
  } catch (err) {
    registerMsg.textContent = `Camera error: ${err.message}`;
  }
});

function closeRegisterModal() {
  registerModal.classList.add('hidden');
  if (registerStream) {
    registerStream.getTracks().forEach(t => t.stop());
    registerStream = null;
  }
  registerVideo.srcObject = null;
}

btnModalClose.addEventListener('click', closeRegisterModal);
registerModal.addEventListener('click', e => {
  if (e.target === registerModal) closeRegisterModal();
});

btnCaptureReg.addEventListener('click', async () => {
  const name = registerName.value.trim();
  if (!name) {
    registerMsg.textContent = '⚠️ Please enter a name first.';
    return;
  }

  if (!registerStream) {
    registerMsg.textContent = '⚠️ Camera not available.';
    return;
  }

  const imageB64 = frameToBase64(registerVideo, registerCanvas);

  btnCaptureReg.disabled = true;
  btnCaptureReg.textContent = 'Registering…';
  registerMsg.textContent = '';

  try {
    const res  = await fetch(`${API_BASE}/register`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ image: imageB64, name }),
    });
    const data = await res.json();

    if (res.ok && data.success) {
      registerMsg.textContent = `✅ ${data.message}`;
      showToast(`Registered: ${name}`, 'success');
      await checkHealth(); // refresh known faces count
      setTimeout(closeRegisterModal, 1500);
    } else {
      registerMsg.textContent = `❌ ${data.detail ?? data.message ?? 'Registration failed.'}`;
    }
  } catch (err) {
    registerMsg.textContent = `❌ Network error: ${err.message}`;
  } finally {
    btnCaptureReg.disabled = false;
    btnCaptureReg.textContent = '📸 Capture & Register';
  }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
(async function init() {
  await checkHealth();
  await refreshAttendance();
  // Re-check health every 30s
  setInterval(checkHealth, 30_000);
})();
