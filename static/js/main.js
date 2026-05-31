// DOM Elements
const video = document.getElementById('webcam-video');
const canvas = document.getElementById('canvas-overlay');
const ctx = canvas.getContext('2d');
const streamImg = document.getElementById('video-stream-img');
const placeholder = document.getElementById('video-placeholder');
const btnToggleCam = document.getElementById('btn-toggle-cam');
const btnClearZone = document.getElementById('btn-clear-zone');
const btnResetStats = document.getElementById('btn-reset-stats');

// Form & Control Inputs
const filterSelect = document.getElementById('filter-select');
const confSlider = document.getElementById('conf-slider');
const confVal = document.getElementById('conf-val');
const toggleAudio = document.getElementById('toggle-audio');
const toggleMirror = document.getElementById('toggle-mirror');
const classCheckboxes = document.querySelectorAll('.class-checkbox');

// Stats Elements
const statusBadge = document.getElementById('status-badge');
const statusText = document.getElementById('status-text');
const valActiveDetect = document.getElementById('val-active-detect');
const valTotalAlert = document.getElementById('val-total-alert');
const valFps = document.getElementById('val-fps');
const logsList = document.getElementById('logs-list');
const appBody = document.getElementById('app-body');

// State Variables
let isStreaming = false;
let stream = null;
let polygonPoints = []; // Stores normalized coordinates: [[x_ratio, y_ratio], ...]
let chart = null;
let animationFrameId = null;
let lastProcessedTime = 0;
let audioCtx = null;
let alarmInterval = null;
let lastAlarmTime = 0;

// Initialize Chart.js
function initChart() {
    const chartCtx = document.getElementById('live-chart').getContext('2d');
    chart = new Chart(chartCtx, {
        type: 'line',
        data: {
            labels: [], // Timestamps
            datasets: [{
                label: 'Aktif Nesne Sayısı',
                data: [],
                borderColor: '#7d5fff',
                backgroundColor: 'rgba(125, 95, 255, 0.15)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 2,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#8e8ea8', font: { size: 9 } }
                },
                y: {
                    min: 0,
                    suggestedMax: 5,
                    ticks: { stepSize: 1, color: '#8e8ea8', font: { size: 9 } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            }
        }
    });
}

// Adjust Canvas sizing to match visual container
function resizeCanvas() {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    drawPolygon();
}

// Draw the current polygon zone on the canvas overlay
function drawPolygon() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (polygonPoints.length === 0) return;

    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#00ebc7'; // Neon cyan border
    ctx.fillStyle = 'rgba(0, 235, 199, 0.12)'; // Cyan semi-transparent fill

    ctx.beginPath();
    // Move to first point
    ctx.moveTo(polygonPoints[0][0] * canvas.width, polygonPoints[0][1] * canvas.height);
    
    // Draw dot for start point
    ctx.arc(polygonPoints[0][0] * canvas.width, polygonPoints[0][1] * canvas.height, 5, 0, 2 * Math.PI);
    ctx.fillStyle = '#7d5fff'; // Neon violet for starting point
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(polygonPoints[0][0] * canvas.width, polygonPoints[0][1] * canvas.height);
    for (let i = 1; i < polygonPoints.length; i++) {
        ctx.lineTo(polygonPoints[i][0] * canvas.width, polygonPoints[i][1] * canvas.height);
    }

    if (polygonPoints.length >= 3) {
        ctx.closePath();
        ctx.fillStyle = 'rgba(0, 235, 199, 0.12)';
        ctx.fill();
    }
    ctx.strokeStyle = '#00ebc7';
    ctx.stroke();

    // Draw other points
    for (let i = 1; i < polygonPoints.length; i++) {
        ctx.beginPath();
        ctx.arc(polygonPoints[i][0] * canvas.width, polygonPoints[i][1] * canvas.height, 4, 0, 2 * Math.PI);
        ctx.fillStyle = '#00ebc7';
        ctx.fill();
    }
}

// Handle canvas mouse click to define zone points
canvas.addEventListener('click', (e) => {
    if (!isStreaming) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Normalize coordinates (0.0 to 1.0) relative to canvas size
    const xRatio = x / canvas.width;
    const yRatio = y / canvas.height;
    
    polygonPoints.push([xRatio, yRatio]);
    drawPolygon();
});

// Clear Zone button listener
btnClearZone.addEventListener('click', () => {
    polygonPoints = [];
    drawPolygon();
    
    // Clear on canvas overlay
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// Update confidence label on slider input
confSlider.addEventListener('input', (e) => {
    confVal.textContent = e.target.value;
});

// Play a high pitched alarm sound using the Web Audio API (no assets needed)
function playAlarmSound() {
    if (!toggleAudio.checked) return;
    
    // Throttle sound to prevent audio system crash (min 400ms between beeps)
    const now = Date.now();
    if (now - lastAlarmTime < 400) return;
    lastAlarmTime = now;

    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(950, audioCtx.currentTime); // Beep frequency
    
    gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.25);
    
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    
    osc.start();
    osc.stop(audioCtx.currentTime + 0.25);
}

// Start/Stop Webcam
btnToggleCam.addEventListener('click', async () => {
    if (isStreaming) {
        stopWebcam();
    } else {
        await startWebcam();
    }
});

async function startWebcam() {
    try {
        // Try accessing user camera
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 480 },
                facingMode: 'user'
            } 
        });
        video.srcObject = stream;
        video.style.display = 'block';
        streamImg.style.display = 'block';
        placeholder.style.display = 'none';
        
        isStreaming = true;
        btnToggleCam.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-square"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg>
            İzlemeyi Durdur
        `;
        btnToggleCam.classList.remove('btn-primary');
        btnToggleCam.classList.add('btn-secondary');
        
        // Resize canvas to visual dimensions
        setTimeout(resizeCanvas, 500);
        
        // Start processing loop
        processFrameLoop();
    } catch (err) {
        console.error("Kamera başlatılamadı:", err);
        alert("Kamera erişimi sağlanamadı! Lütfen kamera izinlerini kontrol edin.");
    }
}

function stopWebcam() {
    isStreaming = false;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    video.srcObject = null;
    video.style.display = 'none';
    streamImg.style.display = 'none';
    placeholder.style.display = 'flex';
    
    btnToggleCam.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-play"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Kamerayı Başlat
    `;
    btnToggleCam.classList.remove('btn-secondary');
    btnToggleCam.classList.add('btn-primary');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Reset Status and Flash
    appBody.classList.remove('screen-flash-active');
    statusBadge.className = 'status-badge status-secure';
    statusText.textContent = 'Güvenli';
    valActiveDetect.textContent = '0';
    valFps.textContent = '0.0';
}

// Grab current video frame, convert to Base64, post to Flask API
async function processFrameLoop() {
    if (!isStreaming) return;

    const canvasTemp = document.createElement('canvas');
    canvasTemp.width = 480; // Gönderilen çözünürlük 480x360 yapıldı (transferi %44 hafifletir)
    canvasTemp.height = 360;
    const ctxTemp = canvasTemp.getContext('2d');
    
    // Draw current video frame to temp canvas
    ctxTemp.drawImage(video, 0, 0, canvasTemp.width, canvasTemp.height);
    
    // Convert to Base64 JPEG
    const base64Img = canvasTemp.toDataURL('image/jpeg', 0.55); // Kalite %55 yapıldı (veri boyutunu yarı yarıya düşürür)

    // Compile active classes
    const targetClasses = [];
    classCheckboxes.forEach(cb => {
        if (cb.checked) targetClasses.push(cb.value);
    });

    const payload = {
        image: base64Img,
        polygon: polygonPoints,
        filter: filterSelect.value,
        confidence: confSlider.value,
        target_classes: targetClasses,
        mirror: toggleMirror.checked
    };

    try {
        const response = await fetch('/process_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}`);
        }

        const data = await response.json();
        
        // 1. Update Video Stream Image source with processed image
        streamImg.src = data.image;
        
        // 2. Update Status Indicators (Secure vs Alert)
        if (data.intrusion_detected) {
            statusBadge.className = 'status-badge status-alarm';
            statusText.textContent = 'İHLAL TESPİTİ!';
            appBody.classList.add('screen-flash-active');
            playAlarmSound();
        } else {
            statusBadge.className = 'status-badge status-secure';
            statusText.textContent = 'Güvenli';
            appBody.classList.remove('screen-flash-active');
        }

        // 3. Update Statistics
        valActiveDetect.textContent = data.detections.length;
        valTotalAlert.textContent = data.cumulative_violations;
        valFps.textContent = data.fps;

        // 4. Update Chart Data
        const timeNow = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        updateChart(timeNow, data.detections.length);

        // 5. Update Log Activity List
        updateLogs(data.logs);

    } catch (error) {
        console.error("Frame işleme hatası:", error);
    }

    // Dynamic self-scheduling based on response loop (approx 10 FPS targets)
    if (isStreaming) {
        setTimeout(processFrameLoop, 30);
    }
}

// Update Chart.js data smoothly
function updateChart(label, value) {
    if (!chart) return;
    
    chart.data.labels.push(label);
    chart.data.datasets[0].data.push(value);
    
    // Limit data points to show a sliding window of the last 12 entries
    if (chart.data.labels.length > 12) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    
    chart.update();
}

// Render Event Logs
function updateLogs(logs) {
    if (!logs) return;
    
    if (logs.length === 0) {
        logsList.innerHTML = `
            <li class="log-item" style="border-left-color: var(--color-secondary);">
                <span class="log-time">--:--:--</span>
                <span class="log-msg">Aktif alarm veya olay bulunmuyor.</span>
            </li>
        `;
        return;
    }

    let logsHTML = '';
    logs.forEach(log => {
        const itemClass = log.type === 'danger' ? 'danger' : '';
        logsHTML += `
            <li class="log-item ${itemClass}">
                <span class="log-time">${log.timestamp}</span>
                <span class="log-msg">${log.message}</span>
            </li>
        `;
    });
    logsList.innerHTML = logsHTML;
}

// Reset stats button listener
btnResetStats.addEventListener('click', async () => {
    try {
        const res = await fetch('/reset_stats', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
            valTotalAlert.textContent = '0';
            updateLogs([]);
            // Clear chart
            if (chart) {
                chart.data.labels = [];
                chart.data.datasets[0].data = [];
                chart.update();
            }
        }
    } catch (err) {
        console.error("İstatistikler sıfırlanamadı:", err);
    }
});

// App Initialization
window.addEventListener('DOMContentLoaded', () => {
    initChart();
    resizeCanvas();
    // Initialize Lucide Icons
    lucide.createIcons();
});

// Re-evaluate canvas boundaries on window resizing
window.addEventListener('resize', resizeCanvas);
