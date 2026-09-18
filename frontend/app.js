const fileBox = document.getElementById('fileBox');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const uploadPrompt = document.getElementById('uploadPrompt');
const webcam = document.getElementById('webcam');
const snapshotCanvas = document.getElementById('snapshotCanvas');
const scanBtn = document.getElementById('scanBtn');
const scannerLine = document.getElementById('scannerLine');
const hudResult = document.getElementById('hudResult');
const mainLabel = document.getElementById('mainLabel');
const latencyText = document.getElementById('latencyText');
const top3List = document.getElementById('top3List');
const voiceToggle = document.getElementById('voiceToggle');

const btnUploadMode = document.getElementById('btnUploadMode');
const btnCamMode = document.getElementById('btnCamMode');

let currentMode = 'upload'; // 'upload' or 'webcam'
let webcamStream = null;
let voiceEnabled = true;
let selectedBlob = null;

// Sci-Fi Beep Sound via Web Audio API
function playBeep() {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(800, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(1200, audioCtx.currentTime + 0.1);
        gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.1);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.12);
    } catch(e) {}
}

// AI Speech synthesis
function speakResult(text) {
    if (!voiceEnabled || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05;
    utterance.pitch = 0.9; // Slightly deep tech pitch
    window.speechSynthesis.speak(utterance);
}

voiceToggle.addEventListener('click', () => {
    voiceEnabled = !voiceEnabled;
    voiceToggle.innerText = voiceEnabled ? '🔊 AI Voice: ON' : '🔇 AI Voice: OFF';
});

// Mode Switching
btnUploadMode.addEventListener('click', () => {
    currentMode = 'upload';
    btnUploadMode.className = 'flex-1 py-2 text-xs uppercase tracking-wider rounded-xl bg-cyan-500/20 border border-cyan-500 text-cyan-300 font-bold';
    btnCamMode.className = 'flex-1 py-2 text-xs uppercase tracking-wider rounded-xl bg-slate-900 border border-slate-800 text-slate-400 font-bold';
    fileBox.classList.remove('hidden');
    webcam.classList.add('hidden');
    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }
});

btnCamMode.addEventListener('click', async () => {
    currentMode = 'webcam';
    btnCamMode.className = 'flex-1 py-2 text-xs uppercase tracking-wider rounded-xl bg-cyan-500/20 border border-cyan-500 text-cyan-300 font-bold';
    btnUploadMode.className = 'flex-1 py-2 text-xs uppercase tracking-wider rounded-xl bg-slate-900 border border-slate-800 text-slate-400 font-bold';
    fileBox.classList.add('hidden');
    webcam.classList.remove('hidden');

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
        webcam.srcObject = webcamStream;
    } catch (err) {
        alert('Webcam access denied or unavailable.');
    }
});

// File Upload Logic
fileBox.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        selectedBlob = e.target.files[0];
        previewImage.src = URL.createObjectURL(selectedBlob);
        previewImage.classList.remove('hidden');
        uploadPrompt.classList.add('hidden');
    }
});

// Trigger Scan
scanBtn.addEventListener('click', async () => {
    let blobToSend = null;

    if (currentMode === 'upload') {
        if (!selectedBlob) {
            alert('Please select or drop an image first!');
            return;
        }
        blobToSend = selectedBlob;
    } else {
        // Capture frame from webcam
        const canvas = snapshotCanvas;
        canvas.width = webcam.videoWidth || 640;
        canvas.height = webcam.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
        blobToSend = await new Promise(r => canvas.toBlob(r, 'image/jpeg'));
    }

    playBeep();
    scanBtn.disabled = true;
    scanBtn.innerText = 'NEURAL ANALYZING...';
    scannerLine.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', blobToSend, 'scan.jpg');

    try {
        const res = await fetch('/predict', { method: 'POST', body: formData });
        if (!res.ok) throw new Error('Inference failure');
        const data = await res.json();

        playBeep();
        hudResult.classList.remove('hidden');
        mainLabel.innerText = data.primary_class;
        latencyText.innerText = `${data.latency_ms} ms`;

        // Render Top-3 probability bars
        top3List.innerHTML = '';
        data.top_3.forEach((item, idx) => {
            const row = document.createElement('div');
            row.className = 'space-y-1';
            row.innerHTML = `
                <div class="flex justify-between text-xs text-slate-300">
                    <span>${idx + 1}. ${item.label}</span>
                    <span class="font-bold text-cyan-400">${item.confidence}%</span>
                </div>
                <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div class="bg-cyan-400 h-full rounded-full transition-all duration-700" style="width: ${item.confidence}%"></div>
                </div>
            `;
            top3List.appendChild(row);
        });

        speakResult(`Target identified: ${data.primary_class}, certainty ${data.confidence} percent.`);

    } catch (e) {
        alert(e.message);
    } finally {
        scannerLine.classList.add('hidden');
        scanBtn.disabled = false;
        scanBtn.innerText = '⚡ INITIATE SCAN';
    }
});