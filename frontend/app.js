const API_URL = "/api/predict";

let selectedFile = null;

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const previewWrapper = document.getElementById("preview-wrapper");
const imagePreview = document.getElementById("image-preview");
const btnClear = document.getElementById("btn-clear");
const btnAnalyze = document.getElementById("btn-analyze");

const predClass = document.getElementById("pred-class");
const predVerdict = document.getElementById("pred-verdict");
const badgeStatus = document.getElementById("badge-status");
const scoreDog = document.getElementById("score-dog");
const barDog = document.getElementById("bar-dog");
const scoreCat = document.getElementById("score-cat");
const barCat = document.getElementById("bar-cat");

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("border-indigo-500", "bg-indigo-950/20");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("border-indigo-500", "bg-indigo-950/20");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("border-indigo-500", "bg-indigo-950/20");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

btnClear.addEventListener("click", resetSelection);

function handleFileSelect(file) {
    if (!file.type.startsWith("image/")) {
        alert("Please select a valid image file (PNG/JPG).");
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (event) => {
        imagePreview.src = event.target.result;
        previewWrapper.classList.remove("hidden");
        btnAnalyze.disabled = false;
        btnAnalyze.textContent = "⚡ Classify with Deep CNN";
    };
    reader.readAsDataURL(file);
}

function resetSelection() {
    selectedFile = null;
    fileInput.value = "";
    imagePreview.src = "";
    previewWrapper.classList.add("hidden");
    btnAnalyze.disabled = true;

    predClass.textContent = "--";
    predVerdict.textContent = "Upload a picture to begin inference";
    badgeStatus.textContent = "Waiting for Image";
    badgeStatus.className = "text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400";
    
    scoreDog.textContent = "0.00%";
    barDog.style.width = "0%";
    scoreCat.textContent = "0.00%";
    barCat.style.width = "0%";
}

btnAnalyze.addEventListener("click", async () => {
    if (!selectedFile) return;

    btnAnalyze.disabled = true;
    btnAnalyze.textContent = "⏳ Computing Feature Maps...";
    badgeStatus.textContent = "Inferring...";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || "Prediction failed");
        }

        const data = await response.json();

        const isCat = data.label === "Cat";
        predClass.textContent = isCat ? "🐱 Cat" : "🐶 Dog";
        predClass.className = isCat ? "text-3xl font-black text-purple-400" : "text-3xl font-black text-indigo-400";
        predVerdict.textContent = `${data.verdict} • ${data.latency_ms}ms`;

        badgeStatus.textContent = `Completed (${data.confidence}%)`;
        badgeStatus.className = "text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";

        scoreDog.textContent = `${data.dog_probability}%`;
        barDog.style.width = `${data.dog_probability}%`;

        scoreCat.textContent = `${data.cat_probability}%`;
        barCat.style.width = `${data.cat_probability}%`;

    } catch (err) {
        alert(`Inference failed: ${err.message}`);
        badgeStatus.textContent = "Failed";
    } finally {
        btnAnalyze.disabled = false;
        btnAnalyze.textContent = "⚡ Classify with Deep CNN";
    }
});