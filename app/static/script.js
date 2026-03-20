const canvas = document.getElementById('whiteboard');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const predLabel = document.getElementById('pred-label');
const confBar = document.getElementById('conf-bar');

let isDrawing = false;
let strokes = [];
let currentStroke = null;
let lastPredictTime = 0;
const PREDICT_DELAY = 400; // ms
const BRUSH_SIZE = 10;
let debounceTimer = null;

// Ensure smooth scaling visually
function resizeCanvas() {
    clearCanvas();
}

function clearCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
clearCanvas();

function setStatus(msg) {
    statusEl.innerHTML = msg;
}

function startStroke(e) {
    isDrawing = true;
    currentStroke = document.createElement('canvas');
    currentStroke.width = canvas.width;
    currentStroke.height = canvas.height;
    const sCtx = currentStroke.getContext('2d');
    sCtx.drawImage(canvas, 0, 0);
    strokes.push(currentStroke);
    
    ctx.beginPath();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = BRUSH_SIZE * 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    const rect = canvas.getBoundingClientRect();
    
    // Support mouse or touch exactly
    let clientX = e.clientX;
    let clientY = e.clientY;
    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    }
    
    // Scale coords to actual canvas size vs CSS bound size
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;
    
    ctx.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    
    let clientX = e.clientX;
    let clientY = e.clientY;
    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    }
    
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;
    
    ctx.lineTo(x, y);
    ctx.stroke();

    if (Date.now() - lastPredictTime > PREDICT_DELAY) {
        livePredict();
        lastPredictTime = Date.now();
    } else {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(livePredict, PREDICT_DELAY);
    }
}

function stopStroke() {
    if (!isDrawing) return;
    isDrawing = false;
    ctx.closePath();
}

canvas.addEventListener('mousedown', startStroke);
canvas.addEventListener('mousemove', draw);
window.addEventListener('mouseup', stopStroke);

canvas.addEventListener('touchstart', e => {
    e.preventDefault();
    startStroke(e);
});
canvas.addEventListener('touchmove', e => {
    e.preventDefault();
    draw(e);
});
canvas.addEventListener('touchend', e => {
    e.preventDefault();
    stopStroke();
});

function undo() {
    if (strokes.length > 0) {
        const lastCanvas = strokes.pop();
        clearCanvas();
        ctx.drawImage(lastCanvas, 0, 0);
        setTimeout(livePredict, 100);
    }
}

function clear() {
    strokes = [];
    clearCanvas();
    predLabel.textContent = "Prediction: —";
    confBar.style.width = "0%";
}

async function livePredict() {
    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ image: canvas.toDataURL('image/png') })
        });
        const data = await res.json();
        
        if (data.error) {
            if(data.error !== "Model not loaded") setStatus("⚠️ " + data.error);
            return;
        }
        
        predLabel.textContent = `Prediction: ${data.prediction}`;
        confBar.style.width = `${Math.round(data.confidence * 100)}%`;
        setStatus("Ready");
    } catch(err) {
        console.error(err);
    }
}

async function saveSample() {
    const label = document.getElementById('label-entry').value;
    setStatus("Saving...");
    try {
        const res = await fetch('/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ image: canvas.toDataURL('image/png'), label })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        setStatus("✅ " + data.message);
        clear();
    } catch(e) {
        setStatus("❌ " + e.message);
    }
}

async function retrain() {
    setStatus("🔁 Retraining model… please wait");
    try {
        const res = await fetch('/retrain', { method: 'POST' });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        setStatus("✅ " + data.message);
    } catch(e) {
        setStatus("❌ " + e.message);
    }
}

document.getElementById('btn-predict').onclick = livePredict;
document.getElementById('btn-undo').onclick = undo;
document.getElementById('btn-clear').onclick = clear;
document.getElementById('btn-save').onclick = saveSample;
document.getElementById('btn-retrain').onclick = retrain;

document.addEventListener('keydown', e => {
    if (e.key === 'Enter') livePredict();
    if (e.key === 'z' && e.ctrlKey) undo();
});
