// State variables
let currentInputMode = 'upload';
let webcamStream = null;
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const previewWrapper = document.getElementById('preview-wrapper');
const imagePreview = document.getElementById('image-preview');
const uploadContainer = document.getElementById('upload-container');
const webcamContainer = document.getElementById('webcam-container');
const resultsPlaceholder = document.getElementById('results-placeholder');
const loadingIndicator = document.getElementById('loading-indicator');
const resultsContent = document.getElementById('results-content');
const webcamFeed = document.getElementById('webcam-feed');
const webcamCanvas = document.getElementById('webcam-canvas');
const webcamMsg = document.getElementById('webcam-msg');
const btnToggleCam = document.getElementById('btn-toggle-cam');
const btnCapture = document.getElementById('btn-capture');

// Input Mode Selection
function setInputMode(mode) {
    currentInputMode = mode;
    document.querySelectorAll('.toggle-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`btn-${mode}`).classList.add('active');
    
    if (mode === 'upload') {
        uploadContainer.classList.add('active');
        webcamContainer.classList.remove('active');
        stopWebcam();
    } else {
        uploadContainer.classList.remove('active');
        webcamContainer.classList.add('active');
        previewWrapper.style.display = 'none';
    }
}

// Drag and Drop Setup
if (dropZone) {
    dropZone.addEventListener('click', () => fileInput.click());
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            fileInput.files = files;
            handleFileSelect();
        }
    });
}

if (fileInput) {
    fileInput.addEventListener('change', handleFileSelect);
}

function handleFileSelect() {
    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewWrapper.style.display = 'block';
            uploadContainer.classList.remove('active');
            
            // Hide results when new image selected
            resultsPlaceholder.style.display = 'block';
            resultsContent.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
}

function clearPreview() {
    fileInput.value = '';
    imagePreview.src = '';
    previewWrapper.style.display = 'none';
    if (currentInputMode === 'upload') {
        uploadContainer.classList.add('active');
    }
    resultsPlaceholder.style.display = 'block';
    resultsContent.style.display = 'none';
}

// Webcam controls
async function toggleWebcam() {
    if (webcamStream) {
        stopWebcam();
    } else {
        try {
            webcamMsg.style.display = 'flex';
            webcamMsg.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Initializing Camera Feed...';
            
            webcamStream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } } 
            });
            
            webcamFeed.srcObject = webcamStream;
            webcamMsg.style.display = 'none';
            btnToggleCam.innerHTML = '<i class="fa-solid fa-power-off"></i> Stop Camera';
            btnToggleCam.classList.add('active');
            btnCapture.disabled = false;
        } catch (err) {
            console.error("Error accessing webcam: ", err);
            webcamMsg.style.display = 'flex';
            webcamMsg.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Camera Access Denied or Unavailable.';
        }
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    webcamFeed.srcObject = null;
    btnToggleCam.innerHTML = '<i class="fa-solid fa-power-off"></i> Start Camera';
    btnToggleCam.classList.remove('active');
    btnCapture.disabled = true;
    webcamMsg.style.display = 'flex';
    webcamMsg.innerHTML = '<i class="fa-solid fa-video-slash"></i> Camera Feed Stopped.';
}

function captureAndPredict() {
    if (!webcamStream) return;
    
    const width = webcamFeed.videoWidth;
    const height = webcamFeed.videoHeight;
    webcamCanvas.width = width;
    webcamCanvas.height = height;
    
    const ctx = webcamCanvas.getContext('2d');
    ctx.drawImage(webcamFeed, 0, 0, width, height);
    
    const dataURL = webcamCanvas.toDataURL('image/jpeg');
    
    // Stop camera after capture
    stopWebcam();
    
    // Show preview and upload
    imagePreview.src = dataURL;
    previewWrapper.style.display = 'block';
    
    // Trigger prediction automatically for captured image
    predictImage(null, dataURL);
}

function uploadAndPredict() {
    const file = fileInput.files[0];
    if (!file) return;
    predictImage(file, null);
}

function predictImage(file, base64Data) {
    // Show Loading
    resultsPlaceholder.style.display = 'none';
    resultsContent.style.display = 'none';
    loadingIndicator.style.display = 'block';
    
    const formData = new FormData();
    if (file) {
        formData.append('image', file);
    } else if (base64Data) {
        formData.append('image_base64', base64Data);
    } else {
        return;
    }
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loadingIndicator.style.display = 'none';
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || "Failed to analyze image.");
        }
    })
    .catch(err => {
        loadingIndicator.style.display = 'none';
        showError("Server communication error occurred.");
        console.error(err);
    });
}

function displayResults(data) {
    resultsContent.style.display = 'block';
    resultsPlaceholder.style.display = 'none';
    
    // Update labels
    const resClass = document.getElementById('result-class');
    const resRisk = document.getElementById('result-risk');
    const resConfidence = document.getElementById('result-confidence-text');
    const resRecommendation = document.getElementById('result-recommendation');
    
    resClass.innerText = data.predicted_class;
    resRisk.innerText = data.risk_level;
    resConfidence.innerText = `${data.confidence}%`;
    resRecommendation.innerText = data.recommendation;
    
    // Risk class style
    resRisk.className = 'val badge';
    if (data.risk_level === 'High') {
        resRisk.classList.add('high');
    } else if (data.risk_level === 'Medium') {
        resRisk.classList.add('medium');
    } else {
        resRisk.classList.add('low');
    }
    
    // Status box style
    const alertBox = document.getElementById('result-alert-box');
    const alertIcon = document.getElementById('result-alert-icon');
    const alertTitle = document.getElementById('result-status-title');
    
    alertBox.className = 'alert-box';
    if (data.is_cancer) {
        alertBox.classList.add('cancer');
        alertIcon.className = 'fa-solid fa-triangle-exclamation alert-icon';
        alertTitle.innerText = `Cancer Detected (${data.predicted_class})`;
    } else {
        alertBox.classList.add('benign');
        alertIcon.className = 'fa-solid fa-circle-check alert-icon';
        alertTitle.innerText = `No Cancer Detected (${data.predicted_class})`;
    }
    
    // Circular gauge
    const gaugeCircle = document.getElementById('gauge-fill-circle');
    const gaugeText = document.getElementById('gauge-value-text');
    
    const radius = gaugeCircle.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    gaugeCircle.style.strokeDasharray = `${circumference} ${circumference}`;
    
    // Animate fill
    const offset = circumference - (data.confidence / 100) * circumference;
    gaugeCircle.style.strokeDashoffset = offset;
    gaugeText.innerText = `${Math.round(data.confidence)}%`;
    
    // Render features table
    const tableBody = document.getElementById('features-table-body');
    tableBody.innerHTML = '';
    
    if (data.features) {
        // Group features conceptually for display
        for (const [key, value] of Object.entries(data.features)) {
            let group = 'Color';
            if (key.includes('glcm') || key.includes('contrast') || key.includes('homogeneity') || key.includes('energy') || key.includes('correlation')) {
                group = 'Texture';
            } else if (key.includes('contour') || key.includes('area') || key.includes('perimeter') || key.includes('circularity') || key.includes('solidity') || key.includes('eccentricity')) {
                group = 'Shape';
            }
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><span class="feature-group-badge ${group.toLowerCase()}">${group}</span></td>
                <td><code>${key}</code></td>
                <td><strong>${value}</strong></td>
            `;
            tableBody.appendChild(row);
        }
    }
}

function showError(errorMsg) {
    resultsPlaceholder.style.display = 'block';
    resultsContent.style.display = 'none';
    
    // Show alert or toast
    alert(`Error: ${errorMsg}`);
}

// Toggle features dropdown
function toggleFeaturesDropdown() {
    const dropdown = document.getElementById('features-table-container');
    const btn = document.querySelector('.features-toggle-btn');
    
    dropdown.classList.toggle('active');
    btn.classList.toggle('active');
}
