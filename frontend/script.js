const canvas = document.getElementById("main-canvas");
const ctx = canvas.getContext("2d");
const predictionBox = document.getElementById("prediction-box");
const confidenceBox = document.getElementById("confidence");
const statusElement = document.getElementById("status");
const modelSelect = document.getElementById("model-select");

let isDrawing = false;

function initCanvas() {
  ctx.fillStyle = "#1a1a1a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
initCanvas();

function isCanvasEmpty() {
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 0; i < imgData.length; i += 4) {
    if (imgData[i] > 10) return false;
  }
  return true;
}

function resetUI() {
  ctx.fillStyle = "#1a1a1a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  predictionBox.innerText = "–";
  confidenceBox.innerHTML = "Ready to predict";
  const table = document.querySelector("#prediction-table tbody");
  for (let i = 0; i < 10; i++) table.rows[i].cells[1].innerText = "—";
  statusElement.textContent = "Draw a digit to begin";
  statusElement.className = "status-bar";
}

// Adjust the line width style
function drawStart(e) {
  isDrawing = true;
  ctx.strokeStyle = "white";
  ctx.lineWidth = 8;
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.beginPath();
}

function drawMove(e) {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.moveTo(x, y);
}

function drawEnd() {
  isDrawing = false;
  if (!isCanvasEmpty()) sendCanvasToAPI(modelSelect.value);
}


// API request
async function sendCanvasToAPI(model) {
  if (isCanvasEmpty()) return;
  statusElement.textContent = `Analyzing with ${
    model === "lenet" ? "LeNet" : "ResNet"
  }...`;
  statusElement.className = "status-bar predicting";

  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "digit.png");

    try {
      const response = await fetch(`http://localhost:8000/predict/${model}`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      updateUI(data);
      statusElement.textContent = "Prediction complete!";
      statusElement.className = "status-bar";
    } catch (err) {
      console.error("Prediction failed:", err);
      statusElement.textContent = "Connection error - check if API is running";
      statusElement.className = "status-bar error";
    }
  });
}

function updateUI(data) {
  predictionBox.innerText = data.khmer;
  confidenceBox.innerHTML = `${Math.round(data.confidence * 100)}% confident`;
  const table = document.querySelector("#prediction-table tbody");
  for (let i = 0; i < 10; i++) {
    table.rows[i].cells[1].innerText =
      Math.round(data.probabilities[i] * 100) + "%";
  }
}


// Event Listeners
canvas.addEventListener("mousedown", drawStart);
canvas.addEventListener("mousemove", drawMove);
canvas.addEventListener("mouseup", drawEnd);
canvas.addEventListener("mouseleave", drawEnd);

canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  drawStart(e.touches[0]);
});
canvas.addEventListener("touchmove", (e) => {
  e.preventDefault();
  drawMove(e.touches[0]);
});
canvas.addEventListener("touchend", (e) => {
  e.preventDefault();
  drawEnd();
});

document.getElementById("erase").addEventListener("click", resetUI);
modelSelect.addEventListener("change", resetUI);
