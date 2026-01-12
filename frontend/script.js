
const canvas = document.getElementById("main-canvas");
const displayBox = document.getElementById("prediction");
const confidenceBox = document.getElementById("confidence");
const statusElement = document.getElementById("status");

const ctx = canvas.getContext("2d");

let isDrawing = false;

// ---------------------------
// Initialize canvas (BLACK background)
// ---------------------------
function initCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

initCanvas();

// ---------------------------
// Utility: check if canvas is empty
// ---------------------------
function isCanvasEmpty() {
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 0; i < imgData.length; i += 4) {
    // If any pixel is not black → drawing exists
    if (imgData[i] > 10) {
      return false;
    }
  }
  return true;
}

// ---------------------------
// Reset EVERYTHING 
// ---------------------------
function resetUI() {
  // Clear canvas
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Clear main prediction
  displayBox.innerText = "";
  confidenceBox.innerHTML = "&#8212";

  // Reset table
  const table = document.querySelector("#prediction-table tbody");
  for (let i = 0; i < 10; i++) {
    table.rows[i].cells[1].innerText = "-";
  }

  // Status
  statusElement.textContent = "API status: Ready";
}

// ---------------------------
// Drawing logic
// ---------------------------
function drawStart(e) {
  isDrawing = true;
  ctx.strokeStyle = "white";
  ctx.lineWidth = 12;              // 12 defualt seem it work great with this value lol !
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.beginPath();
}

function drawMove(e) {
  if (!isDrawing) return;

  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.moveTo(x, y);
}

function drawEnd() {
  isDrawing = false;

  // Do NOT predict if nothing was drawn
  if (isCanvasEmpty()) {
    return;
  }

  sendCanvasToAPI();
}

// ---------------------------
// Send canvas → FastAPI
// ---------------------------
async function sendCanvasToAPI() {
  if (isCanvasEmpty()) {
    console.log("Canvas empty, skip prediction");
    return;
  }

  statusElement.textContent = "API status: Predicting...";

  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "digit.png");

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      updateUI(data);

      statusElement.textContent = "API status: Ready";
    } catch (error) {
      console.error(error);
      statusElement.textContent = "API status: Error";
    }
  });
}

// ---------------------------
// Update UI from API response
// ---------------------------
function updateUI(data) {
  // Big predicted digit (Khmer)
  displayBox.innerText = data.khmer;

  // Confidence
  confidenceBox.innerHTML =
    `<strong>${Math.round(data.confidence * 100)}%</strong> confidence`;

  // Probability table
  const table = document.querySelector("#prediction-table tbody");
  for (let i = 0; i < 10; i++) {
    table.rows[i].cells[1].innerText =
      Math.round(data.probabilities[i] * 100) + "%";
  }
}

// ---------------------------
// Event listeners (mouse)
// ---------------------------
canvas.addEventListener("mousedown", drawStart);
canvas.addEventListener("mousemove", drawMove);
canvas.addEventListener("mouseup", drawEnd);
canvas.addEventListener("mouseleave", drawEnd);

// ---------------------------
// Event listeners (touch / mobile)
// ---------------------------
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

// ---------------------------
// Reset button
// ---------------------------
document.getElementById("erase").addEventListener("click", resetUI);
