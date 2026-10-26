import { FilesetResolver, HandLandmarker } from '/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs';

const webcam = document.querySelector('#webcam');
const canvas = document.querySelector('#landmark-canvas');
const context = canvas.getContext('2d');
const cameraPanel = document.querySelector('#camera-panel');
const cameraPlaceholder = document.querySelector('#camera-placeholder');
const cameraError = document.querySelector('#camera-error');
const cameraBadge = document.querySelector('#camera-badge');
const cameraStatus = document.querySelector('#camera-status');
const fpsOutput = document.querySelector('#fps-output');
const startButton = document.querySelector('#start-camera');
const stopButton = document.querySelector('#stop-camera');
const tokenOutput = document.querySelector('#detected-token');
const detailOutput = document.querySelector('#detected-detail');
const scoreOutput = document.querySelector('#match-score');
const scoreBar = document.querySelector('#match-bar');
const threshold = document.querySelector('#threshold');
const thresholdValue = document.querySelector('#threshold-value');
const textOutput = document.querySelector('#text-output');
const acceptanceMessage = document.querySelector('#acceptance-message');
const clearButton = document.querySelector('#clear-text');
const modeButtons = document.querySelectorAll('.mode-button');

const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
let handLandmarker;
let stream;
let animationFrame;
let lastVideoTime = -1;
let activeMode = 'numbers';
let stableToken = null;
let stableFrames = 0;
let lastAccepted = null;
let noHandFrames = 0;
let fpsFrames = 0;
let fpsStart = 0;

function distance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

function fingerExtended(points, tip, pip) {
  return points[tip].y < points[pip].y - 0.025;
}

function thumbExtended(points) {
  return distance(points[4], points[5]) > distance(points[2], points[5]) * 1.18;
}

function classifyPose(points, mode) {
  const index = fingerExtended(points, 8, 6);
  const middle = fingerExtended(points, 12, 10);
  const ring = fingerExtended(points, 16, 14);
  const pinky = fingerExtended(points, 20, 18);
  const thumb = thumbExtended(points);
  const fingers = [index, middle, ring, pinky];
  const count = fingers.filter(Boolean).length;
  const pinched = distance(points[4], points[8]) < distance(points[0], points[9]) * 0.38;

  if (mode === 'numbers') {
    if (count === 4 && thumb) return { token: '5', score: 0.91, hint: 'Open hand' };
    if (count === 4 && !thumb) return { token: '4', score: 0.86, hint: 'Four fingers' };
    if (index && middle && ring && !pinky) return { token: '3', score: 0.88, hint: 'Index, middle and ring fingers' };
    if (index && middle && !ring && !pinky) return { token: '2', score: 0.89, hint: 'Index and middle fingers' };
    if (index && !middle && !ring && !pinky) return { token: '1', score: 0.90, hint: 'Index finger' };
    return null;
  }

  if (pinched && middle && ring && pinky) return { token: 'F', score: 0.83, hint: 'Thumb and index finger touch' };
  if (thumb && index && !middle && !ring && !pinky) return { token: 'L', score: 0.87, hint: 'Thumb and index finger' };
  if (thumb && !index && !middle && !ring && pinky) return { token: 'Y', score: 0.84, hint: 'Thumb and pinky finger' };
  if (!thumb && index && middle && ring && !pinky) return { token: 'W', score: 0.86, hint: 'Three raised fingers' };
  if (!thumb && index && middle && !ring && !pinky) return { token: 'V', score: 0.87, hint: 'Index and middle fingers' };
  if (!thumb && !index && !middle && !ring && pinky) return { token: 'I', score: 0.82, hint: 'Pinky finger' };
  if (!thumb && index && middle && ring && pinky) return { token: 'B', score: 0.80, hint: 'Four fingers, thumb folded' };
  if (!thumb && count === 0) return { token: 'A', score: 0.76, hint: 'Closed fist' };
  return null;
}

function setStatus(message, live = false) {
  cameraStatus.innerHTML = `<i></i> ${message}`;
  cameraPanel.classList.toggle('live', live);
  cameraBadge.textContent = live ? 'Live' : 'Idle';
  cameraBadge.classList.toggle('live', live);
}

function showError(message) {
  cameraError.textContent = message;
  cameraError.hidden = false;
}

function clearError() { cameraError.hidden = true; cameraError.textContent = ''; }

function drawLandmarks(results) {
  context.clearRect(0, 0, canvas.width, canvas.height);
  results.landmarks.forEach((points) => {
    context.lineWidth = 3;
    context.strokeStyle = '#6cf1d0';
    context.fillStyle = '#ffffff';
    CONNECTIONS.forEach(([a, b]) => {
      context.beginPath();
      context.moveTo(points[a].x * canvas.width, points[a].y * canvas.height);
      context.lineTo(points[b].x * canvas.width, points[b].y * canvas.height);
      context.stroke();
    });
    points.forEach((point) => {
      context.beginPath();
      context.arc(point.x * canvas.width, point.y * canvas.height, 4, 0, Math.PI * 2);
      context.fill();
    });
  });
}

function updateDetection(result) {
  const thresholdScore = Number(threshold.value) / 100;
  if (!result) {
    tokenOutput.textContent = '—';
    detailOutput.textContent = 'Show a supported static hand pose.';
    scoreOutput.textContent = '—';
    scoreBar.style.width = '0%';
    stableToken = null;
    stableFrames = 0;
    noHandFrames += 1;
    if (noHandFrames > 10) lastAccepted = null;
    return;
  }
  noHandFrames = 0;
  tokenOutput.textContent = result.token;
  detailOutput.textContent = result.hint;
  scoreOutput.textContent = `${Math.round(result.score * 100)}%`;
  scoreBar.style.width = `${Math.round(result.score * 100)}%`;
  if (result.token === stableToken) stableFrames += 1;
  else { stableToken = result.token; stableFrames = 1; }

  if (result.score < thresholdScore) {
    acceptanceMessage.textContent = 'Pose is below the selected threshold. Adjust your hand and try again.';
    return;
  }
  if (stableFrames < 7) {
    acceptanceMessage.textContent = `Hold ${result.token} steady…`;
    return;
  }
  if (lastAccepted !== result.token) {
    textOutput.value += result.token;
    lastAccepted = result.token;
    acceptanceMessage.textContent = `${result.token} added to text. Change or lower your hand before repeating it.`;
  }
}

function updateFps(now) {
  fpsFrames += 1;
  if (now - fpsStart >= 1000) {
    fpsOutput.textContent = `${fpsFrames} FPS`;
    fpsFrames = 0;
    fpsStart = now;
  }
}

function detectFrame(now) {
  if (!stream || webcam.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  if (webcam.currentTime !== lastVideoTime) {
    lastVideoTime = webcam.currentTime;
    const result = handLandmarker.detectForVideo(webcam, now);
    drawLandmarks(result);
    updateDetection(result.landmarks.length ? classifyPose(result.landmarks[0], activeMode) : null);
    setStatus(result.landmarks.length ? 'Hand detected' : 'Looking for hand…', true);
    updateFps(now);
  }
  animationFrame = requestAnimationFrame(detectFrame);
}

async function setupLandmarker() {
  if (handLandmarker) return;
  const vision = await FilesetResolver.forVisionTasks('/node_modules/@mediapipe/tasks-vision/wasm');
  const options = { baseOptions: { modelAssetPath: '/frontend/assets/models/hand_landmarker.task' }, runningMode: 'VIDEO', numHands: 1, minHandDetectionConfidence: 0.65, minHandPresenceConfidence: 0.65, minTrackingConfidence: 0.65 };
  try { handLandmarker = await HandLandmarker.createFromOptions(vision, { ...options, baseOptions: { ...options.baseOptions, delegate: 'GPU' } }); }
  catch { handLandmarker = await HandLandmarker.createFromOptions(vision, options); }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) { showError('Camera access needs a recent browser running on localhost or HTTPS.'); return; }
  clearError(); startButton.disabled = true; startButton.textContent = 'Starting…';
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    webcam.srcObject = stream; await webcam.play();
    canvas.width = webcam.videoWidth; canvas.height = webcam.videoHeight;
    setStatus('Loading local hand detector…', true); await setupLandmarker();
    fpsFrames = 0; fpsStart = performance.now(); stopButton.disabled = false; startButton.textContent = 'Camera Running';
    animationFrame = requestAnimationFrame(detectFrame);
  } catch (error) {
    stream?.getTracks().forEach((track) => track.stop()); stream = undefined; webcam.srcObject = null;
    const message = error.name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access in your browser and try again.' : error.name === 'NotFoundError' ? 'No camera was detected. Close other camera apps or connect a webcam.' : `Could not start Action to Text. ${error.message || 'Check that npm install has completed.'}`;
    showError(message); setStatus('Camera unavailable'); startButton.disabled = false; startButton.textContent = 'Start Camera';
  }
}

function stopCamera() {
  cancelAnimationFrame(animationFrame); stream?.getTracks().forEach((track) => track.stop()); stream = undefined; webcam.srcObject = null;
  context.clearRect(0, 0, canvas.width, canvas.height); tokenOutput.textContent = '—'; detailOutput.textContent = 'Show a supported static hand pose.'; scoreOutput.textContent = '—'; scoreBar.style.width = '0%';
  setStatus('Camera stopped'); fpsOutput.textContent = '00 FPS'; startButton.disabled = false; startButton.textContent = 'Start Camera'; stopButton.disabled = true; stableToken = null; stableFrames = 0;
}

modeButtons.forEach((button) => button.addEventListener('click', () => { activeMode = button.dataset.mode; modeButtons.forEach((item) => item.classList.toggle('active', item === button)); stableToken = null; stableFrames = 0; lastAccepted = null; acceptanceMessage.textContent = `${activeMode === 'numbers' ? 'Number' : 'Letter'} mode selected.`; }));
threshold.addEventListener('input', () => { thresholdValue.textContent = `${threshold.value}%`; });
clearButton.addEventListener('click', () => { textOutput.value = ''; acceptanceMessage.textContent = 'Text cleared.'; lastAccepted = null; });
startButton.addEventListener('click', startCamera);
stopButton.addEventListener('click', stopCamera);
window.addEventListener('beforeunload', stopCamera);
document.querySelector('#year').textContent = new Date().getFullYear();
