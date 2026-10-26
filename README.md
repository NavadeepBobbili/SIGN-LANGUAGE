# Action to Text

Action to Text is a browser-based, local webcam prototype that detects a limited set of static hand poses and adds the matched letters or numbers to text.

## Important scope

This is **not** full sign-language translation and it is not a validated ISL or ASL classifier. It uses MediaPipe hand landmarks plus simple pose rules. It supports only the on-screen demo poses and needs consistent lighting, an upright hand, and the palm facing the camera.

## Features

- Modern responsive **ACTION TO TEXT** UI
- Explicit Start/Stop camera control
- Local MediaPipe 21-hand-landmark overlay
- Number demo: 1, 2, 3, 4, 5
- Letter demo: A, B, F, I, L, V, W, Y
- Stable-pose acceptance and adjustable match threshold
- Local-only processing; no camera video is stored or uploaded

## Install

Install Node.js LTS, then open a terminal in this project root and run:

```powershell
npm install
py -m http.server 5500
```

Open `http://localhost:5500/frontend/` in Chrome or Edge. Use a local server rather than opening the HTML file directly.

## Use

1. Select **Start Camera** and grant browser permission.
2. Choose **Numbers** or **Letters**.
3. Hold one supported hand pose steady for about half a second.
4. The token is appended to the text output once. Lower or change your hand before repeating it.

## Project files

```text
Sign AI/
├── frontend/
│   ├── index.html
│   ├── css/styles.css
│   ├── js/app.js
│   └── assets/models/hand_landmarker.task
├── package.json
├── package-lock.json        # created by npm install
└── README.md
```

## Troubleshooting

- **Model or module not found:** run `npm install` from the project root, then start the local server from that same root.
- **Camera permission denied:** click Chrome's camera icon in the address bar, allow access, and refresh.
- **No match:** show the entire upright hand in even lighting; use only the listed static poses.
- **Another app owns the camera:** close Zoom, Teams, or the Windows Camera app and retry.

## Next step

For real sign-language recognition, collect consented landmark examples for an explicitly defined vocabulary, train a classifier, evaluate it with people not represented in the training data, and add temporal models for dynamic signs.
