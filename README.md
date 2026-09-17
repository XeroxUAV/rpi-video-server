# Raspberry Pi WebRTC Video Server & Client

High-performance, ultra-low-latency video streaming server and desktop/web viewers for Raspberry Pi camera modules. Powered by `picamera2`, `aiortc` (WebRTC over UDP / RTP / SRTP), and OpenCV.

---

## Features

- **Ultra-Low Latency:** WebRTC streaming over UDP eliminates TCP bufferbloat and head-of-line blocking.
- **Decoupled Architecture:** Dedicated acquisition thread for `picamera2` capturing direct BGR frames from the hardware ISP.
- **Multi-Peer Broadcasting:** Multiple viewers (browser or desktop) can connect simultaneously without degrading camera frame rates.
- **Hardware & Software Clients:**
  - **OpenCV Desktop Client ([`client_webrtc.py`](client_webrtc.py)):** Native Python client with V-Sync pacing and local MP4 recording support.
  - **HTML5 Web Viewer ([`web/webrtc.html`](web/webrtc.html)):** Zero-install browser player with real-time HUD metrics (FPS, bitrate, resolution, packet loss, jitter).
- **Stream Recording:** Optional non-blocking MP4 video recording on the client without affecting live WebRTC reception.

---

## Installation & Setup

### 1. Requirements

- **Server:** Raspberry Pi running Raspberry Pi OS (Bookworm or newer) with a camera module and `picamera2` installed.
- **Client / Development:** Python `>= 3.13` with `uv` or `pip`.

### 2. Install Dependencies

Using [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or using standard `pip`:

```bash
pip install -r <(uv pip compile pyproject.toml)
# or directly:
pip install aiortc aiohttp opencv-python websockets
```

> **Note on Raspberry Pi Native Libraries:**
> If installing `aiortc` / `av` manually on Raspberry Pi OS, ensure FFmpeg developer libraries are installed:
> ```bash
> sudo apt-get update
> sudo apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev
> ```

---

## Running the WebRTC Server (`server_webrtc.py`)

Run [`server_webrtc.py`](server_webrtc.py) on your Raspberry Pi. It initializes the camera module, begins background frame capture, and exposes an HTTP service for SDP signaling and the browser viewer.

### Quick Start

```bash
python server_webrtc.py
```

By default, the server binds to `0.0.0.0:8080`, running at **480p @ 90 FPS**.

### CLI Arguments

| Argument | Type | Default | Choices | Description |
|---|---|---|---|---|
| `--fps` | `int` | `90` | `30`, `90` | Target camera capture framerate. |
| `--resolution` | `int` | `480` | `480`, `720`, `1080` | Vertical resolution height: `480` (640x480), `720` (1280x720), or `1080` (1920x1080). |
| `--host` | `str` | `0.0.0.0` | Any valid IP | Host IP address to bind the HTTP / WebRTC signaling server to. |
| `--port` | `int` | `8080` | Any valid port | HTTP port for SDP negotiation and static web asset serving. |

### Examples

- **High-speed FPV / Drone Mode (480p @ 90 FPS):**
  ```bash
  python server_webrtc.py --fps 90 --resolution 480
  ```

- **HD Stream (720p @ 30 FPS):**
  ```bash
  python server_webrtc.py --fps 30 --resolution 720
  ```

- **Full HD Stream on custom port (1080p @ 30 FPS, Port 9000):**
  ```bash
  python server_webrtc.py --fps 30 --resolution 1080 --port 9000
  ```

> **Note:** Raspberry Pi camera sensors do not physically support 1080p at 90 FPS. Selecting `--fps 90 --resolution 1080` will cause the sensor to operate at its maximum physical limit (~30–50 FPS).

---

## Running the WebRTC Desktop Client (`client_webrtc.py`)

Run [`client_webrtc.py`](client_webrtc.py) on your client machine (desktop, laptop, or ground station). It performs an HTTP SDP handshake with the server, receives the WebRTC video track over UDP, renders it in an OpenCV window, and optionally records the stream to an MP4 file.

### Quick Start

```bash
python client_webrtc.py --host <PI_IP_ADDRESS>
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--host` | `str` | `172.21.35.248` | IP address of the Raspberry Pi running `server_webrtc.py`. |
| `--port` | `int` | `8080` | HTTP signaling port of the server. |
| `--display-fps` | `int` | `60` | Target display loop refresh rate (in Hz) to pace rendering to monitor V-Sync. |
| `--window-name` | `str` | `"Drone Camera (WebRTC)"` | Title displayed on the OpenCV window header. |
| `--record` | `str` / flag | `False` | Record incoming video stream to an MP4 file. Pass as a bare flag (`--record`) for automatic timestamped naming (`recording_<timestamp>.mp4`), or specify a custom filename (`--record flight.mp4`). |
| `--record-fps` | `float` | `None` | Framerate for the output MP4 video file. If omitted, defaults to `--display-fps` (or `30.0`). |

### Examples

- **Connect to Pi and display stream:**
  ```bash
  python client_webrtc.py --host 192.168.1.50
  ```

- **Record stream with auto-generated timestamp filename:**
  ```bash
  python client_webrtc.py --host 192.168.1.50 --record
  # Saves to recording_YYYYMMDD_HHMMSS.mp4
  ```

- **Record stream to a specific output file:**
  ```bash
  python client_webrtc.py --host 192.168.1.50 --record flight_mission_01.mp4
  ```

- **High-refresh rate display (90 Hz) with recording:**
  ```bash
  python client_webrtc.py --host 192.168.1.50 --display-fps 90 --record flight_90fps.mp4 --record-fps 90
  ```

### Controls & UI

- **Exit:** Press <kbd>q</kbd> or click the window's close (<kbd>X</kbd>) button.
- **HUD Overlay:** Displays incoming WebRTC reception FPS, actual display render FPS, and stream resolution.
- **Recording Status:** When `--record` is active, a red indicator dot and frame count (`REC: <count>`) are displayed in the HUD. The recording worker safely finalizes and flushes the MP4 file on exit.

---

## Web Browser Viewer (`web/webrtc.html`)

For a zero-installation experience on phones, tablets, or computers:

1. Start `server_webrtc.py` on the Raspberry Pi.
2. Open any modern web browser and navigate to:
   ```text
   http://<PI_IP_ADDRESS>:8080/
   ```
3. Click **Connect** (auto-connects when opened directly from the Pi's server address).
4. View live video with sub-50ms latency using browser hardware acceleration.

### Web Viewer Features

- **Protocol:** Native WebRTC RTP/SRTP.
- **Real-Time HUD:** Live FPS counter, bitrate (Mbps / kbps), stream resolution, packet loss count, and jitter.
- **Controls:** Connect / Disconnect button and Fullscreen toggle.

---

## Architecture Summary

```
┌──────────────────────────────────────────────────────────────┐
│                    Raspberry Pi (Server)                     │
│                                                              │
│  [CameraHub]  (Dedicated capture thread)                     │
│  • Picamera2 ISP capture (BGR888 @ 30/90 FPS)                │
│  • Atomic frame buffer (zero queuing delay)                  │
│                               │                              │
│                               ▼                              │
│  [CameraStreamTrack]                                         │
│  • Pulls latest frame on demand                              │
│  • av.VideoFrame conversion + 90kHz PTS timestamps           │
│  • Encodes to H.264 / VP8 via aiortc                         │
│                               │                              │
│                               ▼                              │
│  [aiohttp Server]                                            │
│  • POST /offer : Single-roundtrip SDP exchange               │
│  • GET  /      : Serves web/webrtc.html static viewer        │
└──────────────────────────────┬───────────────────────────────┘
                               │ WebRTC (UDP)
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│     Desktop OpenCV Client     │ │      HTML5 Web Viewer         │
│      (client_webrtc.py)       │ │      (web/webrtc.html)        │
│ • SDP handshake via HTTP      │ │ • Hardware-accelerated <video>│
│ • aiortc track consumer       │ │ • Real-time WebRTC stats HUD  │
│ • OpenCV V-Sync display       │ │ • Sub-50ms playback           │
│ • Optional MP4 file recording │ │ • Zero install required       │
└───────────────────────────────┘ └───────────────────────────────┘
```
