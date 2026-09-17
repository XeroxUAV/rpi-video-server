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

## Running the RTSP Server & Web Viewer (`server_rstp.py`)

[`server_rstp.py`](server_rstp.py) broadcasts standard RTSP 1.0 (RFC 2326 / RFC 7826) streams over UDP and TCP (interleaved) using in-memory H.264 PyAV encoding. It concurrently serves an interactive HTML5 companion web viewer ([`web/rstp.html`](web/rstp.html)) on HTTP port `8080`.

### Quick Start

```bash
python server_rstp.py
```

By default, the RTSP stream is available at `rtsp://0.0.0.0:8554/live` and the companion web viewer at `http://0.0.0.0:8080/`.

### CLI Arguments

| Argument | Type | Default | Choices | Description |
|---|---|---|---|---|
| `--host` | `str` | `0.0.0.0` | Any valid IP | Host IP address to bind both RTSP and HTTP servers to. |
| `--rtsp-port` | `int` | `8554` | Any valid port | RTSP server port for streaming (RFC 2326). |
| `--http-port` | `int` | `8080` | Any valid port | HTTP web viewer and WebSocket stream port. |
| `--fps` | `int` | `30` | `30`, `90` | Target camera capture and streaming framerate. |
| `--resolution` | `int` | `480` | `480`, `720`, `1080` | Vertical resolution: `480` (640x480), `720` (1280x720), or `1080` (1920x1080). |
| `--quality` | `int` | `60` | `1-100` | JPEG quality for in-browser web preview. |

### Examples

- **Standard HD RTSP Stream (720p @ 30 FPS):**
  ```bash
  python server_rstp.py --fps 30 --resolution 720
  ```

- **High-speed 90 FPS Stream on custom ports:**
  ```bash
  python server_rstp.py --fps 90 --resolution 480 --rtsp-port 8554 --http-port 8080
  ```

### Connecting External RTSP Players

The stream can be played directly by any standard RTSP client using the URL `rtsp://<PI_IP_ADDRESS>:8554/live`:

- **VLC Media Player (Low Caching):**
  ```bash
  vlc --network-caching=100 rtsp://<PI_IP>:8554/live
  ```

- **FFplay (Ultra Low Latency):**
  ```bash
  ffplay -fflags nobuffer -flags low_delay -rtsp_transport udp rtsp://<PI_IP>:8554/live
  ```

- **Python OpenCV:**
  ```python
  import cv2

  cap = cv2.VideoCapture("rtsp://<PI_IP>:8554/live")
  while cap.isOpened():
      ret, frame = cap.read()
      if ret:
          cv2.imshow("Drone RTSP", frame)
      if cv2.waitKey(1) == ord("q"):
          break
  cap.release()
  cv2.destroyAllWindows()
  ```

- **GStreamer / OBS:**
  ```bash
  gst-launch-1.0 rtspsrc location=rtsp://<PI_IP>:8554/live latency=50 ! decodebin ! autovideosink
  ```

### HTML5 RTSP Web Viewer (`web/rstp.html`)

Access `http://<PI_IP>:8080/` in any browser to:
- Watch the live video stream with zero external player installation.
- Monitor real-time FPS, stream resolution, and active RTSP client counts.
- Copy the full RTSP URL with 1-click.
- Copy pre-configured commands for VLC, FFplay, OpenCV, and GStreamer.

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
│  [CameraStreamTrack] / [H264Encoder & RTPPacketizer]         │
│  • Pulls latest frame on demand                              │
│  • WebRTC RTP / RTSP RFC 6184 H.264 packetization            │
│  • Encodes to H.264 via aiortc / PyAV                        │
│                               │                              │
│                               ▼                              │
│  [Broadcasting Engines]                                      │
│  • WebRTC : server_webrtc.py (Port 8080)                     │
│  • RTSP   : server_rstp.py   (Port 8554 + HTTP Port 8080)    │
└──────────────────────────────┬───────────────────────────────┘
                               │ WebRTC / RTSP (UDP/TCP)
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│     Desktop / Media Players   │ │      HTML5 Web Viewers        │
│ • client_webrtc.py (OpenCV)   │ │ • web/webrtc.html             │
│ • VLC / FFplay / OBS (RTSP)   │ │ • web/rstp.html               │
│ • Real-time display & record  │ │ • Real-time stats HUD         │
└───────────────────────────────┘ └───────────────────────────────┘
```
