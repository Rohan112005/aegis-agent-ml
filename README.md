# AegisAgent — AI-Powered Network Anomaly Detection

## About the Project

AegisAgent is a cybersecurity project that uses Machine Learning to detect unusual network activity.

It captures network traffic, analyzes network flows, and identifies possible anomalies using a trained AI model.

The project also includes a web dashboard where users can monitor network activity, view alerts, and analyze traffic.

**Technologies used:** Python, PyTorch, Scapy, FastAPI, React, Vite, and optional LangGraph integration.

## Main Features

- **Live Network Monitoring:** Captures network packets using Scapy.
- **AI Anomaly Detection:** Uses a trained Autoencoder to identify unusual network traffic.
- **Network Flow Analysis:** Groups packets into flows for analysis.
- **Security Dashboard:** Displays monitoring status, packet counts, analyzed flows, and alerts.
- **Manual Traffic Analysis:** Allows users to submit network feature values for analysis.
- **Machine Learning Experiments:** Includes Random Forest models and evaluation reports.
- **Optional AI Investigation:** Supports an investigation workflow using LangGraph and Google Gemini.

## Project Structure

```text
aegis-agent-ml/
├── data/                  # Training and captured data
├── frontend/              # React dashboard
├── models/                # Trained ML models and thresholds
├── reports/               # Model evaluation results
├── src/
│   ├── agents/            # AI investigation workflow
│   ├── api/               # FastAPI backend
│   ├── ml_engine/         # Machine learning code
│   └── packet_capture/    # Live packet monitoring
├── tests/                 # Project tests
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Installation

### 1. Download the Project

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd aegis-agent-ml
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the actual GitHub repository link.

### 2. Install Python Dependencies

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For live packet capture on Windows, Scapy may also require Npcap and suitable capture permissions.

### 3. Start the Backend

Open a terminal in the project folder:

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

Backend health check:

http://127.0.0.1:8001/health

API documentation:

http://127.0.0.1:8001/docs

### 4. Start the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the dashboard:

http://localhost:5173/

Keep the backend and frontend terminals running.

## How Live Monitoring Works

1. Open the AegisAgent dashboard.
2. Navigate to the Live Traffic section.
3. Select a network interface or use the default interface.
4. Start monitoring.
5. Scapy captures network packets.
6. The backend groups packets into network flows.
7. The detection system analyzes the flows.
8. The dashboard displays results and potential anomaly alerts.

Live monitoring requires appropriate permissions and operating-system support.

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check backend and model status |
| `GET /api/dashboard` | Retrieve dashboard information |
| `GET /api/live/status` | Check live monitoring status |
| `GET /api/live/events` | Retrieve captured flow events |
| `POST /api/live/start` | Start live network monitoring |
| `POST /api/live/stop` | Stop live network monitoring |
| `POST /extract-features` | Extract network features |
| `POST /analyze-packet` | Analyze feature values using the model |

## Machine Learning Models

The project includes:

- **Autoencoder:** Learns patterns in network data and uses reconstruction error to detect anomalies.
- **Random Forest:** Used in additional classification experiments and model comparisons.

Model files are stored in the `models/` folder.

**Important:** Training and live detection must use compatible feature definitions, feature order, and preprocessing. The included packet feature mapping is a starter implementation and should not be assumed to match all CICIDS2017 flow statistics.

An anomaly means the model detected unusual behavior. It does not automatically mean a cyberattack occurred.

## Dataset

The project uses data from the CICIDS2017 dataset for machine learning experiments.

The large `Wednesday-workingHours.pcap_ISCX.csv` file is excluded from GitHub because of its size.

To reproduce training experiments, obtain the dataset from the official Canadian Institute for Cybersecurity CICIDS2017 source and place it in the expected local data folder.

## Optional AI Investigation

The project includes an optional LangGraph investigation workflow using Google Gemini.

To use it, configure a valid `GOOGLE_API_KEY` in your local environment.

Never upload API keys or secret credentials to GitHub.

## Responsible Use

AegisAgent is intended for cybersecurity learning, experimentation, and authorized network monitoring.

Only capture traffic from devices and networks that you own or have permission to monitor.

## Future Improvements

- Improve the compatibility between training features and live network features.
- Reduce false-positive anomaly alerts.
- Add detailed alert investigation and export options.
- Improve dashboard visualizations.
- Evaluate the models using more network traffic datasets.

## Project Status

The FastAPI backend, React dashboard, manual analysis endpoint, and live-monitoring endpoints have been tested locally.

Further testing is needed to evaluate detection accuracy and reliability across different networks and attack types.