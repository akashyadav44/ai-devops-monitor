 # AI-Based DevOps Monitoring System

LIVE LINK = https://ai-devops-monitor-rfsr.onrender.com/

A real-time monitoring platform that watches **infrastructure** (CPU, memory, disk, network) and **applications / websites** (availability, latency), detects anomalies with **machine learning**, **predicts failures before they happen**, explains the **root cause** and recommends a **solution** for every alert.

## 1. Problem statement
Traditional monitoring tools use fixed thresholds: they alert only *after* a server is already at 90% CPU or a website is already down. Teams get alert fatigue and learn about problems too late, and a raw alert does not say *why* it happened or *what to do*.

**Our system learns what "normal" looks like, flags abnormal behaviour, predicts when a resource will run out, and tells the engineer the probable cause and the fix.**

## 2. Features
| Feature | How it works |
|---|---|
| Live infrastructure metrics | CPU, memory, disk, network, disk I/O and top processes via `psutil`, sampled every 2 s |
| Anomaly detection | **Isolation Forest** (unsupervised ML) + robust z-score (median/MAD). An alert needs both to agree, which keeps false alarms low. The model retrains itself every 25 samples |
| Failure prediction | Sliding-window **linear regression**: slope (%/min), R² confidence and **ETA to 90%**, drawn as a dotted forecast on the chart |
| Website / service monitoring | HTTP health check every 3 s: status (UP / DEGRADED / DOWN), latency, uptime %, latency trend, SSL-certificate expiry. Any real URL can be added from the dashboard |
| Reason and solution for websites | Each check is split into phases (DNS, TCP connect, TLS, server wait, download, redirects). The slowest phase is the reason for a slow site; the error type (DNS failure, timeout, connection refused, SSL error, HTTP 4xx/5xx) is the reason for a down site. Each reason has a matching fix |
| Root cause and recommendation | For infrastructure anomalies, the metric with the largest deviation (in sigma) is blamed and mapped to a DevOps playbook (restart service, rotate logs, scale out, rate-limit ...) |
| Health score | 0-100 gauge combining resource pressure, anomalies and service health |
| Alerts | Critical / warning / info feed, each alert carries its recommended solution |
| Incident simulator | Inject CPU spike, memory leak, disk fill, traffic flood, or make any service slow / down |
| Auto Demo | One click runs a guided 3.5-minute demo of five incidents with on-screen talking points. Works on the built-in services or on any real URL you added |
| PDF report | One click downloads a report with the system snapshot, predictions, service status and every alert with its solution |
| DevOps integration | Prometheus-style `/metrics`, `/health`, and a Dockerfile |

## 3. Architecture
```
 +-----------+  every 2 s   +-------------------+     +------------------+
 | psutil    | -----------> | Collector thread  | --> | Ring buffer      |
 | (OS data) |              +-------------------+     | (last 400 points)|
 +-----------+                        |               +------------------+
 +-----------+  every 3 s             v
 | HTTP      | --> Service checker --> AI engine
 | probes    |     (phase timing)      1. Isolation Forest + z-score (anomalies)
 +-----------+                         2. Trend regression (time to threshold)
                                       3. Root cause + solution playbook
                                                |
                          Flask REST API (/api/state, /metrics, /api/report.pdf)
                                                |
                          Live dashboard (HTML + CSS + JavaScript canvas)
```

## 4. Tech stack
Python 3.11, Flask, psutil, scikit-learn (Isolation Forest), NumPy, ReportLab (PDF), HTML / CSS / JavaScript (Canvas), Docker.

## 5. How to run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open **http://localhost:5050**. The model trains automatically after about 80 seconds of data, so **start the app 5 minutes before presenting**. The port can be changed with the `PORT` environment variable.

Docker:
```bash
docker build -t ai-devops-monitor .
docker run -p 5050:5050 ai-devops-monitor
```

## 6. What is being monitored?
1. **Infrastructure** - the machine running the app (CPU, memory, disk, network, processes).
2. **Applications** - HTTP services. Three demo microservices (Auth, Payment, User API) run inside the app so the demo works offline. In the *Monitored Services* panel you can add **any real URL** (for example the college website) and it is checked the same way. Every row shows "Monitoring active", the number of checks and the time of the last check.

Normal operation uses real data. The simulator only layers a synthetic incident on top, so a failure can be shown live (the same idea as chaos engineering). Simulated failures on real URLs are labelled **SIMULATED** on the dashboard and never touch the real website.

## 7. Demo script (about 5 minutes)
1. Show the dashboard: everything is green, "Model trained on N samples" is visible.
2. Press **Auto Demo** (optionally pick a real URL in the "Demo on" selector). Read the talking point shown in the banner at each step:
   - Service DOWN: status, health score, reason and solution appear.
   - Service SLOW: latency anomaly, the slow phase is named.
   - Memory leak: forecast line and "hits 90% in X min" (prediction before any alert).
   - CPU spike: red anomaly dots and the AI insight.
   - Traffic flood: abnormal network traffic and rate-limit advice.
3. Press **Download PDF Report** and show the alerts with their solutions.
4. Show the real college website row: latency, uptime and the phase-wise breakdown.
5. Open `/metrics` to show the Prometheus-compatible output.

 
