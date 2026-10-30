"""
AI-Based DevOps Monitoring System
---------------------------------
- Collects real system metrics (CPU, memory, disk, network) with psutil
- Detects anomalies with IsolationForest (unsupervised ML) + robust z-score check
- Predicts resource exhaustion (time-to-threshold) with trend regression
- Raises alerts, explains probable root cause, recommends remediation
- Exposes Prometheus-style /metrics so it plugs into real DevOps stacks
"""
import collections
import os
import random
import threading
import time
from datetime import datetime

import numpy as np
import psutil
from flask import Flask, Response, jsonify, send_from_directory
from sklearn.ensemble import IsolationForest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTERVAL = 2            # seconds between samples
WINDOW = 400            # samples kept in memory
MIN_TRAIN = 40          # samples needed before ML model trains
RETRAIN_EVERY = 25      # retrain cadence (samples)
THRESH = {"cpu": 90.0, "mem": 90.0, "disk": 90.0}
FEATURES = ["cpu", "mem", "disk_io", "net"]
# minimum "normal noise" per feature so tiny variance does not cause false alarms
SD_FLOOR = {"cpu": 4.0, "mem": 1.5, "disk_io": 2.0, "net": 80.0}

app = Flask(__name__)
lock = threading.Lock()
history = collections.deque(maxlen=WINDOW)
alerts = collections.deque(maxlen=40)
alert_log = collections.deque(maxlen=500)   # full incident history (used for the PDF report)
started_at = time.time()
sim = {"kind": None, "start": 0.0, "until": 0.0}
model = {"clf": None, "med": None, "sd": None, "samples": 0, "trained_at": None}
last_io = {"t": None, "disk": None, "net": None}
cooldown = {}
counter = {"n": 0}
last_insight = {"v": None, "t": 0.0}   # keeps the last AI insight visible for a while


# ----------------------------------------------------------------- collection
def sim_offsets(now):
    """Demo-only: synthetic incidents layered on top of real metrics."""
    o = {"cpu": 0.0, "mem": 0.0, "disk": 0.0, "disk_io": 0.0, "net": 0.0}
    kind = sim["kind"]
    if not kind:
        return o
    if now > sim["until"]:
        sim["kind"] = None
        return o
    t = now - sim["start"]
    if kind == "cpu_spike":
        o["cpu"] = 55 + random.uniform(-6, 6)
    elif kind == "memory_leak":
        o["mem"] = min(60.0, t * 0.8)
    elif kind == "disk_fill":
        o["disk"] = min(45.0, t * 0.55)
        o["disk_io"] = 18 + random.uniform(0, 10)
    elif kind == "net_flood":
        o["net"] = 4000 + random.uniform(0, 1500)
    return o


def sample():
    now = time.time()
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent
    disk_io = net = 0.0
    try:
        dio, nio = psutil.disk_io_counters(), psutil.net_io_counters()
        if last_io["t"] and dio and nio:
            dt = max(now - last_io["t"], 0.001)
            disk_io = ((dio.read_bytes + dio.write_bytes) - last_io["disk"]) / dt / 1e6
            net = ((nio.bytes_sent + nio.bytes_recv) - last_io["net"]) / dt / 1e3
        if dio and nio:
            last_io.update(t=now, disk=dio.read_bytes + dio.write_bytes,
                           net=nio.bytes_sent + nio.bytes_recv)
    except Exception:
        pass
    o = sim_offsets(now)
    return {
        "ts": now,
        "cpu": round(min(100.0, cpu + o["cpu"]), 2),
        "mem": round(min(100.0, mem + o["mem"]), 2),
        "disk": round(min(100.0, disk + o["disk"]), 2),
        "disk_io": round(max(0.0, disk_io + o["disk_io"]), 2),
        "net": round(max(0.0, net + o["net"]), 2),
        "anom": False, "z": {}, "score": 0.0,
    }


# --------------------------------------------------------------- AI: anomaly
def robust_stats(data):
    med = np.median(data, axis=0)
    mad = np.median(np.abs(data - med), axis=0) * 1.4826
    sd = np.maximum(mad, np.array([SD_FLOOR[f] for f in FEATURES]))
    return med, sd


def train():
    recs = list(history)[:-8]          # keep the freshest points out of training
    data = np.array([[r[f] for f in FEATURES] for r in recs])
    med, sd = robust_stats(data)
    clf = IsolationForest(n_estimators=120, contamination=0.03, random_state=42)
    clf.fit((data - med) / sd)
    model.update(clf=clf, med=med, sd=sd, samples=len(recs),
                 trained_at=datetime.now().strftime("%H:%M:%S"))


def detect(rec):
    if model["clf"] is None:
        return
    x = np.array([[rec[f] for f in FEATURES]])
    zs = (x - model["med"]) / model["sd"]
    rec["z"] = {f: round(float(zs[0][i]), 2) for i, f in enumerate(FEATURES)}
    rec["score"] = round(float(model["clf"].decision_function(zs)[0]), 3)
    is_out = model["clf"].predict(zs)[0] == -1
    # two-stage check: ML outlier AND statistically large deviation -> few false alarms
    rec["anom"] = bool(is_out and np.max(np.abs(zs)) > 3.0) or bool(np.max(zs) > 6.0)


# ------------------------------------------------------------ AI: prediction
def predict(key, n=45):
    recs = list(history)[-n:]
    if len(recs) < 15:
        return None
    y = np.array([r[key] for r in recs])
    x = np.arange(len(y))
    m, b = np.polyfit(x, y, 1)
    fit = m * x + b
    r2 = 1 - ((y - fit) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9)
    cur = float(fit[-1])
    eta = None
    if m > 0.02 and r2 > 0.5 and cur < THRESH[key]:
        eta = (THRESH[key] - cur) / m * INTERVAL / 60.0      # minutes
    fc = [round(float(min(100, max(0, b + m * (len(y) + i)))), 2) for i in range(0, 30)]
    status = "ok"
    if y[-1] >= THRESH[key]:
        status = "critical"
    elif eta is not None and eta < 10:
        status = "warning"
    return {"slope_per_min": round(float(m) * 60 / INTERVAL, 2),
            "eta_min": None if eta is None else round(eta, 1),
            "confidence": round(float(max(0, min(1, r2))), 2),
            "forecast": fc, "status": status}


RECO = {
    "cpu": ("CPU saturation", "Identify the top process, scale out replicas / enable autoscaling, "
            "check for runaway loops or heavy cron jobs."),
    "mem": ("Memory pressure / possible leak", "Restart the leaking service (rolling), take a heap dump, "
            "raise memory limits, enable OOM alerts."),
    "disk": ("Disk filling up", "Rotate and compress logs, clean temp/docker images, "
             "expand the volume or move data to object storage."),
    "disk_io": ("Heavy disk I/O", "Check for log storms or backup jobs; move to faster storage or throttle writers."),
    "net": ("Abnormal network traffic", "Inspect connections for DDoS/scan patterns, enable rate limiting, "
            "check firewall rules and CDN."),
}


def insight(rec):
    if rec["anom"] and rec["z"]:
        top = max(rec["z"], key=lambda k: rec["z"][k])
        cause, action = RECO[top]
        return {"type": "anomaly", "metric": top, "cause": cause, "action": action,
                "detail": f"{top.upper()} deviates {rec['z'][top]:.1f}σ from learned baseline"}
    return None


def push_alert(level, msg, key, action=None):
    if time.time() - cooldown.get(key, 0) < 25:
        return
    cooldown[key] = time.time()
    a = {"time": datetime.now().strftime("%H:%M:%S"), "level": level, "msg": msg, "action": action}
    alerts.appendleft(a)
    alert_log.append(a)


def evaluate(rec):
    counter["n"] += 1
    n = len(history)
    if n >= MIN_TRAIN and (model["clf"] is None or counter["n"] % RETRAIN_EVERY == 0):
        try:
            train()
        except Exception as e:     # never let ML break the collector
            print("train error", e)
    detect(rec)
    ins = insight(rec)
    if ins:
        last_insight.update(v=ins, t=time.time())
        push_alert("critical", f"Anomaly detected: {ins['cause']} ({ins['detail']})",
                   "anom-" + ins["metric"], ins["action"])
    for k in ("cpu", "mem", "disk"):
        if rec[k] >= THRESH[k]:
            push_alert("critical", f"{k.upper()} crossed {THRESH[k]:.0f}% (now {rec[k]:.0f}%)",
                       "thr-" + k, RECO[k][1])
    for k in ("cpu", "mem", "disk"):
        p = predict(k)
        if p and p["eta_min"] is not None and p["eta_min"] < 5:
            push_alert("warning", f"Prediction: {k.upper()} will hit {THRESH[k]:.0f}% in ~{p['eta_min']} min "
                                  f"(trend +{p['slope_per_min']}%/min)", "pred-" + k, RECO[k][1])


def collector():
    psutil.cpu_percent(interval=None)
    sample()                                  # prime counters
    while True:
        time.sleep(INTERVAL)
        rec = sample()
        with lock:
            history.append(rec)
            evaluate(rec)


# ------------------------------------------------------------------ routes
_proc_cache = {"t": 0, "v": []}


def top_processes():
    if time.time() - _proc_cache["t"] < 4:
        return _proc_cache["v"]
    rows = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            rows.append({"pid": p.info["pid"], "name": (p.info["name"] or "?")[:24],
                         "cpu": round(p.info["cpu_percent"] or 0, 1),
                         "mem": round(p.info["memory_percent"] or 0, 1)})
        except Exception:
            continue
    rows.sort(key=lambda r: (r["cpu"], r["mem"]), reverse=True)
    _proc_cache.update(t=time.time(), v=rows[:6])
    return _proc_cache["v"]


def health_score(rec):
    s = 100.0
    s -= max(0, rec["cpu"] - 60) * 0.8
    s -= max(0, rec["mem"] - 70) * 1.0
    s -= max(0, rec["disk"] - 80) * 1.2
    if rec["anom"]:
        s -= 15
    return int(max(0, min(100, s)))


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.route("/api/state")
def state():
    with lock:
        recs = list(history)[-120:]
        if not recs:
            return jsonify({"ready": False})
        cur = recs[-1]
        preds = {k: predict(k) for k in ("cpu", "mem", "disk")}
        score = health_score(cur)
        score = max(0, score - sum(25 if s["status"] == "down" else 10 if s["status"] == "degraded" else 0
                                   for s in services.values()))
        status = "critical" if score < 50 else "warning" if score < 80 else "healthy"
        if any(p and p["status"] == "critical" for p in preds.values()) or \
                any(s["status"] == "down" for s in services.values()):
            status = "critical"
        ins = insight(cur) or next(
            ({"type": "prediction", "metric": k, "cause": RECO[k][0], "action": RECO[k][1],
              "detail": f"{k.upper()} predicted to hit {THRESH[k]:.0f}% in {p['eta_min']} min"}
             for k, p in preds.items() if p and p["eta_min"] is not None and p["eta_min"] < 10), None)
        ins = ins or svc_insight()
        if ins is None and last_insight["v"] and time.time() - last_insight["t"] < 40:
            ins = dict(last_insight["v"], recent=True)       # sticky: show fix for 40 s after anomaly
        return jsonify({
            "ready": True, "health": score, "status": status, "current": cur,
            "series": {k: [r[k] for r in recs] for k in ("cpu", "mem", "disk", "disk_io", "net", "anom")},
            "predictions": preds, "insight": ins, "alerts": list(alerts),
            "procs": top_processes(), "services": svc_list(), "demo": dict(demo),
            "model": {"algo": "IsolationForest + robust z-score; Linear trend forecast",
                      "trained": model["clf"] is not None, "samples": model["samples"],
                      "trained_at": model["trained_at"], "buffer": len(history), "need": MIN_TRAIN},
            "sim": sim["kind"], "host": os.uname().nodename if hasattr(os, "uname") else "host",
        })


@app.route("/api/simulate/<kind>", methods=["POST", "GET"])
def simulate(kind):
    if kind not in ("cpu_spike", "memory_leak", "disk_fill", "net_flood"):
        return jsonify({"ok": False}), 400
    dur = {"cpu_spike": 40, "memory_leak": 120, "disk_fill": 120, "net_flood": 40}[kind]
    sim.update(kind=kind, start=time.time(), until=time.time() + dur)
    push_alert("info", f"Demo incident injected: {kind.replace('_', ' ')}", "sim")
    return jsonify({"ok": True, "kind": kind, "duration": dur})


@app.route("/api/reset", methods=["POST", "GET"])
def reset():
    sim.update(kind=None, start=0, until=0)
    return jsonify({"ok": True})


@app.route("/metrics")
def prom():
    with lock:
        c = history[-1] if history else None
    if not c:
        return Response("", mimetype="text/plain")
    lines = [f"devops_ai_cpu_percent {c['cpu']}", f"devops_ai_memory_percent {c['mem']}",
             f"devops_ai_disk_percent {c['disk']}", f"devops_ai_net_kbps {c['net']}",
             f"devops_ai_anomaly {int(c['anom'])}"]
    for s in services.values():
        lines.append(f'devops_ai_service_up{{service="{s["name"]}"}} {1 if s["status"] in ("up", "degraded") else 0}')
        lines.append(f'devops_ai_service_latency_ms{{service="{s["name"]}"}} {s["latency"] or 0}')
    return Response("\n".join(lines) + "\n", mimetype="text/plain")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ------------------------------------------------- application / service monitoring
# Monitors HTTP services (health check, latency, uptime). Three demo microservices are
# built in (so the demo works offline); any real URL can be added from the dashboard.
import http.client
import socket
import ssl
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse

PORT = int(os.environ.get("PORT", 5050))
services = {}
svc_mode = {}        # simulated state of the built-in demo microservices: normal | slow | down
SVC_RECO = {
    "down": ("Service unreachable (health check failing)",
             "Check container/pod status (docker ps / kubectl get pods), read the service logs, "
             "restart the service, and roll back the last deployment if the failure started after a release."),
    "degraded": ("High response latency (service degraded)",
                 "Check DB connection pool and slow queries, look at upstream dependencies, "
                 "scale out replicas / enable caching, and review the most recent deployment."),
}


def add_service(name, url, key=None, thr=1000):
    services[name] = {"name": name, "url": url, "key": key, "hist": collections.deque(maxlen=120),
                      "fails": 0, "status": "unknown", "latency": None, "code": None, "uptime": 100.0,
                      "thr": thr, "checks": 0, "last_ts": 0.0, "last": None, "diag": None, "slow_streak": 0}


for _n, _k in (("Auth Service", "auth"), ("Payment Service", "payment"), ("User API", "users")):
    add_service(_n, f"http://127.0.0.1:{PORT}/svc/{_k}", _k)
    svc_mode[_k] = "normal"


@app.route("/svc/<key>")
def demo_service(key):
    mode = svc_mode.get(key)
    if mode is None:
        return jsonify({"error": "unknown service"}), 404
    if mode == "down":
        return jsonify({"service": key, "status": "error"}), 503
    time.sleep(random.uniform(1.6, 2.2) if mode == "slow" else random.uniform(0.01, 0.05))
    return jsonify({"service": key, "status": "ok"})


REDIRECTS = (301, 302, 303, 307, 308)
UA = "Mozilla/5.0 (compatible; AI-DevOps-Monitor/1.0)"


def probe(url, timeout=6.0):
    """One health check with a per-phase timing breakdown:
    DNS -> TCP connect -> TLS handshake -> server wait (TTFB) -> content download."""
    t_start = time.time()
    r = {"ok": False, "code": 0, "error": None, "total": 0.0, "dns": 0.0, "connect": 0.0, "tls": 0.0,
         "ttfb": 0.0, "download": 0.0, "redirects": 0, "redirect_ms": 0.0, "size": 0,
         "compressed": None, "cert_days": None, "server": None}
    try:
        for _ in range(5):
            u = urlparse(url)
            https = u.scheme == "https"
            host, port = u.hostname, u.port or (443 if https else 80)
            path = (u.path or "/") + ("?" + u.query if u.query else "")
            t0 = time.time()
            addr = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][4]
            t1 = time.time()
            sock = socket.create_connection(addr[:2], timeout=timeout)
            t2 = time.time()
            if https:
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
                cert = sock.getpeercert()
                if cert and "notAfter" in cert:
                    r["cert_days"] = int((ssl.cert_time_to_seconds(cert["notAfter"]) - time.time()) // 86400)
            t3 = time.time()
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
            conn.sock = sock
            hdr_host = host if port in (80, 443) else f"{host}:{port}"
            t4 = time.time()
            conn.request("GET", path, headers={"Host": hdr_host, "User-Agent": UA, "Accept": "*/*",
                                               "Accept-Encoding": "gzip", "Connection": "close"})
            resp = conn.getresponse()
            t5 = time.time()
            body = resp.read(2_000_000)
            t6 = time.time()
            r["code"] = resp.status
            loc = resp.getheader("Location")
            conn.close()
            if resp.status in REDIRECTS and loc:
                r["redirects"] += 1
                r["redirect_ms"] += (t6 - t0) * 1000
                url = urljoin(url, loc)
                continue
            r.update(dns=(t1 - t0) * 1000, connect=(t2 - t1) * 1000, tls=(t3 - t2) * 1000,
                     ttfb=(t5 - t4) * 1000, download=(t6 - t5) * 1000, size=len(body),
                     compressed=(resp.getheader("Content-Encoding") or "").lower() in ("gzip", "br", "deflate"),
                     server=resp.getheader("Server"))
            r["ok"] = resp.status < 400
            if not r["ok"]:
                r["error"] = f"http_{resp.status}"
            break
    except socket.gaierror:
        r["error"] = "dns"
    except (socket.timeout, TimeoutError):
        r["error"] = "timeout"
    except ConnectionRefusedError:
        r["error"] = "refused"
    except (ssl.SSLError, ssl.CertificateError):
        r["error"] = "ssl"
    except Exception:
        r["error"] = "error"
    r["total"] = (time.time() - t_start) * 1000
    return r


BUDGET = {"dns": 100, "connect": 150, "tls": 250, "ttfb": 500, "download": 500, "redirect": 300}
LABEL = {"dns": "DNS lookup", "connect": "TCP connect", "tls": "TLS handshake",
         "ttfb": "Server wait (time to first byte)", "download": "Content download", "redirect": "Redirects"}
SLOW_WHY = {
    "dns": ("Slow DNS resolution",
            "Use a faster DNS provider (Cloudflare / Route53), raise the DNS TTL, add DNS caching."),
    "connect": ("High network latency / slow TCP connection (server far away or congested route)",
                "Serve through a CDN or a region closer to users, check routing and packet loss, enable keep-alive and HTTP/2."),
    "tls": ("Slow HTTPS/TLS handshake",
            "Enable TLS session resumption and OCSP stapling, use TLS 1.3 + HTTP/2, shorten the certificate chain."),
    "ttfb": ("Slow server response - backend processing time is high",
             "Optimise database queries and add indexes, enable server-side / page caching (Redis, Varnish), "
             "scale the web server up or out, check CPU/memory and PHP/Node worker limits on the server."),
    "download": ("Heavy page or slow content transfer",
                 "Enable gzip/brotli compression, compress and lazy-load images, minify JS/CSS, move static files to a CDN."),
    "redirect": ("Too many or slow redirects (e.g. http -> https -> www)",
                 "Redirect once, directly to the final canonical https URL; update links and sitemap."),
}


def diagnose(r, status):
    """Explain WHY a service is down/slow and WHAT to do (reason, fix, evidence)."""
    err, code = r["error"] or "", r["code"] or 0
    if status == "down":
        if err == "dns":
            out = ("DNS lookup failed - the domain name could not be resolved",
                   "Check the domain's DNS records (A/CNAME) and domain expiry, verify the DNS provider, test with 8.8.8.8 / 1.1.1.1.")
        elif err == "timeout":
            out = ("Connection timed out - the server is not responding",
                   "Check that the server/host is running, review firewall / security-group rules, look for an overloaded server, check the hosting provider's status page.")
        elif err == "refused":
            out = ("Connection refused - nothing is listening on the port",
                   "Start or restart the web server (nginx / apache / node), verify the listening port and firewall rules.")
        elif err == "ssl":
            out = ("SSL/TLS certificate problem (expired, invalid or untrusted)",
                   "Renew the certificate (certbot / provider), check the full chain and that the domain name matches, verify server date/time.")
        elif code >= 500:
            out = (f"Server error HTTP {code} - the application or its backend is failing",
                   "Read the application/server error logs, roll back the latest deployment, check database and upstream health, restart the app service.")
        elif code in (401, 403):
            out = (f"Access denied (HTTP {code})",
                   "Check firewall / WAF rules, authentication and IP allow-lists - the monitor's IP may be blocked.")
        elif code == 404:
            out = ("Page not found (HTTP 404)", "Verify the URL path, check routing and deployment, restore the missing page.")
        elif code == 429:
            out = ("Rate limited (HTTP 429)", "Reduce request frequency or allow-list the monitoring IP.")
        elif code >= 400:
            out = (f"Client error HTTP {code}", "Verify the URL, request headers and access rules.")
        else:
            out = ("Service unreachable (health check failing)",
                   "Check container/pod status (docker ps / kubectl get pods), read the logs, restart the service, roll back the last release.")
        return out[0], out[1], (f"HTTP {code}" if code else f"error: {err or 'no response'}")
    ph = {"dns": r["dns"], "connect": r["connect"], "tls": r["tls"], "ttfb": r["ttfb"],
          "download": r["download"], "redirect": r["redirect_ms"]}
    top = max(ph, key=lambda k: ph[k] / BUDGET[k])
    why, fix = SLOW_WHY[top]
    share = 100 * ph[top] / max(r["total"], 1)
    if top == "download" and r["compressed"] is False:
        fix += " (Compression is currently OFF on this site.)"
    reason = f"{why} - {LABEL[top]} took {ph[top]:.0f} ms ({share:.0f}% of the {r['total']:.0f} ms response)"
    ev = (f"DNS {r['dns']:.0f} ms | Connect {r['connect']:.0f} | TLS {r['tls']:.0f} | Server wait {r['ttfb']:.0f} | "
          f"Download {r['download']:.0f} | Redirects {r['redirect_ms']:.0f} | {r['size'] / 1024:.0f} KB | "
          f"compression {'on' if r['compressed'] else 'OFF'}")
    return reason, fix, ev


def svc_eval(s, r):
    """Update one service from a probe result: uptime, latency anomaly, status, diagnosis, alerts."""
    ok, ms = r["ok"], r["total"]
    h = s["hist"]
    h.append({"ok": ok, "ms": round(ms, 1)})
    s["checks"] += 1
    s["last_ts"], s["last"] = time.time(), r
    s["code"], s["latency"] = r["code"], (round(ms, 1) if ok else None)
    s["fails"] = 0 if ok else s["fails"] + 1
    last = list(h)[-60:]
    s["uptime"] = round(100.0 * sum(1 for x in last if x["ok"]) / len(last), 1)
    base = [x["ms"] for x in list(h)[:-5] if x["ok"]]
    slow = False
    if ok:
        slow = ms > s["thr"]
        if len(base) >= 12:        # learned baseline: robust z-score on this service's own history
            med = float(np.median(base))
            sd = max(float(np.median(np.abs(np.array(base) - med))) * 1.4826, 30.0)
            slow = slow or ((ms - med) / sd > 4 and ms > 150)
    s["slow_streak"] = s["slow_streak"] + 1 if (ok and slow) else 0     # 2 slow checks in a row -> no flapping
    prev = s["status"]
    if s["fails"] >= 2:
        s["status"] = "down"
    elif ok:
        s["status"] = "degraded" if s["slow_streak"] >= 2 else "up"
    if s["status"] in ("down", "degraded"):
        reason, fix, ev = diagnose(r, s["status"])
        s["diag"] = {"reason": reason, "fix": fix, "evidence": ev}
    else:
        s["diag"] = None
    name = s["name"]
    if s["status"] != prev and s["diag"]:
        push_alert("critical" if s["status"] == "down" else "warning",
                   f"Service '{name}' is {s['status'].upper()}: {s['diag']['reason']}",
                   f"svc-{name}-{s['status']}", s["diag"]["fix"])
    elif prev in ("down", "degraded") and s["status"] == "up":
        push_alert("info", f"Service '{name}' recovered and is healthy again", f"svc-{name}-up")
    cd = r["cert_days"]
    if cd is not None and cd < 15:
        push_alert("warning", f"SSL certificate of '{name}' expires in {cd} days", f"ssl-{name}",
                   "Renew the certificate now (certbot renew / hosting dashboard) and enable auto-renewal.")


pool = ThreadPoolExecutor(max_workers=8)


def svc_loop():
    time.sleep(3)
    while True:
        svcs = list(services.values())
        results = list(pool.map(lambda x: probe(x["url"]), svcs))
        with lock:
            for s, r in zip(svcs, results):
                svc_eval(s, apply_overlay(s["name"], r))
        time.sleep(3)


def svc_list():
    now = time.time()
    out = []
    for s in services.values():
        r = s["last"] or {}
        out.append({
            "name": s["name"], "url": s["url"], "key": s["key"], "status": s["status"],
            "latency": s["latency"], "uptime": s["uptime"], "code": s["code"], "checks": s["checks"],
            "age": None if not s["last_ts"] else round(now - s["last_ts"]),
            "series": [x["ms"] if x["ok"] else 0 for x in list(s["hist"])[-40:]],
            "mode": svc_mode.get(s["key"]) if s["key"] else None,
            "sim": (svc_mode.get(s["key"]) if s["key"] else overlay.get(s["name"])) not in (None, "normal")
                   and (svc_mode.get(s["key"]) if s["key"] else overlay.get(s["name"])),
            "phases": {"dns": round(r.get("dns", 0)), "connect": round(r.get("connect", 0)),
                       "tls": round(r.get("tls", 0)), "server": round(r.get("ttfb", 0)),
                       "download": round(r.get("download", 0)), "redirect": round(r.get("redirect_ms", 0))},
            "cert_days": r.get("cert_days"), "size_kb": round(r.get("size", 0) / 1024),
            "compressed": r.get("compressed"), "redirects": r.get("redirects", 0),
            "thr": s["thr"], "diag": s["diag"]})
    return out


def svc_insight():
    for s in services.values():
        if s["status"] in ("down", "degraded") and s["diag"]:
            d = s["diag"]
            return {"type": "service", "metric": s["name"], "cause": f"{s['name']}: {d['reason']}",
                    "action": d["fix"], "detail": d["evidence"] or f"Service '{s['name']}' is {s['status'].upper()}"}
    return None


@app.route("/api/svc_sim/<key>/<mode>", methods=["POST", "GET"])
def svc_sim(key, mode):
    if mode not in ("normal", "slow", "down") or (key not in svc_mode and key not in services):
        return jsonify({"ok": False}), 400
    set_sim(key, mode)
    if mode != "normal":
        push_alert("info", f"Demo incident injected on '{key}': simulated {mode}", "sim-svc-" + key)
    return jsonify({"ok": True})


@app.route("/api/services", methods=["POST"])
def svc_add():
    from flask import request
    d = request.get_json(silent=True) or {}
    name, url = str(d.get("name", "")).strip()[:30], str(d.get("url", "")).strip()
    if not name or not url.startswith(("http://", "https://")):
        return jsonify({"ok": False, "error": "need name and http(s) url"}), 400
    try:
        thr = max(200, min(20000, int(d.get("thr") or 1000)))
    except (TypeError, ValueError):
        thr = 1000
    with lock:
        add_service(name, url, None, thr)
    return jsonify({"ok": True})


# ------------------------------------------------------------------ auto demo
demo = {"running": False, "step": 0, "total": 0, "title": "", "say": "", "left": 0, "target": ""}
_demo_stop = threading.Event()
overlay = {}                 # simulated failures layered on top of REAL monitored URLs (demo only)
demo_target = {"name": None}


def set_sim(ident, mode):
    """Simulate a failure on a built-in demo service (by key) or on ANY monitored URL (by name)."""
    if ident in svc_mode:
        svc_mode[ident] = mode
    elif ident in services:
        if mode == "normal":
            overlay.pop(ident, None)
        else:
            overlay[ident] = mode


def apply_overlay(name, r):
    """Turn a real probe result into a simulated incident (only while a demo failure is active)."""
    mode = overlay.get(name)
    if not mode:
        return r
    r = dict(r)
    if mode == "down":
        r.update(ok=False, code=503, error="http_503", ttfb=0.0)
    elif mode == "slow" and r["ok"]:
        r.update(ttfb=r["ttfb"] + 1800, total=r["total"] + 1800)
    return r


def _inject(kind, dur):
    sim.update(kind=kind, start=time.time(), until=time.time() + dur)


def _reset_all():
    sim.update(kind=None, start=0, until=0)
    overlay.clear()
    for k in svc_mode:
        svc_mode[k] = "normal"


def _dn():
    return demo_target["name"] or "payment"


def _sl():
    return demo_target["name"] or "auth"


# (seconds, title, what-to-say, action). {down}/{slow} are replaced by the service name used.
DEMO_STEPS = [
    (10, "Normal state - the AI has learned the baseline",
     "Everything is green. The system has learned this machine's normal behaviour (see 'Model trained' on the health card).",
     lambda: _reset_all()),
    (22, "Incident 1: {down} goes DOWN",
     "The service stops responding. Watch the status turn DOWN, the health score drop, and the system explain the REASON and the SOLUTION on its own.",
     lambda: set_sim(_dn(), "down")),
    (8, "Recovery: {down} is healed",
     "Once the service responds again, the system raises a 'recovered' alert.",
     lambda: set_sim(_dn(), "normal")),
    (28, "Incident 2: {slow} becomes SLOW (degraded)",
     "The service is up but slow. Fixed-threshold tools miss this; our AI compares latency with the learned baseline and identifies the slow phase.",
     lambda: set_sim(_sl(), "slow")),
    (8, "Recovery: {slow} is healed",
     "Latency is back to normal and the status returns to UP.",
     lambda: set_sim(_sl(), "normal")),
    (70, "Incident 3: Memory leak - FAILURE PREDICTION",
     "Memory has not reached 90% yet, but the pink dotted forecast and the 'hits 90% in X min' label predict it - before any threshold alert would fire.",
     lambda: _inject("memory_leak", 120)),
    (8, "Recovery: memory back to normal",
     "The leak is fixed and memory returns to normal.",
     lambda: sim.update(kind=None, start=0, until=0)),
    (36, "Incident 4: CPU spike - ANOMALY DETECTION",
     "A sudden CPU spike. The red dots are anomalies found by the Isolation Forest, and the AI Insights box shows the recommended fix.",
     lambda: _inject("cpu_spike", 40)),
    (28, "Incident 5: Traffic flood - possible DDoS",
     "Network traffic suddenly jumps. The system flags it as abnormal and suggests rate limiting.",
     lambda: _inject("net_flood", 40)),
    (12, "Demo complete - download the report",
     "Click 'Download PDF Report' to get every alert with its reason and solution in one document.",
     lambda: _reset_all()),
]


def _run_demo():
    _demo_stop.clear()
    demo.update(running=True, total=len(DEMO_STEPS), target=demo_target["name"] or "")
    names = {"down": demo_target["name"] or "Payment Service", "slow": demo_target["name"] or "Auth Service"}
    for i, (dur, title, say, action) in enumerate(DEMO_STEPS):
        if _demo_stop.is_set():
            break
        suffix = " (simulated on a real URL)" if demo_target["name"] and "{" in title else ""
        demo.update(step=i + 1, title=title.format(**names) + suffix, say=say, left=dur)
        try:
            action()
        except Exception as e:
            print("demo step error", e)
        for left in range(dur, 0, -1):
            demo["left"] = left
            if _demo_stop.wait(1):
                break
    _reset_all()
    demo.update(running=False, step=0, title="", say="", left=0, target="")


@app.route("/api/demo/<cmd>", methods=["POST", "GET"])
def demo_ctl(cmd):
    from flask import request
    if cmd == "start" and not demo["running"]:
        t = request.args.get("target", "")
        demo_target["name"] = t if t in services else None
        threading.Thread(target=_run_demo, daemon=True).start()
    elif cmd == "stop":
        _demo_stop.set()
    return jsonify({"ok": True, "demo": dict(demo)})


# ------------------------------------------------------------------ PDF report
def _clean(t):
    t = str(t if t is not None else "").replace("σ", " sigma").replace("→", "->").replace("–", "-")
    return t.encode("cp1252", "replace").decode("cp1252")


def build_report():
    from io import BytesIO
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    with lock:
        cur = dict(history[-1]) if history else None
        preds = {k: predict(k) for k in ("cpu", "mem", "disk")}
        svcs = svc_list()
        log = list(alert_log)
        score = health_score(cur) if cur else 100
        mdl = dict(model)
    ss = getSampleStyleSheet()
    P = lambda t, st=None: Paragraph(escape(_clean(t)), st or ss["BodyText"])
    small = ParagraphStyle("s", parent=ss["BodyText"], fontSize=8, leading=10)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#1e3a8a"))
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title="AI DevOps Monitoring Report")
    W = A4[0] - 28 * mm
    st = [P("AI-Based DevOps Monitoring System - Incident Report", ss["Title"]),
          P(f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}   |   Host: "
            f"{os.uname().nodename if hasattr(os, 'uname') else 'host'}   |   Uptime of monitor: "
            f"{int((time.time() - started_at) // 60)} min   |   Health score: {score}/100", small),
          Spacer(1, 6)]

    def table(rows, widths, head=True):
        t = Table(rows, colWidths=widths, repeatRows=1 if head else 0)
        sty = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c7cde0")),
               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 8)]
        if head:
            sty += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        t.setStyle(TableStyle(sty))
        return t

    st.append(P("1. System snapshot", h2))
    if cur:
        rows = [["CPU %", "Memory %", "Disk %", "Network KB/s", "Disk I/O MB/s", "Anomaly"],
                [cur["cpu"], cur["mem"], cur["disk"], cur["net"], cur["disk_io"], "YES" if cur["anom"] else "no"]]
        st.append(table(rows, [W / 6] * 6))
    st.append(Spacer(1, 4))
    st.append(P(f"AI model: Isolation Forest + robust z-score (anomaly detection), linear-regression trend forecast "
                f"(prediction). Trained on {mdl['samples']} samples at {mdl['trained_at'] or '-'}.", small))

    st.append(P("2. Failure prediction", h2))
    rows = [["Metric", "Trend per min", "Time to 90%", "Confidence", "Status"]]
    for k, p in preds.items():
        rows.append([k.upper(), "-" if not p else f"{p['slope_per_min']:+}%",
                     "-" if not p or p["eta_min"] is None else f"{p['eta_min']} min",
                     "-" if not p else f"{int(p['confidence'] * 100)}%", "-" if not p else p["status"].upper()])
    st.append(table(rows, [W / 5] * 5))

    st.append(P("3. Monitored services", h2))
    rows = [["Service", "Status", "Latency", "Uptime", "Checks", "Reason / Solution"]]
    for s in svcs:
        d = s["diag"]
        txt = P(f"Reason: {d['reason']}  Solution: {d['fix']}", small) if d else P("Healthy", small)
        rows.append([P(f"{s['name']} ({s['url'].replace('127.0.0.1', 'localhost')})", small), s["status"].upper(),
                     "-" if s["latency"] is None else f"{s['latency']:.0f} ms", f"{s['uptime']}%", s["checks"], txt])
    st.append(table(rows, [W * .24, W * .10, W * .10, W * .09, W * .08, W * .39]))

    st.append(P(f"4. Alerts and recommended solutions ({len(log)} events)", h2))
    rows = [["Time", "Level", "Alert", "Recommended solution"]]
    for a in reversed(log):
        rows.append([a["time"], a["level"].upper(), P(a["msg"], small), P(a["action"] or "-", small)])
    if len(rows) == 1:
        rows.append(["-", "-", P("No alerts recorded yet.", small), "-"])
    t = table(rows, [W * .09, W * .09, W * .42, W * .40])
    for i, a in enumerate(reversed(log), start=1):
        c = {"critical": "#fee2e2", "warning": "#fef3c7", "info": "#e0f2fe"}.get(a["level"])
        if c:
            t.setStyle(TableStyle([("BACKGROUND", (0, i), (1, i), colors.HexColor(c))]))
    st.append(t)
    doc.build(st)
    return buf.getvalue()


def simple_pdf(lines, per_page=64, width=104):
    """Dependency-free fallback: a plain-text PDF (Courier) so the report always downloads."""
    import textwrap
    wrapped = []
    for ln in lines:
        wrapped += textwrap.wrap(_clean(ln), width) or [""]
    pages = [wrapped[i:i + per_page] for i in range(0, len(wrapped), per_page)] or [[]]
    esc = lambda s: s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    objs = {1: "<< /Type /Catalog /Pages 2 0 R >>", 3: "<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"}
    kids, n = [], 3
    for pg in pages:
        pid, cid = n + 1, n + 2
        n += 2
        kids.append(f"{pid} 0 R")
        stream = "BT /F1 8 Tf 10 TL 36 806 Td\n" + "\n".join(f"({esc(l)}) '" for l in pg) + "\nET"
        objs[pid] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                     f"/Resources << /Font << /F1 3 0 R >> >> /Contents {cid} 0 R >>")
        objs[cid] = f"<< /Length {len(stream.encode('latin-1', 'replace'))} >>\nstream\n{stream}\nendstream"
    objs[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>"
    out, offs = bytearray(b"%PDF-1.4\n"), {}
    for i in sorted(objs):
        offs[i] = len(out)
        out += f"{i} 0 obj\n{objs[i]}\nendobj\n".encode("latin-1", "replace")
    xref = len(out)
    out += f"xref\n0 {n + 1}\n0000000000 65535 f \n".encode()
    for i in range(1, n + 1):
        out += f"{offs[i]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {n + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def report_lines():
    with lock:
        cur = dict(history[-1]) if history else None
        preds = {k: predict(k) for k in ("cpu", "mem", "disk")}
        svcs = svc_list()
        log = list(alert_log)
        score = health_score(cur) if cur else 100
    L = ["AI-BASED DEVOPS MONITORING SYSTEM - INCIDENT REPORT",
         f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}   Health score: {score}/100", "",
         "1. SYSTEM SNAPSHOT"]
    if cur:
        L.append(f"CPU {cur['cpu']}%  Memory {cur['mem']}%  Disk {cur['disk']}%  Network {cur['net']} KB/s  "
                 f"Anomaly: {'YES' if cur['anom'] else 'no'}")
    L += ["", "2. FAILURE PREDICTION"]
    for k, p in preds.items():
        L.append(f"{k.upper()}: " + ("learning" if not p else
                 f"trend {p['slope_per_min']:+}%/min, time to 90%: "
                 f"{'-' if p['eta_min'] is None else str(p['eta_min']) + ' min'}, status {p['status'].upper()}"))
    L += ["", "3. MONITORED SERVICES"]
    for s in svcs:
        L.append(f"- {s['name']} ({s['url']}): {s['status'].upper()}, latency "
                 f"{'-' if s['latency'] is None else str(round(s['latency'])) + ' ms'}, uptime {s['uptime']}%")
        if s["diag"]:
            L += [f"    Reason:   {s['diag']['reason']}", f"    Solution: {s['diag']['fix']}"]
    L += ["", f"4. ALERTS AND RECOMMENDED SOLUTIONS ({len(log)} events)"]
    for a in reversed(log):
        L.append(f"[{a['time']}] {a['level'].upper()}: {a['msg']}")
        if a["action"]:
            L.append(f"    Solution: {a['action']}")
    if not log:
        L.append("No alerts recorded yet.")
    return L


@app.route("/api/report.pdf")
def report_pdf():
    try:
        data = build_report()
    except Exception as e:           # e.g. reportlab not installed -> plain-text PDF fallback
        print("report fallback:", e)
        data = simple_pdf(report_lines())
    name = datetime.now().strftime("devops-report-%Y%m%d-%H%M%S.pdf")
    return Response(data, mimetype="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


threading.Thread(target=collector, daemon=True).start()
threading.Thread(target=svc_loop, daemon=True).start()

if __name__ == "__main__":
    print(f"Dashboard -> http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)