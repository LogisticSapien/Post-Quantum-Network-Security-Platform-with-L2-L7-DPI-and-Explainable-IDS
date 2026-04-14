"""
Web Dashboard — Premium Edition
=================================
Flask-based web UI with:
  • Glassmorphism dark theme with gradient accents
  • Real-time traffic timeline (animated line chart)
  • Threat heatmap timeline with severity colors
  • Protocol distribution doughnut with animations
  • Top talkers horizontal bar chart
  • Active flows table with state badges
  • PQC status card with quantum shield icon
  • Alert feed with severity badges + explainability
  • Performance metrics with animated counters
  • Deep TLS analysis widget
  • Auto-refresh every 2 seconds via fetch API

Uses Chart.js 4 + Google Fonts (Inter).
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from flask import Flask, jsonify, render_template_string
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


class DashboardDataStore:
    """Thread-safe data store for dashboard metrics."""

    def __init__(self, max_points: int = 300):
        self._lock = threading.Lock()
        self.max_points = max_points
        self.traffic_timeline: deque = deque(maxlen=max_points)
        self.threat_timeline: deque = deque(maxlen=max_points)
        self.protocol_counts: Dict[str, int] = defaultdict(int)
        self.top_talkers: Dict[str, int] = defaultdict(int)
        self.alerts: deque = deque(maxlen=200)
        self.active_flows: List[dict] = []
        self.tls_analyses: List[dict] = []
        self.pqc_status = {
            "enabled": True, "kem_algorithm": "Kyber-512",
            "hash_chain_length": 0, "encrypted_entries": 0,
            "quantum_threats_found": 0,
        }
        self.performance = {
            "packets_per_sec": 0, "bytes_per_sec": 0,
            "total_packets": 0, "uptime": 0, "threats_total": 0,
            "avg_latency_us": 0,
        }
        self._last_second = int(time.time())
        self._pkt_count_this_sec = 0

    def record_packet(self, protocol: str, src_ip: str, size: int):
        with self._lock:
            self.protocol_counts[protocol] += 1
            self.top_talkers[src_ip] += size
            self.performance["total_packets"] += 1
            now = int(time.time())
            if now != self._last_second:
                self.traffic_timeline.append({"time": self._last_second, "count": self._pkt_count_this_sec})
                self._pkt_count_this_sec = 0
                self._last_second = now
            self._pkt_count_this_sec += 1

    def record_alert(self, alert_dict: dict):
        with self._lock:
            self.alerts.appendleft(alert_dict)
            self.threat_timeline.append({
                "time": int(time.time()), "severity": alert_dict.get("severity", 1),
                "category": alert_dict.get("category", "UNKNOWN"),
            })
            self.performance["threats_total"] = len(self.alerts)

    def update_flows(self, flows: List[dict]):
        with self._lock: self.active_flows = flows[:50]

    def update_pqc(self, status: dict):
        with self._lock: self.pqc_status.update(status)

    def update_performance(self, perf: dict):
        with self._lock: self.performance.update(perf)

    def update_tls(self, analyses: List[dict]):
        with self._lock: self.tls_analyses = analyses[:10]

    def get_state(self) -> dict:
        with self._lock:
            top = sorted(self.top_talkers.items(), key=lambda x: -x[1])[:10]
            return {
                "traffic": list(self.traffic_timeline),
                "threats": list(self.threat_timeline),
                "protocols": dict(self.protocol_counts),
                "top_talkers": [{"ip": ip, "bytes": b} for ip, b in top],
                "alerts": list(self.alerts)[:50],
                "flows": self.active_flows[:20],
                "tls": self.tls_analyses[:8],
                "pqc": dict(self.pqc_status),
                "performance": dict(self.performance),
            }


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantum Sniffer Dashboard</title>
<meta name="description" content="Real-time quantum-resistant network analysis dashboard">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #06080f; --bg2: #0c1021; --card: rgba(15,23,42,0.85);
  --border: rgba(99,102,241,0.12); --border-hover: rgba(99,102,241,0.3);
  --text: #e2e8f0; --text2: #94a3b8; --text3: #64748b;
  --accent: #818cf8; --accent2: #6366f1; --accent-glow: rgba(99,102,241,0.15);
  --safe: #34d399; --safe-bg: rgba(52,211,153,0.1);
  --warn: #fbbf24; --warn-bg: rgba(251,191,36,0.1);
  --danger: #f87171; --danger-bg: rgba(248,113,113,0.1);
  --critical: #ef4444; --critical-bg: rgba(239,68,68,0.12);
  --radius: 12px; --glow: 0 0 30px rgba(99,102,241,0.08);
}

* { margin:0; padding:0; box-sizing:border-box; }
html { font-size: 14px; }
body {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  background: var(--bg); color: var(--text);
  min-height: 100vh;
  background-image:
    radial-gradient(ellipse at 20% 50%, rgba(99,102,241,0.06) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 20%, rgba(139,92,246,0.04) 0%, transparent 50%);
}

/* ─── Header ─── */
.header {
  background: linear-gradient(135deg, rgba(30,27,75,0.9), rgba(49,46,129,0.7));
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px; display: flex; align-items: center;
  justify-content: space-between; position: sticky; top:0; z-index:100;
}
.header-left { display: flex; align-items: center; gap: 12px; }
.logo {
  width: 36px; height: 36px; border-radius: 10px;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; box-shadow: 0 0 20px rgba(99,102,241,0.3);
}
.header h1 { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; }
.header h1 span { color: var(--accent); }
.header-stats {
  display: flex; gap: 24px; font-size: 0.78rem; color: var(--text2);
}
.header-stats .stat-val { color: var(--accent); font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.live-dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: #34d399; animation: pulse 2s infinite; margin-right: 6px;
}
@keyframes pulse {
  0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(52,211,153,0.4); }
  50% { opacity:0.6; box-shadow: 0 0 0 6px rgba(52,211,153,0); }
}

/* ─── Grid ─── */
.grid {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 12px; padding: 16px 20px; max-width: 1600px; margin: 0 auto;
}

/* ─── Cards ─── */
.card {
  background: var(--card); backdrop-filter: blur(12px);
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: 16px; box-shadow: var(--glow);
  transition: border-color 0.3s, box-shadow 0.3s;
}
.card:hover { border-color: var(--border-hover); box-shadow: 0 0 40px rgba(99,102,241,0.1); }
.card-title {
  font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--text3); margin-bottom: 12px;
  display: flex; align-items: center; gap: 6px;
}
.card-title .icon { font-size: 0.85rem; }

.c1 { grid-column: span 1; } .c2 { grid-column: span 2; }
.c3 { grid-column: span 3; } .c4 { grid-column: span 4; }
.c5 { grid-column: span 5; } .c6 { grid-column: span 6; }
.c8 { grid-column: span 8; } .c12 { grid-column: span 12; }

/* ─── Metric Boxes ─── */
.metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }
.metric-box { text-align: center; }
.metric-val {
  font-size: 1.7rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;
  background: linear-gradient(135deg, var(--accent), #a78bfa);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; line-height: 1.1;
}
.metric-label { font-size: 0.65rem; color: var(--text3); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }

/* ─── Charts ─── */
canvas { max-height: 200px; }

/* ─── Alerts ─── */
.alert-feed { max-height: 240px; overflow-y: auto; scrollbar-width: thin; scrollbar-color: var(--border) transparent; }
.alert-feed::-webkit-scrollbar { width: 4px; }
.alert-feed::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
.alert-item {
  padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.04);
  display: flex; gap: 8px; align-items: flex-start; font-size: 0.78rem;
  transition: background 0.2s;
}
.alert-item:hover { background: rgba(99,102,241,0.04); }
.badge {
  padding: 2px 8px; border-radius: 6px; font-size: 0.6rem;
  font-weight: 700; letter-spacing: 0.03em; white-space: nowrap; flex-shrink: 0;
}
.badge-critical { background: var(--critical-bg); color: var(--critical); border: 1px solid rgba(239,68,68,0.2); }
.badge-high { background: var(--danger-bg); color: #fb923c; border: 1px solid rgba(251,146,60,0.2); }
.badge-medium { background: var(--warn-bg); color: var(--warn); border: 1px solid rgba(251,191,36,0.2); }
.badge-low { background: rgba(56,189,248,0.1); color: #38bdf8; border: 1px solid rgba(56,189,248,0.2); }
.badge-info { background: rgba(148,163,184,0.1); color: var(--text2); border: 1px solid rgba(148,163,184,0.15); }
.alert-text { color: var(--text2); line-height: 1.35; }
.alert-time { color: var(--text3); font-size: 0.65rem; font-family: 'JetBrains Mono', monospace; }

/* ─── Tables ─── */
table { width: 100%; border-collapse: collapse; }
th { text-align: left; color: var(--text3); padding: 6px 8px; font-size: 0.68rem;
  text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;
  border-bottom: 1px solid var(--border); }
td { padding: 6px 8px; font-size: 0.78rem; border-bottom: 1px solid rgba(255,255,255,0.03);
  font-family: 'JetBrains Mono', monospace; color: var(--text2); }
tr:hover td { background: rgba(99,102,241,0.03); }

.state-badge {
  padding: 2px 6px; border-radius: 4px; font-size: 0.6rem; font-weight: 600;
}
.state-est { background: rgba(52,211,153,0.12); color: var(--safe); }
.state-syn { background: rgba(251,191,36,0.12); color: var(--warn); }
.state-fin { background: rgba(248,113,113,0.12); color: var(--danger); }
.state-new { background: rgba(99,102,241,0.12); color: var(--accent); }

/* ─── PQC Card ─── */
.pqc-card {
  background: linear-gradient(135deg, rgba(6,78,59,0.6), rgba(6,95,70,0.4));
  border-color: rgba(52,211,153,0.15);
}
.pqc-card:hover { border-color: rgba(52,211,153,0.3); }
.pqc-item {
  display: flex; justify-content: space-between; padding: 5px 0;
  font-size: 0.78rem; border-bottom: 1px solid rgba(52,211,153,0.08);
}
.pqc-item:last-child { border-bottom: none; }
.pqc-item .label { color: rgba(167,243,208,0.7); }
.pqc-item .value { color: #6ee7b7; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.pqc-shield { font-size: 2rem; text-align: center; margin-bottom: 8px; }

/* ─── TLS Card ─── */
.tls-item {
  display: flex; align-items: center; gap: 8px; padding: 6px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.75rem;
}
.tls-item:last-child { border-bottom: none; }
.grade {
  width: 28px; height: 28px; border-radius: 6px; display: flex;
  align-items: center; justify-content: center; font-weight: 800;
  font-size: 0.75rem; flex-shrink: 0;
}
.grade-a { background: rgba(52,211,153,0.15); color: var(--safe); }
.grade-b { background: rgba(56,189,248,0.15); color: #38bdf8; }
.grade-c { background: rgba(251,191,36,0.15); color: var(--warn); }
.grade-d { background: rgba(248,113,113,0.15); color: var(--danger); }
.grade-f { background: rgba(239,68,68,0.2); color: var(--critical); }
.tls-suite { color: var(--text2); font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; }
.tls-verdict { font-size: 0.6rem; font-weight: 600; padding: 1px 5px; border-radius: 3px; }
.verdict-safe { background: var(--safe-bg); color: var(--safe); }
.verdict-risk { background: var(--warn-bg); color: var(--warn); }
.verdict-crit { background: var(--critical-bg); color: var(--critical); }

/* ─── Responsive ─── */
@media (max-width: 1200px) {
  .c8 { grid-column: span 12; }
  .c4 { grid-column: span 6; }
  .c3 { grid-column: span 6; }
}
@media (max-width: 768px) {
  .grid { grid-template-columns: 1fr; }
  .c2,.c3,.c4,.c5,.c6,.c8,.c12 { grid-column: span 1; }
  .metric-grid { grid-template-columns: repeat(2,1fr); }
}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="logo">&#9883;</div>
    <h1>Quantum <span>Sniffer</span></h1>
  </div>
  <div class="header-stats">
    <div><span class="live-dot"></span>LIVE</div>
    <div>Packets <span class="stat-val" id="hdr-pkts">0</span></div>
    <div>Rate <span class="stat-val" id="hdr-pps">0</span>/s</div>
    <div>Threats <span class="stat-val" id="hdr-threats">0</span></div>
  </div>
</div>

<div class="grid">
  <!-- Metrics Row -->
  <div class="card c12">
    <div class="metric-grid">
      <div class="metric-box"><div class="metric-val" id="m-pps">0</div><div class="metric-label">Packets/sec</div></div>
      <div class="metric-box"><div class="metric-val" id="m-bps">0</div><div class="metric-label">KB/sec</div></div>
      <div class="metric-box"><div class="metric-val" id="m-total">0</div><div class="metric-label">Total Packets</div></div>
      <div class="metric-box"><div class="metric-val" id="m-up">0s</div><div class="metric-label">Uptime</div></div>
    </div>
  </div>

  <!-- Traffic + Protocol -->
  <div class="card c8">
    <div class="card-title"><span class="icon">&#128225;</span> Traffic Timeline</div>
    <canvas id="trafficChart" height="180"></canvas>
  </div>
  <div class="card c4">
    <div class="card-title"><span class="icon">&#128246;</span> Protocol Distribution</div>
    <canvas id="protoChart" height="180"></canvas>
  </div>

  <!-- Top Talkers + PQC + TLS -->
  <div class="card c4">
    <div class="card-title"><span class="icon">&#128101;</span> Top Talkers</div>
    <canvas id="talkersChart" height="180"></canvas>
  </div>
  <div class="card pqc-card c4">
    <div class="card-title" style="color:rgba(167,243,208,0.6)"><span class="icon">&#128274;</span> Post-Quantum Security</div>
    <div class="pqc-shield">&#128737;</div>
    <div class="pqc-item"><span class="label">KEM Algorithm</span><span class="value" id="pqc-alg">Kyber-512</span></div>
    <div class="pqc-item"><span class="label">Hash Chain</span><span class="value" id="pqc-chain">0</span></div>
    <div class="pqc-item"><span class="label">Encrypted Logs</span><span class="value" id="pqc-logs">0</span></div>
    <div class="pqc-item"><span class="label">Quantum Threats</span><span class="value" id="pqc-threats">0</span></div>
  </div>
  <div class="card c4">
    <div class="card-title"><span class="icon">&#128272;</span> TLS Analysis</div>
    <div id="tlsList" style="max-height:200px;overflow-y:auto"></div>
  </div>

  <!-- Alerts + Flows -->
  <div class="card c8">
    <div class="card-title"><span class="icon">&#128737;</span> Threat Alerts</div>
    <div class="alert-feed" id="alertFeed"></div>
  </div>
  <div class="card c4">
    <div class="card-title"><span class="icon">&#128279;</span> Active Flows</div>
    <div style="max-height:240px;overflow-y:auto">
      <table><thead><tr><th>Source</th><th>Dest</th><th>Proto</th><th>State</th></tr></thead>
      <tbody id="flowTable"></tbody></table>
    </div>
  </div>
</div>

<script>
const C = ['#818cf8','#a78bfa','#f472b6','#fb7185','#fbbf24','#34d399','#22d3ee','#60a5fa','#c084fc','#fb923c'];

Chart.defaults.color = '#64748b';
Chart.defaults.font.family = "'Inter', system-ui";
Chart.defaults.font.size = 11;

const trafficChart = new Chart(document.getElementById('trafficChart'), {
  type:'line', data:{labels:[], datasets:[{
    label:'pkt/s', data:[], borderColor:'#818cf8', backgroundColor:'rgba(129,140,248,0.08)',
    fill:true, tension:0.4, pointRadius:0, borderWidth:2}]},
  options:{responsive:true,maintainAspectRatio:false,
    scales:{y:{beginAtZero:true,grid:{color:'rgba(99,102,241,0.06)'},ticks:{font:{family:"'JetBrains Mono'"}}},
    x:{grid:{display:false},ticks:{maxTicksLimit:8,font:{family:"'JetBrains Mono'",size:9}}}},
    plugins:{legend:{display:false}},animation:{duration:400}}
});

const protoChart = new Chart(document.getElementById('protoChart'), {
  type:'doughnut', data:{labels:[], datasets:[{data:[], backgroundColor:C, borderWidth:0, hoverOffset:6}]},
  options:{responsive:true,maintainAspectRatio:false,cutout:'65%',
    plugins:{legend:{position:'right',labels:{color:'#94a3b8',font:{size:10},padding:8,usePointStyle:true,pointStyleWidth:8}}}}
});

const talkersChart = new Chart(document.getElementById('talkersChart'), {
  type:'bar', data:{labels:[], datasets:[{data:[], backgroundColor:C, borderRadius:4, borderSkipped:false}]},
  options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
    scales:{x:{grid:{color:'rgba(99,102,241,0.06)'},ticks:{font:{family:"'JetBrains Mono'",size:9}}},
    y:{grid:{display:false},ticks:{font:{family:"'JetBrains Mono'",size:9}}}},
    plugins:{legend:{display:false}}}
});

function fmtBytes(b){if(b<1024)return b+'B';if(b<1048576)return(b/1024).toFixed(1)+'KB';return(b/1048576).toFixed(1)+'MB';}

function update(){
  fetch('/api/state').then(r=>r.json()).then(d=>{
    if(d.traffic.length){
      trafficChart.data.labels=d.traffic.map(t=>new Date(t.time*1000).toLocaleTimeString());
      trafficChart.data.datasets[0].data=d.traffic.map(t=>t.count);
      trafficChart.update('none');
    }
    const pk=Object.keys(d.protocols),pv=Object.values(d.protocols);
    protoChart.data.labels=pk;protoChart.data.datasets[0].data=pv;protoChart.update('none');
    talkersChart.data.labels=d.top_talkers.map(t=>t.ip);
    talkersChart.data.datasets[0].data=d.top_talkers.map(t=>t.bytes);talkersChart.update('none');

    const af=document.getElementById('alertFeed');
    af.innerHTML=d.alerts.map(a=>{
      const sev=['info','low','medium','high','critical'][Math.min((a.severity||1)-1,4)];
      const t=a.timestamp?new Date(a.timestamp*1000).toLocaleTimeString():'';
      return `<div class="alert-item"><span class="badge badge-${sev}">${sev.toUpperCase()}</span><div><div class="alert-text">${a.description||''}</div>${t?`<div class="alert-time">${t}</div>`:''}</div></div>`;
    }).join('');

    const ft=document.getElementById('flowTable');
    ft.innerHTML=(d.flows||[]).map(f=>{
      const sc=f.state=='ESTABLISHED'?'est':f.state?.includes('SYN')?'syn':f.state?.includes('FIN')?'fin':'new';
      return `<tr><td>${f.src||''}</td><td>${f.dst||''}</td><td>${f.proto||''}</td><td><span class="state-badge state-${sc}">${f.state||''}</span></td></tr>`;
    }).join('');

    const tl=document.getElementById('tlsList');
    tl.innerHTML=(d.tls||[]).map(t=>{
      const gc=t.grade?.startsWith('A')?'a':t.grade=='B'?'b':t.grade=='C'?'c':t.grade=='D'?'d':'f';
      const vc=t.pqc_verdict=='SAFE'?'safe':t.pqc_verdict=='CRITICAL'?'crit':'risk';
      return `<div class="tls-item"><div class="grade grade-${gc}">${t.grade||'?'}</div><div style="flex:1;min-width:0"><div class="tls-suite" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${t.suite||''}</div><div style="margin-top:2px"><span class="tls-verdict verdict-${vc}">${t.pqc_verdict||''}</span> <span style="color:var(--text3);font-size:0.65rem">${t.kex||''} | ${t.cipher||''}</span></div></div></div>`;
    }).join('');

    document.getElementById('pqc-alg').textContent=d.pqc.kem_algorithm||'-';
    document.getElementById('pqc-chain').textContent=d.pqc.hash_chain_length||0;
    document.getElementById('pqc-logs').textContent=d.pqc.encrypted_entries||0;
    document.getElementById('pqc-threats').textContent=d.pqc.quantum_threats_found||0;

    const p=d.performance;
    document.getElementById('m-pps').textContent=(p.packets_per_sec||0).toLocaleString(undefined,{maximumFractionDigits:0});
    document.getElementById('m-bps').textContent=((p.bytes_per_sec||0)/1024).toFixed(1);
    document.getElementById('m-total').textContent=(p.total_packets||0).toLocaleString();
    const up=Math.floor(p.uptime||0);
    const m=Math.floor(up/60),s=up%60;
    document.getElementById('m-up').textContent=m>0?`${m}m ${s}s`:`${s}s`;
    document.getElementById('hdr-pkts').textContent=(p.total_packets||0).toLocaleString();
    document.getElementById('hdr-pps').textContent=(p.packets_per_sec||0).toFixed(0);
    document.getElementById('hdr-threats').textContent=p.threats_total||d.alerts.length||0;
  }).catch(()=>{});
}
setInterval(update,2000);update();
</script>
</body>
</html>"""


def create_web_app(data_store: DashboardDataStore) -> Optional[Any]:
    if not HAS_FLASK:
        print("  Flask not installed. Install with: pip install flask")
        return None
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/state")
    def api_state():
        return jsonify(data_store.get_state())

    @app.route("/api/stats")
    def api_stats():
        s = data_store.get_state()
        return jsonify({k: s[k] for k in ("protocols", "performance", "pqc")})

    @app.route("/api/alerts")
    def api_alerts():
        return jsonify(list(data_store.alerts)[:100])

    @app.route("/api/flows")
    def api_flows():
        return jsonify(data_store.active_flows[:50])

    @app.route("/api/pqc/migration")
    def api_pqc_migration():
        """PQC migration readiness endpoint."""
        try:
            scorer = getattr(data_store, '_pqc_migration_scorer', None)
            if scorer is None:
                from pqc_migration_scorer import PQCMigrationScorer
                data_store._pqc_migration_scorer = PQCMigrationScorer()
                scorer = data_store._pqc_migration_scorer
            return jsonify(scorer.get_api_response())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/metrics")
    def prometheus_metrics():
        """Prometheus metrics endpoint."""
        try:
            from metrics import get_metrics
            m = get_metrics()
            return m.generate_metrics(), 200, {'Content-Type': m.content_type}
        except Exception as e:
            return f"# Error: {e}\n", 500, {'Content-Type': 'text/plain'}

    @app.route("/health")
    def health_check():
        """Health check endpoint for Docker/Kubernetes liveness probes."""
        state = data_store.get_state()
        return jsonify({
            "status": "ok",
            "uptime": state["performance"].get("uptime", 0),
            "total_packets": state["performance"].get("total_packets", 0),
            "total_alerts": len(state["alerts"]),
            "pqc_enabled": state["pqc"].get("enabled", False),
        })

    return app


def start_web_dashboard(data_store: DashboardDataStore, port: int = 5000):
    app = create_web_app(data_store)
    if app is None:
        return
    def _run():
        import logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print(f"  Web dashboard running at http://localhost:{port}")
    return t
