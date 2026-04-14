# ═══════════════════════════════════════════════════════════════════
# Quantum Sniffer — Dockerfile
# ═══════════════════════════════════════════════════════════════════
# Multi-stage build for minimal production image
#
# Usage:
#   docker build -t quantum-sniffer .
#   docker run --net=host --cap-add=NET_RAW quantum-sniffer
#   docker run quantum-sniffer --mode simulate
#   docker-compose up
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: Build dependencies ──
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install pyyaml

# ── Stage 2: Production image ──
FROM python:3.12-slim

LABEL maintainer="Quantum Sniffer Team"
LABEL description="Post-Quantum Protected Network Security Platform"
LABEL version="2.0"

# Install libpcap for Scapy
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpcap0.8 tcpdump && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /install /usr/local

# App directory
WORKDIR /app
COPY . .

# Create directories
RUN mkdir -p /app/pqc_logs /app/forensics /app/exports

# Non-root user (for non-capture modes)
RUN useradd -m -r qsniffer
# Note: capture mode requires --cap-add=NET_RAW or --net=host

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV QS_LOGGING_LEVEL=INFO
ENV QS_PQC_LOGDIR=/app/pqc_logs

# Expose ports
EXPOSE 5000/tcp
EXPOSE 9090/tcp
EXPOSE 9100/tcp

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Default: run in simulate mode (no network access needed)
ENTRYPOINT ["python", "-m", "quantum_sniffer"]
CMD ["--mode", "simulate"]
