"""
Unified Cross-Layer Explanation Synthesizer
=============================================
Correlates rule-based IDS detections (from ids.py) with ML-based
anomaly detections (from iforest_detector.py) on the same flow within
a configurable time window (default: 30s).

When both layers flag the same flow:
  1. Computes joint_confidence = 1 − (1−P_rule)(1−P_ml)
  2. Merges evidence chains from both detectors
  3. Generates a natural-language cross-layer explanation
  4. Emits a CORRELATED_ALERT ThreatEvent at CRITICAL severity

Design rationale:
  Cross-layer correlation dramatically reduces false-positive rates.
  If BOTH an expert rule system AND an unsupervised ML model flag the
  same flow, the probability of a true attack is much higher than
  either alone. The joint confidence formula models this as independent
  Bernoulli events.

Thread-safe for concurrent IDS and ML pipelines.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ids import ThreatEvent, EvidenceFactor, Severity

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Alert Records
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RuleAlertRecord:
    """Stashed rule-based IDS alert for correlation."""
    threat_event: ThreatEvent
    flow_key: str
    timestamp: float
    confidence: float = 0.0


@dataclass
class MLAlertRecord:
    """Stashed ML-based anomaly alert for correlation."""
    flow_key: str
    anomaly_score: float
    feature_importances: Dict[str, float]
    detector_type: str  # "iforest", "eif", "autoencoder"
    timestamp: float
    confidence: float = 0.0


@dataclass
class CorrelatedAlert:
    """Result of cross-layer co-detection."""
    flow_key: str
    joint_confidence: float
    rule_event: ThreatEvent
    ml_score: float
    ml_detector_type: str
    explanation: str
    timestamp: float
    threat_event: Optional[ThreatEvent] = None  # Synthesized ThreatEvent


# ──────────────────────────────────────────────────────────────────────
# Unified Explainer
# ──────────────────────────────────────────────────────────────────────

class UnifiedExplainer:
    """Cross-layer correlation engine for IDS + ML co-detections.

    Maintains a sliding window of rule-based and ML-based alerts.
    When both layers flag the same flow within `correlation_window`
    seconds, synthesizes a cross-layer CORRELATED_ALERT.

    Args:
        correlation_window: Max seconds between rule and ML alerts
                           to qualify as co-detection (default: 30).
        max_pending: Max stashed alerts per category before cleanup (default: 1000).
    """

    def __init__(
        self,
        correlation_window: float = 30.0,
        max_pending: int = 1000,
    ):
        self.correlation_window = correlation_window
        self.max_pending = max_pending

        self._lock = threading.Lock()
        # Pending alerts keyed by flow_key
        self._pending_rule: Dict[str, List[RuleAlertRecord]] = defaultdict(list)
        self._pending_ml: Dict[str, List[MLAlertRecord]] = defaultdict(list)

        # Statistics
        self._total_rule_alerts: int = 0
        self._total_ml_alerts: int = 0
        self._correlated_alerts_total: int = 0
        self._correlation_history: List[CorrelatedAlert] = []

    def record_rule_alert(
        self,
        flow_key: str,
        threat_event: ThreatEvent,
    ) -> Optional[CorrelatedAlert]:
        """Record a rule-based IDS alert and check for ML correlation.

        Args:
            flow_key: Flow identifier (e.g., "10.0.0.1->10.0.0.2:443").
            threat_event: The ThreatEvent from the IDS engine.

        Returns:
            CorrelatedAlert if an ML alert exists for the same flow, else None.
        """
        now = time.time()
        record = RuleAlertRecord(
            threat_event=threat_event,
            flow_key=flow_key,
            timestamp=now,
            confidence=threat_event.confidence,
        )

        with self._lock:
            self._total_rule_alerts += 1
            self._pending_rule[flow_key].append(record)
            self._cleanup_stale(now)

            # Check for matching ML alert
            return self._check_correlation(flow_key, now)

    def record_ml_alert(
        self,
        flow_key: str,
        anomaly_score: float,
        feature_importances: Optional[Dict[str, float]] = None,
        detector_type: str = "iforest",
    ) -> Optional[CorrelatedAlert]:
        """Record an ML anomaly alert and check for rule correlation.

        Args:
            flow_key: Flow identifier.
            anomaly_score: Combined anomaly score from the ML pipeline.
            feature_importances: Feature → deviation mapping for explainability.
            detector_type: Which ML detector flagged this ("iforest", "eif", etc.).

        Returns:
            CorrelatedAlert if a rule alert exists for the same flow, else None.
        """
        now = time.time()
        record = MLAlertRecord(
            flow_key=flow_key,
            anomaly_score=anomaly_score,
            feature_importances=feature_importances or {},
            detector_type=detector_type,
            timestamp=now,
            confidence=anomaly_score,
        )

        with self._lock:
            self._total_ml_alerts += 1
            self._pending_ml[flow_key].append(record)
            self._cleanup_stale(now)

            # Check for matching rule alert
            return self._check_correlation(flow_key, now)

    def _check_correlation(
        self, flow_key: str, now: float,
    ) -> Optional[CorrelatedAlert]:
        """Check if both rule and ML alerts exist for a flow within the window.

        Must be called with self._lock held.
        """
        rule_alerts = self._pending_rule.get(flow_key, [])
        ml_alerts = self._pending_ml.get(flow_key, [])

        if not rule_alerts or not ml_alerts:
            return None

        # Find the most recent of each
        latest_rule = max(rule_alerts, key=lambda r: r.timestamp)
        latest_ml = max(ml_alerts, key=lambda r: r.timestamp)

        # Check time window
        time_diff = abs(latest_rule.timestamp - latest_ml.timestamp)
        if time_diff > self.correlation_window:
            return None

        # Correlation found!
        rule_conf = latest_rule.confidence
        ml_conf = latest_ml.confidence

        # Joint confidence: 1 − (1−P_rule)(1−P_ml)
        joint_confidence = 1.0 - (1.0 - rule_conf) * (1.0 - ml_conf)
        joint_confidence = min(1.0, max(0.0, joint_confidence))

        # Generate cross-layer explanation
        explanation = self._generate_explanation(
            flow_key, latest_rule, latest_ml, joint_confidence
        )

        # Build merged evidence
        merged_evidence = list(latest_rule.threat_event.evidence_factors or [])

        # Add ML feature importances as evidence
        for feat_name, importance in sorted(
            latest_ml.feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]:
            merged_evidence.append(EvidenceFactor(
                metric=f"ml_{feat_name}",
                observed=f"{importance:.2f}σ",
                threshold="baseline",
                weight=min(1.0, abs(importance) / 5.0),
            ))

        # Synthesize CORRELATED_ALERT ThreatEvent
        threat_event = ThreatEvent(
            timestamp=now,
            severity=Severity.CRITICAL,
            category="CORRELATED_ALERT",
            description=(
                f"Cross-layer co-detection on flow {flow_key}: "
                f"Rule-based ({latest_rule.threat_event.category}) + "
                f"ML ({latest_ml.detector_type}) agree. "
                f"Joint confidence: {joint_confidence:.1%}."
            ),
            confidence=joint_confidence,
            mitre_ref=latest_rule.threat_event.mitre_ref,
            explanation=explanation,
            evidence_factors=merged_evidence,
            detection_logic=(
                f"Cross-layer correlation: IDS rule '{latest_rule.threat_event.category}' "
                f"(conf={rule_conf:.3f}) + ML {latest_ml.detector_type} "
                f"(score={latest_ml.anomaly_score:.3f}) co-detected within "
                f"{self.correlation_window}s window. "
                f"Joint confidence = 1 − (1−{rule_conf:.3f})(1−{ml_conf:.3f}) "
                f"= {joint_confidence:.4f}."
            ),
            response_actions=[
                "IMMEDIATE: Investigate correlated flow for active threat",
                "Cross-reference both rule evidence and ML feature deviations",
                "Check temporal correlation for campaign-level activity",
                "Consider blocking source IP pending investigation",
                "Escalate to SOC team — high-confidence multi-layer detection",
            ],
        )

        correlated = CorrelatedAlert(
            flow_key=flow_key,
            joint_confidence=joint_confidence,
            rule_event=latest_rule.threat_event,
            ml_score=latest_ml.anomaly_score,
            ml_detector_type=latest_ml.detector_type,
            explanation=explanation,
            timestamp=now,
            threat_event=threat_event,
        )

        self._correlated_alerts_total += 1
        self._correlation_history.append(correlated)

        # Clear pending alerts for this flow (avoid duplicate correlations)
        self._pending_rule.pop(flow_key, None)
        self._pending_ml.pop(flow_key, None)

        logger.warning(
            f"CORRELATED_ALERT: {flow_key} — joint_conf={joint_confidence:.4f} "
            f"(rule={rule_conf:.3f}, ml={ml_conf:.3f})"
        )

        return correlated

    def _generate_explanation(
        self,
        flow_key: str,
        rule: RuleAlertRecord,
        ml: MLAlertRecord,
        joint_confidence: float,
    ) -> str:
        """Generate natural-language cross-layer explanation."""
        rule_ev = rule.threat_event

        # Rule-based explanation
        rule_desc = rule_ev.description or rule_ev.category
        rule_evidence_str = ""
        if rule_ev.evidence_factors:
            top_evidence = rule_ev.evidence_factors[:3]
            rule_evidence_str = "; ".join(
                f"{e.metric}={e.observed} (threshold {e.threshold})"
                for e in top_evidence
            )

        # ML explanation
        ml_top_features = sorted(
            ml.feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]
        ml_feat_str = ", ".join(
            f"{name} ({val:+.1f}σ)" for name, val in ml_top_features
        )

        explanation = (
            f"CROSS-LAYER ANALYSIS for flow {flow_key}:\n"
            f"\n"
            f"• Rule-based IDS detected: {rule_desc}\n"
        )
        if rule_evidence_str:
            explanation += f"  Evidence: {rule_evidence_str}\n"

        explanation += (
            f"  Confidence: {rule.confidence:.1%} | "
            f"MITRE: {rule_ev.mitre_ref or 'N/A'}\n"
            f"\n"
            f"• ML anomaly detector ({ml.detector_type}) flagged this flow:\n"
            f"  Score: {ml.anomaly_score:.3f}\n"
        )
        if ml_feat_str:
            explanation += f"  Top deviating features: {ml_feat_str}\n"

        explanation += (
            f"\n"
            f"• Joint confidence: {joint_confidence:.1%}\n"
            f"  Formula: 1 − (1−{rule.confidence:.3f})(1−{ml.confidence:.3f})\n"
            f"\n"
            f"VERDICT: Both independent detection layers agree that flow {flow_key} "
            f"exhibits malicious behavior. This cross-validation significantly "
            f"reduces false-positive probability. Immediate investigation recommended."
        )

        return explanation

    def _cleanup_stale(self, now: float) -> None:
        """Remove alerts older than the correlation window (must hold _lock)."""
        cutoff = now - self.correlation_window * 2  # 2x window as buffer

        for key in list(self._pending_rule.keys()):
            self._pending_rule[key] = [
                r for r in self._pending_rule[key]
                if r.timestamp > cutoff
            ]
            if not self._pending_rule[key]:
                del self._pending_rule[key]

        for key in list(self._pending_ml.keys()):
            self._pending_ml[key] = [
                r for r in self._pending_ml[key]
                if r.timestamp > cutoff
            ]
            if not self._pending_ml[key]:
                del self._pending_ml[key]

        # Cap pending alerts
        while sum(len(v) for v in self._pending_rule.values()) > self.max_pending:
            oldest_key = min(self._pending_rule, key=lambda k: self._pending_rule[k][0].timestamp)
            self._pending_rule[oldest_key].pop(0)
            if not self._pending_rule[oldest_key]:
                del self._pending_rule[oldest_key]

        while sum(len(v) for v in self._pending_ml.values()) > self.max_pending:
            oldest_key = min(self._pending_ml, key=lambda k: self._pending_ml[k][0].timestamp)
            self._pending_ml[oldest_key].pop(0)
            if not self._pending_ml[oldest_key]:
                del self._pending_ml[oldest_key]

    @property
    def correlated_alerts_total(self) -> int:
        """Total correlated alerts emitted (for Prometheus)."""
        return self._correlated_alerts_total

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-format metrics."""
        return (
            "# HELP correlated_alerts_total Total cross-layer correlated alerts\n"
            "# TYPE correlated_alerts_total counter\n"
            f"correlated_alerts_total {self._correlated_alerts_total}\n"
        )

    def get_stats(self) -> dict:
        """Get explainer statistics."""
        with self._lock:
            pending_rule = sum(len(v) for v in self._pending_rule.values())
            pending_ml = sum(len(v) for v in self._pending_ml.values())
        return {
            "total_rule_alerts": self._total_rule_alerts,
            "total_ml_alerts": self._total_ml_alerts,
            "correlated_alerts_total": self._correlated_alerts_total,
            "pending_rule_alerts": pending_rule,
            "pending_ml_alerts": pending_ml,
            "correlation_window": self.correlation_window,
        }
