"""
Webhook Alerting Service for Data Drift and Covariate Shift.

Formats and dispatches Slack/Discord notifications detailing drifted features,
statistical thresholds breached, and recommended MLOps remediation actions.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_slack_alert_payload(
    drift_results: Dict[str, Any],
    service_name: str = "B2B SaaS Churn Early-Warning Microservice",
    model_version: str = "1.0.0",
) -> Dict[str, Any]:
    """
    Construct rich Slack Blocks JSON payload summarizing the statistical drift event.
    """
    total_feats = drift_results.get("total_features", 0)
    drifted_count = drift_results.get("drifted_features_count", 0)
    drift_share = drift_results.get("drift_share", 0.0) * 100.0
    drifted_list = drift_results.get("drifted_features", [])

    # Format feature details
    feature_bullets = []
    for f in drifted_list[:8]:  # display up to top 8 drifted features
        feat_name = f.get("feature", "unknown")
        test_type = f.get("test", "test")
        stat_val = f.get("statistic", 0.0)
        p_val = f.get("p_value")
        p_str = f", p={p_val:.5f}" if p_val is not None else ""
        ref_mean = f.get("ref_mean")
        cur_mean = f.get("cur_mean")
        shift_str = f" (mean: {ref_mean} -> {cur_mean})" if ref_mean is not None else ""
        feature_bullets.append(f"• *`{feat_name}`*: {test_type} stat={stat_val}{p_str}{shift_str}")

    feature_text = "\n".join(feature_bullets) if feature_bullets else "No details available."

    payload = {
        "text": f":rotating_light: [MLOps Drift Alert] {service_name} detected covariate shift!",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 MLOps Alert: Covariate Shift Detected",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Service:*\n{service_name}"},
                    {"type": "mrkdwn", "text": f"*Model Version:*\nv{model_version}"},
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"*Drifted Features:*\n{drifted_count} / {total_feats} "
                            f"({drift_share:.1f}%)"
                        ),
                    },
                    {
                        "type": "mrkdwn",
                        "text": "*Threshold:*\nKS-test p < 0.05 | PSI > 0.10",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Breached Behavioral Features:*\n{feature_text}",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"⏰ *Timestamp:* "
                            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | "
                            "Action: Trigger model retraining and audit upstream telemetry."
                        ),
                    }
                ],
            },
        ],
    }
    return payload


def send_drift_alert(
    drift_results: Dict[str, Any],
    webhook_url: Optional[str] = None,
    log_dir: str = "monitoring/reports",
) -> bool:
    """
    Dispatch alert to configured webhook. If no webhook URL is set,
    persists alert payload to disk as an audit trail.
    """
    url = (
        webhook_url or os.environ.get("SLACK_WEBHOOK_URL") or os.environ.get("DISCORD_WEBHOOK_URL")
    )
    payload = build_slack_alert_payload(drift_results)

    # Always persist alert payload locally for audit & verification
    log_path = Path(log_dir) / "last_drift_alert.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved drift alert payload to %s", log_path)

    if not url:
        logger.info(
            "No webhook URL configured (SLACK_WEBHOOK_URL unset). "
            "Alert payload saved to %s for local verification.",
            log_path,
        )
        return True

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Successfully dispatched drift alert webhook (HTTP %d).", response.status_code)
        return True
    except Exception as e:
        logger.error("Failed to send webhook alert: %s", str(e))
        return False
