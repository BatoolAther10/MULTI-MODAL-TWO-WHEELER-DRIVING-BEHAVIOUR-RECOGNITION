"""Persist the implementation status of unavailable or rejected models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_layer.stub_adapter import EgoDriveMaxAdapter, EgoDriveRTAdapter


def write_status() -> Path:
    """Write the EgoDrive implementation decisions to a JSON log."""
    statuses = [
        {
            "model": "egodrive_rt",
            "status": "NOT_IMPLEMENTED",
            "reason": EgoDriveRTAdapter.reason,
        },
        {
            "model": "egodrive_max",
            "status": "REJECTED",
            "reason": EgoDriveMaxAdapter.reason,
        },
    ]
    output = ROOT / "logs" / "egodrive_status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    for status in statuses:
        print(f"{status['model']}: {status['status']} - {status['reason']}")
    return output


if __name__ == "__main__":
    write_status()
