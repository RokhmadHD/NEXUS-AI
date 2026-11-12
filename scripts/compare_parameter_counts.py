"""Verify the EXP-003A parameter budget for NEXUS-HSR and its baseline."""
import json

from nexus_ai.models import NexusHSR, TransformerBaseline


def main():
    nexus = NexusHSR()
    baseline = TransformerBaseline()

    nexus_count = nexus.count_parameters()
    baseline_count = baseline.count_parameters()
    difference = abs(nexus_count - baseline_count)
    difference_percent = difference / nexus_count * 100

    report = {
        "experiment": "EXP-003A",
        "nexus_hsr_parameters": nexus_count,
        "transformer_baseline_parameters": baseline_count,
        "absolute_difference": difference,
        "difference_percent": round(difference_percent, 6),
        "tolerance_percent": 0.1,
        "parameter_matched": difference_percent <= 0.1,
    }
    print(json.dumps(report, indent=2))

    if not report["parameter_matched"]:
        raise SystemExit("Parameter budget mismatch exceeds 0.1%")


if __name__ == "__main__":
    main()
