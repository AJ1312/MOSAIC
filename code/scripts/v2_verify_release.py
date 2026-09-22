"""Verify the non-scientific release gates without declaring the paper ready."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    checks = {}
    required = [
        "research_v2/00_research_spec/V0_CODEBASE_AUDIT.md",
        "research_v2/00_research_spec/V0_ARTIFACT_INVENTORY.csv",
        "_archive/V0/ARCHIVE_MANIFEST.csv",
        "research_v2/11_reproducibility/device_profile.json",
        "research_v2/11_reproducibility/software_lock.txt",
        "research_v2/02_dataset_registry/dataset_registry.csv",
        "research_v2/02_dataset_registry/local_manifest_v2.csv",
        "research_v2/02_dataset_registry/local_manifest_validation.json",
        "research_v2/06_experiments/pilot_resources.json",
        "research_v2/09_claim_ledger/claims.csv",
    ]
    checks["required_files"] = {p: (ROOT / p).is_file() for p in required}
    validation = json.loads((ROOT / "research_v2/02_dataset_registry/local_manifest_validation.json").read_text())
    checks["local_manifest_hash_gate"] = validation["status"] == "PASS" and validation["n_hash_mismatched"] == 0
    rows = list(csv.DictReader((ROOT / "research_v2/09_claim_ledger/claims.csv").open()))
    checks["claim_ledger_schema"] = len(rows) >= 10 and all(r["STATUS"] in {"CONDITIONAL", "UNEVALUATED"} for r in rows)
    checks["av_not_overclaimed"] = next(r for r in rows if r["CLAIM_ID"] == "V2-AV-001")["STATUS"] == "UNEVALUATED"
    checks["c2pa_not_overclaimed"] = next(r for r in rows if r["CLAIM_ID"] == "V2-C2PA-001")["STATUS"] == "UNEVALUATED"
    checks["result_freeze"] = "NOT FROZEN" in (ROOT / "research_v2/00_research_spec/RESULT_FREEZE_STATUS.md").read_text()
    checks["release_status"] = "NOT IEEE ACCESS READY" in (ROOT / "research_v2/12_release/RELEASE_AUDIT.md").read_text()
    checks["internal_consistency_status"] = "PASS" if all(checks[k] for k in checks if k not in {"status", "required_files", "internal_consistency_status"}) and all(checks["required_files"].values()) else "FAIL"
    checks["status"] = "NOT_RELEASE_READY"
    out = ROOT / "research_v2/12_release/release_gate_check.json"
    out.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))

if __name__ == "__main__": main()
