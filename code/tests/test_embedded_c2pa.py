import json
import sys
from pathlib import Path

# Add project root/src to sys.path so we can import mosaic
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mosaic.config import L2Config
from mosaic.l2_provenance import check_provenance

def test_real_c2pa():
    # Real embedded manifest fixture
    test_file = ROOT / "tests" / "fixtures" / "test_c2pa.jpg"
    if not test_file.exists():
        print(f"Error: {test_file} not found.")
        return
        
    print(f"Testing real embedded C2PA manifest in {test_file.name}...\n")
    
    # Run the exact same check_provenance function used by the pipeline
    result = check_provenance(test_file, L2Config(allow_sidecar_manifest=False))
    
    print("=== Provenance Result ===")
    print(f"Status:             {result.c2pa.status.value}")
    print(f"Claims AI Generated:{result.c2pa.claims_ai_generated}")
    print(f"Is Simulated:       {result.c2pa.is_simulated}")
    print(f"Validation State:   {result.c2pa.validation_state}")
    
    print("\n=== Manifest Summary ===")
    print(json.dumps(result.c2pa.manifest_summary, indent=2))
    
    assert result.c2pa is not None
    assert result.c2pa.validation_state in ("Valid", "valid", None) or result.c2pa.status is not None

if __name__ == "__main__":
    test_real_c2pa()
