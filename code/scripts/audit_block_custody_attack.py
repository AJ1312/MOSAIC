import os
import json
import uuid
import copy
import hashlib
from mosaic.l4_custody import LocalLedger

def run_tamper_matrix():
    print("Running L4 Custody Tamper Matrix...")
    ledger_path = "outputs_real/06_custody/ledger_test.jsonl"
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    if os.path.exists(ledger_path):
        os.remove(ledger_path)
        
    ledger = LocalLedger(ledger_path)
    
    # Create valid entry
    rec1 = {
        "timestamp": "2026-08-29T00:00:00Z",
        "asset_hash": "dummy_asset_hash",
        "combined_root": "dummy_root_1",
        "verdict": "REAL",
        "confidence": 0.95
    }
    
    rec2 = {
        "timestamp": "2026-08-29T00:01:00Z",
        "asset_hash": "dummy_asset_hash_2",
        "combined_root": "dummy_root_2",
        "verdict": "FAKE",
        "confidence": 0.88
    }
    
    ledger.append(rec1)
    ledger.append(rec2)
    
    # 1. Honest verification
    print("Test 1: Honest ledger verification")
    v_res = ledger.verify()
    print(f"Valid: {v_res['valid']} (Err: {v_res.get('reason')})")
    
    # Read the ledger lines
    with open(ledger_path, 'r') as f:
        lines = f.readlines()
        
    # Attack 1: Modify a field in the first entry
    print("Attack 1: Modify probability/confidence in ledger entry")
    entry1 = json.loads(lines[0])
    entry1['payload']['confidence'] = 0.99  # tampered
    tampered_lines = [json.dumps(entry1) + '\n', lines[1]]
    
    tamper_path1 = "outputs_real/06_custody/ledger_tamper1.jsonl"
    with open(tamper_path1, 'w') as f:
        f.writelines(tampered_lines)
        
    tamper_ledger1 = LocalLedger(tamper_path1)
    v_res1 = tamper_ledger1.verify()
    print(f"Attack 1 caught: {not v_res1['valid']} (Err: {v_res1.get('reason')})")
    
    # Attack 2: Reorder entries
    print("Attack 2: Reorder ledger entries")
    tampered_lines = [lines[1], lines[0]]
    tamper_path2 = "outputs_real/06_custody/ledger_tamper2.jsonl"
    with open(tamper_path2, 'w') as f:
        f.writelines(tampered_lines)
        
    tamper_ledger2 = LocalLedger(tamper_path2)
    v_res2 = tamper_ledger2.verify()
    print(f"Attack 2 caught: {not v_res2['valid']} (Err: {v_res2.get('reason')})")
    
    # Attack 3: Delete entry
    print("Attack 3: Delete ledger entry")
    tampered_lines = [lines[1]]
    tamper_path3 = "outputs_real/06_custody/ledger_tamper3.jsonl"
    with open(tamper_path3, 'w') as f:
        f.writelines(tampered_lines)
        
    tamper_ledger3 = LocalLedger(tamper_path3)
    v_res3 = tamper_ledger3.verify()
    print(f"Attack 3 caught: {not v_res3['valid']} (Err: {v_res3.get('reason')})")
    
    results = {
        "honest_verification": v_res['valid'],
        "attack_modify_record": not v_res1['valid'],
        "attack_reorder_records": not v_res2['valid'],
        "attack_delete_record": not v_res3['valid']
    }
    
    with open('outputs_real/05_ablation/custody_tamper_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    run_tamper_matrix()
