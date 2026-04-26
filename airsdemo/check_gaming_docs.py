#!/usr/bin/env python3
"""
Quick script to verify gaming documents are in the right place
"""
from pathlib import Path

# Check local
local_path = Path("gaming/documents")
print(f"LOCAL - Checking: {local_path.absolute()}")
if local_path.exists():
    files = list(local_path.glob("*"))
    print(f"✅ Found {len(files)} files:")
    for f in files:
        if f.is_file():
            size = f.stat().st_size
            print(f"  - {f.name} ({size:,} bytes)")
else:
    print(f"❌ Directory not found")

print()

# Check production
prod_path = Path("/var/www/airsdemo/gaming/documents")
print(f"PRODUCTION - Checking: {prod_path}")
if prod_path.exists():
    files = list(prod_path.glob("*"))
    print(f"✅ Found {len(files)} files:")
    for f in files:
        if f.is_file():
            size = f.stat().st_size
            print(f"  - {f.name} ({size:,} bytes)")
else:
    print(f"❌ Directory not found (you're not on GCP)")
