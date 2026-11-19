#!/bin/bash
cd "$(dirname "$0")/.."
source .venv/bin/activate
while pgrep -f scripts/make_fp64_ref.py >/dev/null; do sleep 15; done
caffeinate -i -s python scripts/certify.py --batch 2
python scripts/export_ref.py
echo CERT DONE
