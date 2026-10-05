import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from football_profiler.importers import prepare_demo
p=argparse.ArgumentParser();p.add_argument("--audit-dir",required=True);a=p.parse_args()
print(prepare_demo(a.audit_dir))
