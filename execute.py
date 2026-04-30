#!/usr/bin/env python
"""Run Phase 1 Training"""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "run_phase1.py"],
    cwd=r"c:\Users\vikas\source\repos\tennis-prediction-engine"
)
sys.exit(result.returncode)
