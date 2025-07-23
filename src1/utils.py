# src1/utils.py

import pandas as pd

def save_warning_report(warnings, max_anomaly):
    print("--- Final System Report ---")
    print(f"Total Warnings: {len(warnings)}")
    print(f"Max Anomaly: {max_anomaly:.1f}x background")
    if warnings:
        for i, warning in enumerate(warnings):
            print(f"\n--- Warning #{i+1} ---")
            for key, value in warning.items():
                print(f"  {key:<25}: {value}")
    print("-" * 28)
    pd.DataFrame(warnings).to_csv("issued_alerts.csv", index=False)
