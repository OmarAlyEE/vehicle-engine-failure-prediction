import pandas as pd
import numpy as np
import json
from datetime import datetime
import os


class DriftDetector:
    def __init__(self, reference_path: str):
        """
        reference_path: path to training (reference) dataset
        """
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference file not found: {reference_path}")

        self.reference_data = pd.read_csv(reference_path)
        self.reference_stats = self._compute_stats(self.reference_data)

    def _compute_stats(self, df: pd.DataFrame):
        """
        Compute mean and std for numerical columns
        """
        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std())
            }

        return stats

    def detect_drift(self, new_data_path: str, threshold: float = 0.2):
        """
        Compare new data distribution with reference data using normalized mean shift
        """
        if not os.path.exists(new_data_path):
            raise FileNotFoundError(f"New data file not found: {new_data_path}")

        new_data = pd.read_csv(new_data_path)
        new_stats = self._compute_stats(new_data)

        drift_report = {}
        drift_detected = False

        for col in self.reference_stats:
            if col not in new_stats:
                continue

            ref_mean = self.reference_stats[col]["mean"]
            ref_std = self.reference_stats[col]["std"]
            new_mean = new_stats[col]["mean"]

            # Avoid division by zero
            if ref_std == 0:
                continue

            shift = abs(new_mean - ref_mean) / ref_std

            drift_flag = shift > threshold

            drift_report[col] = {
                "reference_mean": ref_mean,
                "new_mean": new_mean,
                "shift_score": round(shift, 4),
                "drift": drift_flag
            }

            if drift_flag:
                drift_detected = True

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "drift_detected": drift_detected,
            "details": drift_report
        }

        return result


if __name__ == "__main__":
    # Base project directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Paths
    reference_path = os.path.join(BASE_DIR, "data", "processed", "features.csv")

    # 🔥 IMPORTANT: use different file for drift testing
    new_data_path = os.path.join(BASE_DIR, "data", "processed", "new_data.csv")

    # Output path
    output_path = os.path.join(BASE_DIR, "monitoring", "drift_report.json")

    # Ensure monitoring folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize detector
    detector = DriftDetector(reference_path)

    # Run drift detection
    report = detector.detect_drift(new_data_path)

    # Print result
    print(json.dumps(report, indent=4))

    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nDrift report saved to: {output_path}")