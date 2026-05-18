"""
Clinical Adaptive Streaming LOSO Trainer (Route B).

Approach 1: Dynamic Source-Domain Selection
- For each test subject, select training subjects with clinically similar profiles
- Match by: ParalysisSide, NIHSS severity bucket, Duration bucket, StrokeLocation category

Approach 2 support (Conditional Feature Extractor) is prepared via config flags
but requires model/dataset changes not included in this file.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from trainers.streaming_loso_trainer import StreamingLOSOTrainer


class ClinicalStreamingLOSOTrainer(StreamingLOSOTrainer):
    """
    Streaming LOSO trainer with clinical adaptive subject selection.
    
    Config flags (under ``trainer.args``):
        - clinical_filtering.enabled (bool): Enable clinical filtering
        - clinical_filtering.features (list): Features to match, e.g.
          ["ParalysisSide", "NIHSS_bucket", "Duration_bucket"]
        - clinical_filtering.min_subjects (int): Minimum training subjects to retain
        - clinical_filtering.fallback_to_all (bool): If filtered set too small, use all
    """

    def __init__(self, config: Dict[str, Any], device: str = 'cuda', paths=None):
        super().__init__(config, device, paths)
        self.clinical_metadata: Dict[int, Dict[str, Any]] = {}
        self._filter_cfg = self.config.get("trainer.args.clinical_filtering", {})

    # ------------------------------------------------------------------
    # Clinical metadata loading
    # ------------------------------------------------------------------

    def _load_clinical_metadata(self, dataset_info: Dict[str, Any]) -> None:
        """Load participants.tsv and build metadata lookup dict."""
        data_dir = dataset_info.get("dataset", {}).get("data_dir", "")
        if not data_dir:
            self._log("Warning: data_dir not found, clinical filtering disabled", level="warning")
            return

        tsv_path = Path(data_dir).parent / "participants.tsv"
        if not tsv_path.exists():
            # Try alternate location
            tsv_path = Path(data_dir) / "participants.tsv"
        if not tsv_path.exists():
            self._log(f"Warning: participants.tsv not found at {tsv_path}, clinical filtering disabled", level="warning")
            return

        df = pd.read_csv(tsv_path, sep='\t')
        for _, row in df.iterrows():
            pid = row["Participant_ID"]
            # Convert "sub-01" → 1
            if isinstance(pid, str) and pid.startswith("sub-"):
                sid = int(pid.replace("sub-", ""))
            else:
                try:
                    sid = int(pid)
                except (ValueError, TypeError):
                    continue

            self.clinical_metadata[sid] = {
                "Gender": row.get("Gender", ""),
                "Age": self._to_float(row.get("Age")),
                "Duration": self._to_float(row.get("Duration")),
                "ParalysisSide": str(row.get("ParalysisSide", "")).lower().strip(),
                "Handedness": str(row.get("Handedness", "")).lower().strip(),
                "IsFirstTime": str(row.get("IsFirstTime", "")).lower().strip(),
                "StrokeLocation": str(row.get("StrokeLocation", "")).strip(),
                "NIHSS": self._to_float(row.get("NIHSS")),
                "MBI": self._to_float(row.get("MBI")),
                "mRS": self._to_float(row.get("mRS")),
                # Derived buckets
                "NIHSS_bucket": self._nihss_bucket(self._to_float(row.get("NIHSS"))),
                "Duration_bucket": self._duration_bucket(self._to_float(row.get("Duration"))),
                "StrokeLocation_category": self._simplify_stroke_location(str(row.get("StrokeLocation", ""))),
            }

        self._log(f"Loaded clinical metadata for {len(self.clinical_metadata)} subjects")

    @staticmethod
    def _to_float(val) -> Optional[float]:
        if pd.isna(val):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _nihss_bucket(nihss: Optional[float]) -> str:
        if nihss is None:
            return "unknown"
        if nihss <= 3:
            return "light"
        if nihss <= 7:
            return "medium"
        return "severe"

    @staticmethod
    def _duration_bucket(duration: Optional[float]) -> str:
        if duration is None:
            return "unknown"
        if duration <= 3:
            return "acute"
        return "chronic"

    @staticmethod
    def _simplify_stroke_location(location: str) -> str:
        loc = location.lower()
        if not loc or loc == "nan":
            return "unknown"
        # Count distinct anatomical regions
        regions = []
        if "cortex" in loc or "cortical" in loc:
            regions.append("cortical")
        if "pons" in loc or "brainstem" in loc or "midbrain" in loc:
            regions.append("brainstem")
        if "cerebellum" in loc:
            regions.append("cerebellum")
        if "basal ganglia" in loc or "thalamus" in loc or "internal capsule" in loc:
            regions.append("subcortical")

        if len(regions) == 0:
            # Default inference from common terms
            if any(x in loc for x in ["corona radiata", "centrum semiovale", "periventricular"]):
                return "subcortical"
            if any(x in loc for x in ["frontal", "parietal", "temporal", "occipital"]):
                return "cortical"
            return "unknown"
        if len(regions) > 1:
            return "mixed"
        return regions[0]

    # ------------------------------------------------------------------
    # Filtering logic
    # ------------------------------------------------------------------

    def _filter_train_subjects(
        self,
        test_subject_id: int,
        candidate_train_ids: List[int]
    ) -> List[int]:
        """
        Select training subjects clinically similar to the test subject.
        Returns the filtered list (may be same as input if no filtering applied).
        """
        if not self._filter_cfg.get("enabled", False):
            return candidate_train_ids

        if test_subject_id not in self.clinical_metadata:
            self._log(f"  [ClinicalFilter] No metadata for test subject {test_subject_id}, skipping filter")
            return candidate_train_ids

        test_meta = self.clinical_metadata[test_subject_id]
        features = self._filter_cfg.get("features", ["ParalysisSide", "NIHSS_bucket"])
        min_subjects = self._filter_cfg.get("min_subjects", 5)
        fallback = self._filter_cfg.get("fallback_to_all", True)

        filtered = []
        for tid in candidate_train_ids:
            if tid not in self.clinical_metadata:
                continue
            train_meta = self.clinical_metadata[tid]
            if self._is_clinically_similar(test_meta, train_meta, features):
                filtered.append(tid)

        n_orig = len(candidate_train_ids)
        n_filt = len(filtered)

        if n_filt < min_subjects:
            msg = (f"  [ClinicalFilter] Filtered from {n_orig} to {n_filt} subjects "
                   f"(below min={min_subjects})")
            if fallback:
                self._log(msg + " → falling back to ALL training subjects")
                return candidate_train_ids
            else:
                self._log(msg + " → keeping filtered set (may underfit)")
                return filtered

        self._log(f"  [ClinicalFilter] Selected {n_filt}/{n_orig} clinically similar subjects "
                  f"(features: {features})")
        return filtered

    @staticmethod
    def _is_clinically_similar(
        test_meta: Dict[str, Any],
        train_meta: Dict[str, Any],
        features: List[str]
    ) -> bool:
        """Hard-filter: all selected features must match exactly."""
        for feat in features:
            t_val = test_meta.get(feat)
            tr_val = train_meta.get(feat)
            if t_val is None or tr_val is None:
                # If either side missing this feature, treat as mismatch
                # (unless configured otherwise)
                return False
            if t_val != tr_val:
                return False
        return True

    # ------------------------------------------------------------------
    # Train — override LOSO loop to insert clinical filtering
    # ------------------------------------------------------------------

    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run streaming LOSO with clinical adaptive subject selection.
        """
        subjects = datasets['subjects']
        dataset_info = datasets.get('info', {})
        dataset_cls = datasets.get('dataset_cls')

        if not dataset_cls:
            raise ValueError("ClinicalStreamingLOSOTrainer requires 'dataset_cls'")

        # Load clinical metadata before LOSO loop
        self._load_clinical_metadata(dataset_info)

        all_results = []
        all_histories = []

        self._log("=" * 60)
        self._log("Starting Clinical Adaptive Streaming LOSO Training")
        self._log(f"Total subjects: {len(subjects)}")
        if self._filter_cfg.get("enabled", False):
            self._log(f"Clinical filtering: ENABLED")
            self._log(f"  Features: {self._filter_cfg.get('features', [])}")
            self._log(f"  Min subjects: {self._filter_cfg.get('min_subjects', 5)}")
        else:
            self._log("Clinical filtering: DISABLED (standard LOSO)")
        self._log(f"Memory-efficient mode: {self.subject_buffer_size} subject(s) in memory")
        self._log(f"Gradient accumulation: {self.gradient_accumulation_steps} steps")
        if self.mixed_precision:
            self._log("Mixed precision training: Enabled")
        self._log("=" * 60)

        # LOSO loop
        for test_subject_idx, test_subject_id in enumerate(subjects):
            self._log(f"\n[{test_subject_idx + 1}/{len(subjects)}] Test Subject: {test_subject_id}")
            self._log("-" * 40)

            # Get all candidate training subjects
            candidate_train_ids = [s for s in subjects if s != test_subject_id]

            # Apply clinical filtering
            train_subject_ids = self._filter_train_subjects(test_subject_id, candidate_train_ids)

            # Train and evaluate
            subject_results, subject_history = self._train_loso_round_streaming(
                test_subject_id, train_subject_ids, dataset_cls, dataset_info
            )

            all_results.append(subject_results)
            all_histories.append({
                'test_subject_id': test_subject_id,
                'history': subject_history
            })

            # Plot history
            subject_viz_dir = self.paths.get_subject_viz_dir(test_subject_id)
            self._plot_history(subject_history, subject_viz_dir, test_subject_id)

            import gc
            gc.collect()

        # Final comparison
        self._plot_comparison(all_results)

        # Statistics
        test_accuracies = [r['test_acc'] for r in all_results]
        final_results = {
            'subjects': all_results,
            'overall_mean': float(np.mean(test_accuracies)),
            'overall_std': float(np.std(test_accuracies)),
            'overall_min': float(np.min(test_accuracies)),
            'overall_max': float(np.max(test_accuracies)),
            'clinical_filtering_config': self._filter_cfg,
            'streaming_config': {
                'num_subjects': len(subjects),
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'subject_buffer_size': self.subject_buffer_size,
                'mixed_precision': self.mixed_precision
            }
        }

        # Optionally save training histories for diagnostic analysis
        save_hist = self.config.get("trainer.args.save_training_history", False)
        if save_hist and all_histories:
            try:
                import json
                hist_path = self.paths.logs_dir / "training_histories.json"
                def _convert(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif hasattr(obj, "item"):
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {k: _convert(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [_convert(v) for v in obj]
                    return obj
                with open(hist_path, "w", encoding="utf-8") as f:
                    json.dump(_convert(all_histories), f, indent=2)
                self._log(f"Training histories saved to: {hist_path}")
            except Exception as e:
                self._log(f"Warning: Failed to save training histories: {e}", level="warning")

        self._log("\n" + "=" * 60)
        self._log("Clinical Adaptive Streaming LOSO Training Complete")
        self._log(f"Mean Accuracy: {final_results['overall_mean']:.4f} ± {final_results['overall_std']:.4f}")
        self._log("=" * 60)

        return final_results
