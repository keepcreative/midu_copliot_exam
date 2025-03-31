"""
Author: marvo marvo@qq.com
Date: 2025-03-31 10:20:37
LastEditors: marvo marvo@qq.com
LastEditTime: 2025-03-31 16:13:20
FilePath: train.py
Description: Optimized training script with enhanced features
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import ClassifierMixin
import logging
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib

matplotlib.use('Agg')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)


class ModelTrainer:
    def __init__(
            self,
            train_data_path: str = "train.jsonl",
            model_dir: str = "models",
            tensorboard_log_dir: str = "logs/tensorboard",
            model_params: Dict[str, Any] = None,
            test_size: float = 0.2,
            random_state: int = 42,
            n_folds: int = 5
    ):
        """Initialize with JSONL support"""
        self.train_data_path = Path(train_data_path)
        self.model_dir = Path(model_dir)
        self.tensorboard_log_dir = Path(tensorboard_log_dir)
        self.model_params = model_params or {
            "n_estimators": 200,
            "class_weight": "balanced",
            "random_state": random_state
        }
        self.test_size = test_size
        self.random_state = random_state
        self.n_folds = n_folds
        self.model: Optional[ClassifierMixin] = None
        self.logger = logging.getLogger(__name__)
        self.writer = None
        self.label_binarizer = LabelBinarizer()

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
        """Load JSONL data with strict label validation"""
        try:
            if not self.train_data_path.exists():
                raise FileNotFoundError(f"Training file {self.train_data_path} not found")

            # Define exactly which classes we expect
            expected_classes = {1, 2, 3, 4, 5, 7}
            data = []
            problematic_samples = []

            with open(self.train_data_path, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if not {'feature', 'label', 'id'}.issubset(item.keys()):
                            raise ValueError("Invalid data format in line")

                        # Convert label to int if it's stored as string
                        label = int(item['label'])

                        if label not in expected_classes:
                            problematic_samples.append((item['id'], label))
                            continue

                        item['label'] = label  # Ensure label is int
                        data.append(item)
                    except (json.JSONDecodeError, ValueError) as e:
                        self.logger.warning(f"Skipping invalid line: {line.strip()} - Error: {str(e)}")

            if problematic_samples:
                self.logger.error(f"Found {len(problematic_samples)} invalid labels:")
                for sample_id, bad_label in problematic_samples[:10]:  # Show first 10 bad samples
                    self.logger.error(f"  Sample {sample_id} has invalid label {bad_label}")
                if len(problematic_samples) > 10:
                    self.logger.error(f"  ...plus {len(problematic_samples) - 10} more")
                return None, None, None

            features = np.array([item['feature'] for item in data])
            labels = np.array([item['label'] for item in data])
            ids = [item['id'] for item in data]

            # Get actual class distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            class_dist = dict(zip(unique_labels, counts))

            self.logger.info("Class distribution:")
            for cls, count in sorted(class_dist.items()):
                self.logger.info(f"  Class {cls}: {count} samples ({count / len(labels):.1%})")

            # Fit label binarizer with expected classes in correct order
            self.label_binarizer.fit(sorted(expected_classes))

            self.logger.info(f"Successfully loaded {len(features)} samples")
            return features, labels, ids

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}", exc_info=True)
            return None, None, None

    def _save_model(self, model: ClassifierMixin) -> Path:
        """Save trained model to disk with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"model_{timestamp}.joblib"

        try:
            joblib.dump(model, model_path)
            self.logger.info(f"Model successfully saved to {model_path}")
            return model_path
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise
    def _evaluate(
            self,
            model: ClassifierMixin,
            X: np.ndarray,
            y: np.ndarray,
            prefix: str = "Validation",
            step: Optional[int] = None
    ) -> Dict[str, float]:
        """Enhanced multi-class evaluation"""
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision_macro": precision_score(y, y_pred, average='macro'),
            "recall_macro": recall_score(y, y_pred, average='macro'),
            "f1_macro": f1_score(y, y_pred, average='macro'),
            "precision_weighted": precision_score(y, y_pred, average='weighted'),
            "recall_weighted": recall_score(y, y_pred, average='weighted'),
            "f1_weighted": f1_score(y, y_pred, average='weighted'),
            "report": classification_report(y, y_pred, digits=4)
        }

        # Calculate ROC-AUC for multi-class
        try:
            y_bin = self.label_binarizer.transform(y)
            metrics["roc_auc_ovr"] = roc_auc_score(y_bin, y_proba, multi_class="ovr")
            metrics["roc_auc_ovo"] = roc_auc_score(y_bin, y_proba, multi_class="ovo")
        except Exception as e:
            self.logger.warning(f"ROC-AUC calculation failed: {str(e)}")

        if self.writer and step is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{prefix}/{name}", value, step)

            # Log confusion matrix
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=range(6), yticklabels=range(6))
            ax.set_title(f'{prefix} Confusion Matrix')
            self.writer.add_figure(f"{prefix}/confusion_matrix", fig, step)
            plt.close(fig)

        return metrics

    def train_model(self, use_cross_validation: bool = False) -> bool:
        """Training with enhanced multi-class support"""
        features, labels, _ = self.load_data()
        if features is None or labels is None:
            return False

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_log_dir = self.tensorboard_log_dir / f"run_{timestamp}"
            self.writer = SummaryWriter(log_dir=str(run_log_dir))

            # Log class distribution
            class_dist = np.bincount(labels)
            for cls_idx, count in enumerate(class_dist):
                self.writer.add_scalar("dataset/class_distribution", count, cls_idx)

            if use_cross_validation:
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                fold_metrics = []

                for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
                    self.logger.info(f"Training fold {fold + 1}/{self.n_folds}")

                    X_train, y_train = features[train_idx], labels[train_idx]
                    X_val, y_val = features[val_idx], labels[val_idx]

                    model = RandomForestClassifier(**self.model_params)
                    model.fit(X_train, y_train)

                    # Log fold metrics
                    metrics = self._evaluate(model, X_val, y_val, f"Fold_{fold + 1}", fold)
                    fold_metrics.append(metrics)

                    # Log feature importance for first fold
                    if fold == 0:
                        self._log_feature_importance(model)

                # Log average metrics
                avg_metrics = {
                    "accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
                    "f1_macro": np.mean([m["f1_macro"] for m in fold_metrics]),
                    "roc_auc_ovr": np.mean([m.get("roc_auc_ovr", 0) for m in fold_metrics])
                }
                for name, value in avg_metrics.items():
                    self.writer.add_scalar(f"CV_average/{name}", value, 0)

                # Final model
                self.model = RandomForestClassifier(**self.model_params)
                self.model.fit(features, labels)
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    features, labels,
                    test_size=self.test_size,
                    stratify=labels,
                    random_state=self.random_state
                )

                self.model = RandomForestClassifier(**self.model_params)
                self.model.fit(X_train, y_train)
                self._evaluate(self.model, X_val, y_val, "Validation", 0)

            # Save and log final model
            model_path = self._save_model(self.model)
            self._log_feature_importance(self.model)
            self.logger.info(f"Model saved to {model_path}")

            self.writer.close()
            self.print_tensorboard_instructions()
            return True

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            if self.writer:
                self.writer.close()
            return False

    def _log_feature_importance(self, model: ClassifierMixin):
        """Log feature importance visualization"""
        if hasattr(model, 'feature_importances_') and self.writer:
            importances = model.feature_importances_
            fig, ax = plt.subplots(figsize=(12, 8))
            indices = np.argsort(importances)[::-1][:20]

            ax.set_title("Top 20 Feature Importances")
            ax.bar(range(len(indices)), importances[indices], align='center')
            ax.set_xticks(range(len(indices)))
            ax.set_xticklabels(indices, rotation=45)
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("Importance")

            self.writer.add_figure("features/importance", fig)
            plt.close(fig)
    def print_tensorboard_instructions(self):
        """Print instructions for viewing TensorBoard logs"""
        self.logger.info("\nTo view TensorBoard logs, run the following command:")
        self.logger.info(f"tensorboard --logdir={self.tensorboard_log_dir}")
        self.logger.info("Then open http://localhost:6006 in your web browser")

    def evaluate_model(self):
        """Evaluate the trained model on the full dataset"""
        if not self.model:
            self.logger.error("No model has been trained yet")
            return

        features, labels, _ = self.load_data()
        if features is None or labels is None:
            return

        metrics = self._evaluate(self.model, features, labels, "Final Evaluation")
        self.logger.info("\nFinal Model Evaluation:")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Macro F1: {metrics['f1_macro']:.4f}")
        self.logger.info(f"ROC-AUC OVR: {metrics.get('roc_auc_ovr', 'N/A')}")
        self.logger.info("\nClassification Report:\n" + metrics['report'])

if __name__ == "__main__":
    trainer = ModelTrainer(
        model_params={
            "n_estimators": 500,  # 增加树的数量以提高性能
            "max_depth": 15,  # 可以稍微增加深度
            "min_samples_split": 5,  # 防止过拟合
            "min_samples_leaf": 2,  # 防止过拟合
            "max_features": "sqrt",  # 每个分裂考虑的特征数
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        },
        n_folds=5
    )

    if trainer.train_model(use_cross_validation=True):
        trainer.evaluate_model()