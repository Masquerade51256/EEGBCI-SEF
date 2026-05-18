"""
Domain-Adversarial Neural Network (DANN) Streaming LOSO Trainer.

Extends StreamingLOSOTrainer with domain-adversarial training:
- Each training sample is tagged with a domain label (e.g., stroke location)
- The model learns domain-invariant features via a gradient reversal layer
  and domain discriminator.

Key components:
- Domain labels are loaded from participants.tsv based on a grouping strategy
- Training loss = task_loss + lambda * domain_loss
- Lambda follows Ganin et al.'s scheduling: increases from 0 to 1 during training
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trainers.streaming_loso_trainer import StreamingLOSOTrainer
from utils.domain_info import get_domain_manager


class DANNStreamingLOSOTrainer(StreamingLOSOTrainer):
    """
    DANN-enabled streaming LOSO trainer.

    Additional config keys under trainer.args:
        - domain_grouping: str, domain strategy (default: 'stroke_location_4class')
        - grl_gamma: float, GRL scheduling parameter (default: 10.0)
        - domain_loss_weight: float, weight for domain loss term (default: 1.0)
    """

    def __init__(
        self,
        model=None,
        config=None,
        device=None,
        paths=None,
        logger=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            config=config,
            device=device,
            paths=paths,
            logger=logger,
            **kwargs,
        )

        # DANN-specific hyperparameters
        self.domain_grouping = kwargs.get(
            "domain_grouping",
            self.config.get("trainer.args.domain_grouping", "stroke_location_4class"),
        )
        self.grl_gamma = kwargs.get(
            "grl_gamma", self.config.get("trainer.args.grl_gamma", 10.0)
        )
        self.domain_loss_weight = kwargs.get(
            "domain_loss_weight",
            self.config.get("trainer.args.domain_loss_weight", 1.0),
        )

        # Domain manager will be initialized per experiment in train()
        self.domain_manager = None
        self.num_domains = None

        # Separate loss functions
        self.task_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

        self._log("=" * 60)
        self._log("DANN Streaming LOSO Trainer initialized")
        self._log(
            f"Domain grouping: {self.domain_grouping}, "
            f"GRL gamma: {self.grl_gamma}, "
            f"Domain loss weight: {self.domain_loss_weight}"
        )
        self._log("=" * 60)

    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run DANN streaming LOSO cross-validation.

        Initializes domain manager from dataset info before starting LOSO loops.
        """
        dataset_info = datasets.get("info", {})

        # Initialize domain manager
        try:
            self.domain_manager = get_domain_manager(
                dataset_info, strategy=self.domain_grouping
            )
            self.num_domains = self.domain_manager.num_domains
            self._log(f"Domain manager loaded: {self.domain_manager}")
        except Exception as e:
            self._log(
                f"Warning: Could not load domain manager: {e}. "
                f"Falling back to single-domain training.",
                level="warning",
            )
            self.domain_manager = None
            self.num_domains = 1

        # Delegate to parent LOSO loop
        return super().train(datasets)

    def _build_cached_train_loader(
        self,
        train_subject_ids: List[int],
        dataset_cls,
        dataset_info: Dict,
    ) -> DataLoader:
        """
        Preload training subjects and attach domain labels.

        Overrides parent to concatenate domain labels alongside data and labels.
        """
        import time

        start = time.time()

        all_data = []
        all_labels = []
        all_domains = []
        augmentor = None

        for i, subject_id in enumerate(train_subject_ids):
            ds = dataset_cls(subject_id=subject_id, dataset_info=dataset_info)
            all_data.append(ds.data)
            all_labels.append(ds.labels)

            # Domain label for this subject
            if self.domain_manager is not None:
                domain_id = self.domain_manager.get_domain_id(subject_id)
            else:
                domain_id = 0
            all_domains.append(np.full(len(ds), domain_id, dtype=np.int64))

            if i == 0 and hasattr(ds, "augmentor"):
                augmentor = ds.augmentor

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_domains = np.concatenate(all_domains, axis=0)

        class _InMemoryDatasetWithDomain(Dataset):
            def __init__(self, data, labels, domains, augmentor=None):
                self.data = data
                self.labels = labels
                self.domains = domains
                self.augmentor = augmentor

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                data = self.data[idx]
                label = self.labels[idx]
                domain = self.domains[idx]
                if self.augmentor is not None:
                    data = self.augmentor.process(data)
                return (
                    torch.from_numpy(data).float(),
                    torch.tensor(label).long(),
                    torch.tensor(domain).long(),
                )

        train_dataset = _InMemoryDatasetWithDomain(
            all_data, all_labels, all_domains, augmentor
        )

        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "drop_last": False,
        }

        if self.device.type == "cuda":
            loader_kwargs["pin_memory"] = self.config.get("training.pin_memory", True)
            num_workers = self.config.get("training.num_workers", 0)
            if num_workers > 0:
                loader_kwargs["num_workers"] = num_workers
                loader_kwargs["persistent_workers"] = True
                loader_kwargs["prefetch_factor"] = self.config.get(
                    "training.prefetch_factor", 2
                )

        train_loader = DataLoader(train_dataset, **loader_kwargs)

        elapsed = time.time() - start
        self._log(
            f"  Preloaded {len(train_subject_ids)} subjects "
            f"({len(train_dataset)} samples, {self.num_domains} domains) in {elapsed:.1f}s"
        )

        return train_loader

    def _compute_grl_lambda(self, epoch: int) -> float:
        """
        Compute adaptive lambda for gradient reversal layer.

        Following Ganin et al. (2016):
            lambda_p = 2 / (1 + exp(-gamma * p)) - 1
        where p is the training progress (0 -> 1).
        """
        p = float(epoch) / float(max(1, self.epochs - 1))
        lambda_p = 2.0 / (1.0 + math.exp(-self.grl_gamma * p)) - 1.0
        return lambda_p

    def _train_epoch_cached(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
    ) -> Tuple[float, float]:
        """
        Train one epoch with domain-adversarial loss.

        Returns:
            (combined_loss, accuracy)
        """
        model.train()

        total_task_loss = 0.0
        total_domain_loss = 0.0
        total_combined_loss = 0.0
        correct = 0
        total = 0

        accumulation_counter = 0
        use_acc = self.gradient_accumulation_steps > 1

        # Adaptive lambda based on current epoch
        lambda_p = self._compute_grl_lambda(self.current_epoch)

        for batch in train_loader:
            inputs, labels, domains = batch
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            domains = domains.to(self.device, non_blocking=True)

            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    class_output, domain_output = model(inputs, alpha=lambda_p)
                    task_loss = self.task_criterion(class_output, labels)
                    domain_loss = self.domain_criterion(domain_output, domains)
                    combined_loss = task_loss + self.domain_loss_weight * domain_loss
                    if use_acc:
                        combined_loss = combined_loss / self.gradient_accumulation_steps

                self.scaler.scale(combined_loss).backward()
            else:
                class_output, domain_output = model(inputs, alpha=lambda_p)
                task_loss = self.task_criterion(class_output, labels)
                domain_loss = self.domain_criterion(domain_output, domains)
                combined_loss = task_loss + self.domain_loss_weight * domain_loss
                if use_acc:
                    combined_loss = combined_loss / self.gradient_accumulation_steps
                combined_loss.backward()

            # Gradient accumulation
            if use_acc:
                accumulation_counter += 1
                if accumulation_counter % self.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
            else:
                if self.scaler:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # Statistics (undo accumulation division for logging)
            scale = self.gradient_accumulation_steps if use_acc else 1
            total_task_loss += task_loss.item() * inputs.size(0) * scale
            total_domain_loss += domain_loss.item() * inputs.size(0) * scale
            total_combined_loss += combined_loss.item() * inputs.size(0) * scale

            _, predicted = torch.max(class_output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Handle remaining gradients
        if use_acc and accumulation_counter % self.gradient_accumulation_steps != 0:
            if self.scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        avg_task_loss = total_task_loss / total if total > 0 else 0.0
        avg_domain_loss = total_domain_loss / total if total > 0 else 0.0
        avg_combined_loss = total_combined_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        # Store detailed metrics for logging (accessible by postfix in parent loop)
        self._last_task_loss = avg_task_loss
        self._last_domain_loss = avg_domain_loss
        self._last_lambda = lambda_p

        return avg_combined_loss, accuracy

    def _train_loso_round_streaming(
        self,
        test_subject_id: int,
        train_subject_ids: List[int],
        dataset_cls,
        dataset_info: Dict,
    ) -> Tuple[Dict, Dict]:
        """
        Train one LOSO round with DANN.

        Overrides parent to inject domain-loss-aware logging into the postfix.
        The core training logic (early stopping, model selection) remains identical.
        """
        # --- Copy early-stopping and setup logic from parent ---
        es_patience = self.config.get("trainer.args.early_stopping_patience", 15)
        use_val_es = self.config.get("trainer.args.use_validation_early_stopping", False)

        val_subject_id = None
        train_subject_ids_inner = train_subject_ids.copy()
        val_loader = None

        if use_val_es and len(train_subject_ids) >= 2:
            import random

            seed = self.config.get("experiment.seed", 42)
            rng = random.Random(test_subject_id + seed)
            val_subject_id = rng.choice(train_subject_ids)
            train_subject_ids_inner = [
                s for s in train_subject_ids if s != val_subject_id
            ]

            val_dataset = dataset_cls(
                subject_id=val_subject_id, dataset_info=dataset_info
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
            self._log(f"  Val subject: {val_subject_id} (ES patience={es_patience})")
        else:
            self._log(f"  No val split (train={len(train_subject_ids)} subjects)")

        # Create model
        sample_ds = dataset_cls(
            subject_id=train_subject_ids_inner[0], dataset_info=dataset_info
        )
        model = self._create_model(sample_ds)
        del sample_ds

        # Create test dataset
        test_dataset = dataset_cls(
            subject_id=test_subject_id, dataset_info=dataset_info
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # History
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": [],
            "test_acc": [],
            "task_loss": [],
            "domain_loss": [],
            "grl_lambda": [],
        }

        best_val_acc = -1.0
        best_val_epoch = 0
        best_val_state = None
        epochs_no_improve = 0
        stopped_early = False

        # Preload training data (with domain labels)
        train_loader = self._build_cached_train_loader(
            train_subject_ids_inner, dataset_cls, dataset_info
        )

        # Progress bar
        desc = f"Sub{test_subject_id}"
        if val_subject_id:
            desc += f"(Val{val_subject_id})"
        epoch_bar = tqdm(
            range(self.epochs),
            desc=desc,
            bar_format="{desc} |{bar:20}| {n_fmt}/{total_fmt} {postfix}",
        )

        for epoch in epoch_bar:
            self.current_epoch = epoch

            # Update learning rate
            lr = self._compute_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Train
            train_loss, train_acc = self._train_epoch_cached(
                model, optimizer, train_loader
            )
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["task_loss"].append(getattr(self, "_last_task_loss", 0.0))
            history["domain_loss"].append(getattr(self, "_last_domain_loss", 0.0))
            history["grl_lambda"].append(getattr(self, "_last_lambda", 0.0))

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._validate(model, val_loader)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
            else:
                val_loss, val_acc = float("nan"), float("nan")
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            # Test (monitoring only)
            test_loss, test_acc = self._validate(model, test_loader)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            # Postfix with domain-adversarial info
            postfix = f"LR:{lr:.1e} Tr:{train_loss:.3f}/{train_acc:.3f}"
            if hasattr(self, "_last_task_loss"):
                postfix += f" Task:{self._last_task_loss:.3f} Dom:{self._last_domain_loss:.3f}"
            if val_loader:
                postfix += f" Va:{val_loss:.3f}/{val_acc:.3f}"
            postfix += f" Te:{test_loss:.3f}/{test_acc:.3f}"
            epoch_bar.set_postfix_str(postfix)

            # Model selection based on validation accuracy
            if val_loader is not None:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_epoch = epoch
                    epochs_no_improve = 0
                    best_val_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= es_patience:
                    self._log(
                        f"  Early stopping @ epoch {epoch} (best val @ {best_val_epoch})"
                    )
                    stopped_early = True
                    break

            self.global_step += 1

        epoch_bar.close()

        # Final evaluation
        if val_loader is not None and best_val_state is not None:
            model.load_state_dict(best_val_state)
            self._log(
                f"  Loaded best val model (epoch {best_val_epoch}, val_acc={best_val_acc:.4f})"
            )

        final_loss, final_acc = self._validate(model, test_loader)

        final_val_loss, final_val_acc = float("nan"), float("nan")
        if val_loader is not None:
            final_val_loss, final_val_acc = self._validate(model, val_loader)

        return {
            "test_subject_id": test_subject_id,
            "test_acc": final_acc,
            "test_loss": final_loss,
            "val_acc": final_val_acc,
            "val_loss": final_val_loss,
            "best_val_acc": best_val_acc,
            "best_val_epoch": best_val_epoch,
            "stopped_early": stopped_early,
            "actual_epochs": epoch + 1 if stopped_early else self.epochs,
            "train_subjects": len(train_subject_ids_inner),
            "val_subject": val_subject_id,
            "test_samples": len(test_dataset),
        }, history
