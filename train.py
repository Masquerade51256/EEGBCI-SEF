import os
import math
import logging
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import config
from Dataloader import dataloader
from Models.get_model import get_model

# ==================== Configuration & Setup ====================
train_device = torch.device(config.train_device if torch.cuda.is_available() else "cpu")
print(f"Using device: {train_device}")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

DATASET_NAME = config.DATASETS[config.SELECTED_DATASET]
MODEL_NAME = config.MODELS[config.SELECTED_MODEL]

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log", encoding='utf-8'),
        # logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# ===============================================================

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """Trains the model for a single epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(data_loader, desc='Training', leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Update progress bar description
        progress_bar.set_postfix({'Loss': loss.item()})

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy

def validate(model, data_loader, criterion, device):
    """Validates the model."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy

def plot_training_history(training_history, save_path):
    """
    Generates and saves visualization plots for the training process.
    Args:
        training_history: A list of dictionaries, one per fold, containing
                         'train_loss', 'val_loss', 'train_acc', 'val_acc' lists.
        save_path: Directory where the plot image will be saved.
    """
    if not training_history:
        logger.warning("Training history is empty. Skipping plot generation.")
        return

    num_folds = len(training_history)
    fig, axes = plt.subplots(2, num_folds, figsize=(5*num_folds, 8), squeeze=False)
    fig.suptitle('Training History per Fold', fontsize=16)

    for fold_idx, fold_history in enumerate(training_history):
        epochs = range(1, len(fold_history['train_loss']) + 1)

        # Plot Loss
        ax_loss = axes[0, fold_idx]
        ax_loss.plot(epochs, fold_history['train_loss'], 'b-', label='Train Loss')
        ax_loss.plot(epochs, fold_history['val_loss'], 'r-', label='Val Loss')
        ax_loss.set_title(f'Fold {fold_idx}')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # Plot Accuracy
        ax_acc = axes[1, fold_idx]
        ax_acc.plot(epochs, fold_history['train_acc'], 'b-', label='Train Acc')
        ax_acc.plot(epochs, fold_history['val_acc'], 'r-', label='Val Acc')
        ax_acc.set_title(f'Fold {fold_idx}')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plot_filename = os.path.join(save_path, 'training_history_per_fold.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close(fig)
    logger.info(f"Training history plot saved to: {plot_filename}")

def plot_subject_comparison(subject_results, save_path):
    """
    Generates and saves a bar plot comparing final metrics across all subjects.
    Args:
        subject_results: A list of dictionaries, one per subject, containing
                        'subject_id' and 'final_val_acc'.
        save_path: Directory where the plot image will be saved.
    """
    if not subject_results:
        logger.warning("No subject results for comparison plot.")
        return

    subject_ids = [res['subject_id'] for res in subject_results]
    avg_accuracies = [res['avg_val_acc'] for res in subject_results]

    fig, ax = plt.subplots(figsize=(max(6, len(subject_ids)*0.5), 5))
    bars = ax.bar(subject_ids, avg_accuracies, color='skyblue', edgecolor='black')
    ax.set_xlabel('Subject ID')
    ax.set_ylabel('Average Validation Accuracy')
    ax.set_title('Final Model Performance by Subject')
    ax.set_ylim([0, 1.05])
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on top of bars
    for bar, acc in zip(bars, avg_accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plot_filename = os.path.join(save_path, 'subject_performance_comparison.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close(fig)
    logger.info(f"Subject comparison plot saved to: {plot_filename}")

def train_model(model, dataset, subject_id=None):
    """
    Main training routine for a single subject using k-fold cross-validation.
    Returns the trained model and a history dictionary for visualization.
    """
    weight_decay = config.weight_decay
    initial_learning_rate = config.learning_rate
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    num_folds = config.k_folds

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=config.seed if hasattr(config, 'seed') else 0)
    logger.info(f"Starting training for Subject {subject_id} with {num_folds}-fold CV.")

    # Data structure to store history for plotting
    fold_training_histories = []
    fold_best_accuracies = []

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        logger.info(f"--- Processing Fold {fold_idx + 1}/{num_folds} ---")
        # Create data subsets and loaders for this fold
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                   generator=torch.Generator(device=train_device))
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                 generator=torch.Generator(device=train_device))

        # Re-initialize model and optimizer for each fold
        fold_model = get_model(config.SELECTED_MODEL).to(train_device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(fold_model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)

        # History for this specific fold
        fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        fold_best_val_accuracy = 0.0

        epoch_progress_bar = tqdm(range(num_epochs), desc=f'Fold {fold_idx+1} Epochs')
        for epoch in epoch_progress_bar:
            # Cosine annealing learning rate schedule
            current_lr = (1 + math.cos(epoch * math.pi / num_epochs)) * initial_learning_rate / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Train phase
            train_loss, train_acc = train_one_epoch(fold_model, train_loader, criterion, optimizer, train_device)
            # Validation phase
            val_loss, val_acc = validate(fold_model, val_loader, criterion, train_device)

            # Record history
            fold_history['train_loss'].append(train_loss)
            fold_history['train_acc'].append(train_acc)
            fold_history['val_loss'].append(val_loss)
            fold_history['val_acc'].append(val_acc)

            # Update progress bar
            epoch_progress_bar.set_postfix({
                'LR': f'{current_lr:.2e}',
                'Train_L': f'{train_loss:.3f}',
                'Val_L': f'{val_loss:.3f}',
                'Val_Acc': f'{val_acc:.3f}'
            })
            # Log detailed info
            logger.info(f"Sub{subject_id}_Fold{fold_idx}_E{epoch:03d}: LR={current_lr:.6f}, "
                       f"Train=[L:{train_loss:.4f}, A:{train_acc:.4f}], "
                       f"Val=[L:{val_loss:.4f}, A:{val_acc:.4f}]")

            # Save best model for the fold
            if val_acc > fold_best_val_accuracy:
                fold_best_val_accuracy = val_acc
                # Save checkpoint
                model_to_save = fold_model.module if hasattr(fold_model, 'module') else fold_model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'fold': fold_idx,
                    'subject_id': subject_id
                }
                checkpoint_dir = os.path.join(config.ckpt_path, DATASET_NAME, MODEL_NAME, str(subject_id))
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_name = f"best_fold{fold_idx}_acc{fold_best_val_accuracy:.4f}.pt"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"  -> Saved new best model for fold {fold_idx} to: {checkpoint_path}")

        fold_training_histories.append(fold_history)
        fold_best_accuracies.append(fold_best_val_accuracy)
        logger.info(f"--- Fold {fold_idx} finished. Best Val Acc: {fold_best_val_accuracy:.4f} ---")
        epoch_progress_bar.close()

    subject_avg_accuracy = sum(fold_best_accuracies) / len(fold_best_accuracies)
    logger.info(f"===== Subject {subject_id} training complete. Best Accuracy: {subject_avg_accuracy:.4f} =====")
    # Note: The function returns the model from the *last* fold.
    # A more robust implementation might return an ensemble or the single best model.
    return fold_model, fold_training_histories, subject_avg_accuracy

def main():
    """Main execution function."""
    all_subjects_models = []
    all_subjects_history = []
    all_subjects_results = []  # For final comparison plot

    output_viz_dir = os.path.join("results", "visualizations", DATASET_NAME, MODEL_NAME)
    os.makedirs(output_viz_dir, exist_ok=True)

    for subject_id in config.target_subjects:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Subject: {subject_id}")
        logger.info(f"{'='*60}")

        # Load single subject data
        subject_dataset = dataloader.load_single_subject_data(config.SELECTED_DATASET, subject_id)
        # Build model
        model_instance = get_model(config.SELECTED_MODEL).to(train_device)
        # Train model
        trained_model, subject_history, subject_avg_acc = train_model(model_instance, subject_dataset, subject_id)

        all_subjects_models.append(trained_model)
        all_subjects_history.append({'subject_id': subject_id, 'history': subject_history})
        all_subjects_results.append({'subject_id': subject_id, 'avg_val_acc': subject_avg_acc})

        # Generate and save training history plot for this subject
        plot_training_history(subject_history, os.path.join(output_viz_dir, f"subject_{subject_id}"))

        # Evaluate model (Placeholder for your evaluation logic)
        # evaluation_results = evaluate_model(trained_model, test_data)
        # Visualize results (Placeholder for other specific visualizations)
        # visualize_specific_results(trained_model, subject_id)

    # After all subjects are processed, generate a summary comparison plot
    plot_subject_comparison(all_subjects_results, output_viz_dir)
    logger.info(f"\nAll subjects processed. Visualizations saved in: {output_viz_dir}")

if __name__ == "__main__":
    main()