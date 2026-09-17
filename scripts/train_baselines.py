import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.config import PROJECT_ROOT as ROOT_DIR
from src.models import GCN_EEG
from src.utils import load_data_for_subjects, FocalLoss


# --- CLI Configuration ---
parser = argparse.ArgumentParser(description="Cross-validation grid search for static graph sparsification baselines.")
parser.add_argument("--kfold_cv", type=int, default=5, help="Number of folds for cross-validation")
parser.add_argument("--bs", type=int, default=256, help="Batch size")

opt = parser.parse_args()
print(opt)

kfold_cv = opt.kfold_cv
batch_size = opt.bs

# Hyperparameters
BATCH_SIZE = batch_size
LR = 3e-4
WEIGHT_DECAY = 1e-3
print(f"HPs: BATCH_SIZE={BATCH_SIZE}, LR={LR}, WEIGHT_DECAY={WEIGHT_DECAY}")


EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EARLY_STOPPING_PATIENCE = 5

LABEL_INTERICTAL = 0
LABEL_ICTAL = 1



def run_training(dataloader_args):
    """Trains and evaluates GCN_EEG on a given train/val/test patient split.

    Executes training with Focal Loss, evaluates across multiple classification
    metrics every epoch, applies early stopping on validation loss, and returns
    final test metrics on the held-out test cohort.

    Args:
        dataloader_args: Dictionary containing static pruning parameters
            passed down to `load_data_for_subjects` (e.g., threshold, quantile, or top_k).

    Returns:
        dict: Evaluation metrics on the test partition and best validation loss.
    """

    # Load PLV-based graphs with baseline scaling and self-loops
    train_data = load_data_for_subjects(SUBJECTS_TRAIN, only_plv=True, scaling=True, add_self_loops_flag=True, **dataloader_args)
    val_data = load_data_for_subjects(SUBJECTS_VAL, only_plv=True, scaling=True, add_self_loops_flag=True, **dataloader_args)
    test_data = load_data_for_subjects(SUBJECTS_TEST, only_plv=True, scaling=True, add_self_loops_flag=True, **dataloader_args)


    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


    num_features = train_data[0].num_node_features
    # Initialize model using static PLV edge weights ('raw' passthrough mode) - no automatic sparsification
    model = GCN_EEG(num_node_features=num_features, num_classes=2, only_plv=True, mode='raw').to(DEVICE)


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(gamma=2.0, alpha=None)

    best_val_loss = np.inf
    print("\nStarting training loop...")
    early_stopping_counter = 0

    for epoch in tqdm(range(EPOCHS)):
        # Training phase
        model.train()
        total_loss = 0
        for batch in train_loader:

            batch = batch.to(DEVICE)

            optimizer.zero_grad()
            out, edge_weight = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase
        model.eval()
        y_true_v, y_pred_v = [], []
        y_true_v, y_prob_v = [], []

        with torch.no_grad():
            val_loss_total = 0
            for batch in val_loader:
                batch = batch.to(DEVICE)

                out, edge_weight = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                y_pred_v.extend(out.argmax(dim=1).cpu().numpy())
                probs = F.softmax(out, dim=1).cpu().numpy()
                y_prob_v.append(probs.reshape(-1, probs.shape[-1]))
                y_true_v.append(batch.y.cpu().numpy().reshape(-1))
                val_loss = criterion(out, batch.y)
                val_loss_total += val_loss.item()

            
        y_true_v = np.concatenate(y_true_v)
        y_prob_v = np.concatenate(y_prob_v)


        y_true_det = y_true_v 
        y_score_det = y_prob_v[:, 1]

        # Metrics
        val_det_auroc = roc_auc_score(y_true_det, y_score_det)
        val_pr_auc = average_precision_score(y_true_det, y_score_det)
        val_balanced_acc = balanced_accuracy_score(y_true_v, y_pred_v)
        val_recall = recall_score(y_true_v, y_pred_v, pos_label=1, zero_division=0)
        val_precision = precision_score(y_true_v, y_pred_v, pos_label=1, zero_division=0)
        val_specificity = recall_score(y_true_v, y_pred_v, pos_label=0, zero_division=0)
        val_f1_ictal = f1_score(y_true_v, y_pred_v, pos_label=1, zero_division=0)


        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | "
        f"Val AUROC: {val_det_auroc:.4f} | Val PR-AUC: {val_pr_auc:.4f} | "
        f"Val F1: {val_f1_ictal:.4f} | Val Bal.Acc: {val_balanced_acc:.4f} | "
        f"Val Recall: {val_recall:.4f} | Val Precision: {val_precision:.4f} | "
        f"Val Specificity: {val_specificity:.4f} | Val Loss: {val_loss_total/len(val_loader):.6f}")

         
        val_loss = val_loss_total/len(val_loader) 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"No improvement for {EARLY_STOPPING_PATIENCE} epochs. Early stopping.")
                break


    # Test phase
    print("\n" + "=" * 30 + "\nTEST SET EVALUATION\n" + "=" * 30)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    model.eval()

    y_prob_t = []
    y_true_t, y_pred_t = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)

            out, edge_weight = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            y_pred_t.extend(out.argmax(dim=1).cpu().numpy())
            y_true_t.extend(batch.y.cpu().numpy().reshape(-1))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            y_prob_t.extend(probs)


    print(classification_report(y_true_t, y_pred_t, labels=[0, 1], 
                                target_names=['Non-Ictal', 'Ictal'], zero_division=0))

    y_true_t = np.array(y_true_t)
    y_prob_t = np.array(y_prob_t)

    # test AUROC
    y_true_det = y_true_t
    y_score_det = y_prob_t[:, 1] # Ictal class probability

    test_det_auroc = roc_auc_score(y_true_det, y_score_det)
    print(f"\nDetection AUROC (Non-Ictal vs Ictal): {test_det_auroc:.4f}")

    # Metrics
    test_pr_auc = average_precision_score(y_true_det, y_score_det)
    test_balanced_acc = balanced_accuracy_score(y_true_t, y_pred_t)
    test_recall = recall_score(y_true_t, y_pred_t, pos_label=1, zero_division=0)
    test_precision = precision_score(y_true_t, y_pred_t, pos_label=1, zero_division=0)
    test_specificity = recall_score(y_true_t, y_pred_t, pos_label=0, zero_division=0)

    print(f"Test PR-AUC: {test_pr_auc:.4f} | Test Balanced Acc: {test_balanced_acc:.4f} | "
      f"Test Recall: {test_recall:.4f} | Test Precision: {test_precision:.4f} | "
      f"Test Specificity: {test_specificity:.4f}")


    test_f1_macro = f1_score(y_true_t, y_pred_t, average="macro")
    test_f1_weighted = f1_score(y_true_t, y_pred_t, average="weighted")
    test_f1_ictal = f1_score(y_true_t, y_pred_t, pos_label=1, zero_division=0)
   
    return {
        "test_auroc": test_det_auroc,
        "test_pr_auc": test_pr_auc,
        "test_balanced_accuracy": test_balanced_acc,
        "test_recall": test_recall,
        "test_precision": test_precision,
        "test_specificity": test_specificity,
        "test_f1_ictal": test_f1_ictal,
        "test_f1_macro": test_f1_macro,
        "test_f1_weighted": test_f1_weighted,
        "best_val_loss": best_val_loss,
    }


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 24 subjects from the CHB-MIT EEG database
    patients = np.array([f"chb{str(i).zfill(2)}" for i in range(1, 25)])

    modes = ['threshold', 'quantile', 'top_k']

    configurations = {
        'threshold': [round(i * 0.1, 1) for i in range(10)],
        'quantile': [round(i * 0.1, 1) for i in range(10)],
        'top_k': list(range(1, 22))
    }

    # Grid search across all static pruning baselines
    for mode in modes:

        dataloader_args = {'threshold': None, 'quantile': None, 'top_k': None}
        output_csv = ROOT_DIR / f"results/baselines_{mode}.csv"
        grid = configurations[mode]

        for value in grid:

            dataloader_args[mode] = value
        
            print(
                f"\n=== RUNNING EXPERIMENT FOR {mode.upper()}: Value: {value} ==="
            )

            # K-fold for every configuration
            if kfold_cv:
                kf = KFold(n_splits=kfold_cv, shuffle=True, random_state=42)
                print(f"Starting {kfold_cv}-Fold CV across {len(patients)} patients:\n")

                # Outer CV loop: Split patients to avoid data leakage
                for fold, (train_val_idx, test_idx) in enumerate(kf.split(patients)):
                    
                    # Inner split for validation cohort
                    patients_test = patients[test_idx]
                    patients_train_val = patients[train_val_idx]
                    
                    patients_train, patients_val = train_test_split(
                        patients_train_val, 
                        test_size=3, 
                        random_state=21 + fold
                    )

                    print(f"=== FOLD {fold + 1} ===")
                    print(f"Training on ({len(patients_train)}):    {[str(i) for i in patients_train]}")
                    print(f"Validation on ({len(patients_val)}):  {[str(i) for i in patients_val]}")
                    print(f"Test on ({len(patients_test)}):       {[str(i) for i in patients_test]}")
                    print("-" * 50)

                    # Export current split globally for run_training
                    SUBJECTS_TRAIN = list(patients_train)
                    SUBJECTS_VAL = list(patients_val)
                    SUBJECTS_TEST = list(patients_test)

                    SAVE_MODEL_PATH = ROOT_DIR / "models" / f"best_model_{mode}_{value}_fold_{fold + 1}.pth"
                    metrics = run_training(dataloader_args)

                    # Append fold metrics to CSV
                    row_data = {
                        "grid_value": value,
                        "fold": fold + 1,
                        "test_auroc": metrics["test_auroc"],
                        "test_pr_auc": metrics["test_pr_auc"],
                        "test_balanced_accuracy": metrics["test_balanced_accuracy"],
                        "test_recall": metrics["test_recall"],
                        "test_precision": metrics["test_precision"],
                        "test_specificity": metrics["test_specificity"],
                        "test_f1_ictal": metrics["test_f1_ictal"],
                        "test_f1_macro": metrics["test_f1_macro"],
                        "test_f1_weighted": metrics["test_f1_weighted"],
                        "best_val_loss": metrics["best_val_loss"],
                    }

                    df_row = pd.DataFrame([row_data])

                    if not output_csv.exists():
                        df_row.to_csv(output_csv, index=False)
                    else:
                        df_row.to_csv(
                            output_csv, mode="a", index=False, header=False
                        )

                    print(f"Fold {fold + 1} results saved to {output_csv}")