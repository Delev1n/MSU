import numpy as np
import pandas as pd
import math
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    fbeta_score,
)
import torch


def select_best_validation_threshold(fin_targets, fin_outputs):
    tpr = []
    fpr = []
    precision = []
    thresholds = np.arange(-0.01, 0.9, 0.06)
    positive_samples = sum([1 for targets in fin_targets if targets[0] == 1])
    for w in thresholds:
        outputs = torch.where(
            fin_outputs[:, 0] > w, torch.tensor(1.0), torch.tensor(0.0)
        )
        positive_outputs = list(outputs).count(1)
        if positive_outputs == 0:
            break
        tpr.append(
            sum(1 for yp, yt in zip(outputs, fin_targets) if yp == 1.0 and yt[0] == 1.0)
            / positive_samples
        )
        fpr.append(
            sum(1 for yp, yt in zip(outputs, fin_targets) if yp == 1.0 and yt[0] != 1.0)
            / (len(fin_targets) - positive_samples)
        )
        precision.append(
            sum(1 for yp, yt in zip(outputs, fin_targets) if yp == 1.0 and yt[0] == 1.0)
            / positive_outputs
        )
    gmeans = np.sqrt(tpr * (1 - np.array(fpr)))
    ix = np.argmax(gmeans)
    print("Best threshold: ", thresholds[ix])
    return thresholds[ix]


def metrics_report(
    targets: np.ndarray,
    bin_preds: np.ndarray,
    pathology_names: list,
) -> tuple:

    metrics, conf_matrix = compute_metrics(targets, bin_preds, pathology_names)
    print(metrics)
    print(conf_matrix)
    print(classification_report(targets, bin_preds, zero_division=False))


def compute_metrics(
    target,
    prediction,
    pathology_names,
):

    df = pd.DataFrame(
        columns=pathology_names,
        index=[
            "Specificity",
            "Sensitivity",
            "G-mean",
            "f1-score",
            "fbeta2-score",
            "ROC-AUC",
            "AP",
            "Precision (PPV)",
            "NPV",
        ],
    )
    conf_mat_df = pd.DataFrame(columns=["TN", "FP", "FN", "TP"], index=pathology_names)
    target = np.array(target, int)
    prediction = np.array(prediction, int)

    for i, col in enumerate(pathology_names):
        try:
            tn, fp, fn, tp = confusion_matrix(target[:, i], prediction[:, i]).ravel()
            df.loc["Specificity", col] = tn / (tn + fp)
            df.loc["Sensitivity", col] = tp / (tp + fn)
            df.loc["G-mean", col] = math.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
            df.loc["f1-score", col] = f1_score(target[:, i], prediction[:, i])
            df.loc["fbeta2-score", col] = fbeta_score(
                target[:, i], prediction[:, i], beta=2
            )
            df.loc["ROC-AUC", col] = roc_auc_score(target[:, i], prediction[:, i])
            df.loc["AP", col] = average_precision_score(target[:, i], prediction[:, i])
            df.loc["Precision (PPV)", col] = tp / (tp + fp)
            df.loc["NPV", col] = tn / (tn + fn)

            conf_mat_df.loc[col] = [tn, fp, fn, tp]
        except ValueError:
            raise
    return df, conf_mat_df


def calculate_metrics(fin_targets, fin_outputs):
    sigmoid = torch.nn.Sigmoid()

    fin_outputs = sigmoid(torch.as_tensor(fin_outputs))
    prediction_threshold = select_best_validation_threshold(fin_targets, fin_outputs)
    results = (fin_outputs > prediction_threshold).float()
    metrics_report(fin_targets, results.tolist(), ["AFIB"])
