"""Utilities for the project"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder


def plot_ks_statistic(
    y_true,
    y_probas,
    title="KS Statistic Plot",
    ax=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
):
    """Generates the KS Statistic plot from labels and scores/probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "KS Statistic Plot".

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(
            "Cannot calculate KS statistic for data with "
            f"{len(classes)} category/ies"
        )
    probas = y_probas

    # Compute KS Statistic curves
    thresholds, pct1, pct2, ks_statistic, max_distance_at, classes = (
        binary_ks_curve(y_true, probas[:, 1].ravel())
    )

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot(thresholds, pct1, lw=3, label=f"Class {classes[0]}")
    ax.plot(thresholds, pct2, lw=3, label=f"Class {classes[1]}")
    idx = np.where(thresholds == max_distance_at)[0][0]
    ax.axvline(
        max_distance_at,
        *sorted([pct1[idx], pct2[idx]]),
        label=f"KS Statistic: {ks_statistic:.3f} at {max_distance_at:.3f}",
        linestyle=":",
        lw=3,
        color="black",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.set_xlabel("Threshold", fontsize=text_fontsize)
    ax.set_ylabel("Percentage below threshold", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc="lower right", fontsize=text_fontsize)

    return ax


def binary_ks_curve(y_true, y_score):
    """This function generates the points necessary to calculate the KS
    Statistic curve.

    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.

        y_score (array-like, shape (n_samples)): Probability predictions of
            the positive class.

    Returns:
        thresholds (numpy.ndarray): An array containing the X-axis values for
            plotting the KS Statistic plot.

        pct1 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.

        pct2 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.

        ks_statistic (float): The KS Statistic, or the maximum vertical
            distance between the two curves.

        max_distance_at (float): The X-axis value at which the maximum vertical
            distance between the two curves is seen.

        classes (np.ndarray, shape (2)): An array containing the labels of the
            two classes making up `y_true`.

    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The KS Statistic
            is only relevant in binary classification.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    lb = LabelEncoder()
    encoded_labels = lb.fit_transform(y_true)
    if len(lb.classes_) != 2:
        raise ValueError(
            "Cannot calculate KS statistic for data with "
            f"{len(lb.classes_)} category/ies"
        )
    idx = encoded_labels == 0
    data1 = np.sort(y_score[idx])
    data2 = np.sort(y_score[np.logical_not(idx)])

    ctr1, ctr2 = 0, 0
    thresholds, pct1, pct2 = [], [], []
    while ctr1 < len(data1) or ctr2 < len(data2):
        # Check if data1 has no more elements
        if ctr1 >= len(data1):
            current = data2[ctr2]
            while ctr2 < len(data2) and current == data2[ctr2]:
                ctr2 += 1

        # Check if data2 has no more elements
        elif ctr2 >= len(data2):
            current = data1[ctr1]
            while ctr1 < len(data1) and current == data1[ctr1]:
                ctr1 += 1

        else:
            if data1[ctr1] > data2[ctr2]:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1

            elif data1[ctr1] < data2[ctr2]:
                current = data1[ctr1]
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

            else:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

        thresholds.append(current)
        pct1.append(ctr1)
        pct2.append(ctr2)

    thresholds = np.asarray(thresholds)
    pct1 = np.asarray(pct1) / float(len(data1))
    pct2 = np.asarray(pct2) / float(len(data2))

    if thresholds[0] != 0:
        thresholds = np.insert(thresholds, 0, [0.0])
        pct1 = np.insert(pct1, 0, [0.0])
        pct2 = np.insert(pct2, 0, [0.0])
    if thresholds[-1] != 1:
        thresholds = np.append(thresholds, [1.0])
        pct1 = np.append(pct1, [1.0])
        pct2 = np.append(pct2, [1.0])

    differences = pct1 - pct2
    ks_statistic, max_distance_at = (
        np.max(differences),
        thresholds[np.argmax(differences)],
    )

    return thresholds, pct1, pct2, ks_statistic, max_distance_at, lb.classes_


def business_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amount_tp: float,
    amount_tn: float = 0,
    amount_fp: float = 0,
    amount_fn: float = 0,
) -> float:
    """
    Calculate a custom business metric based on true positives, true negatives,
    false positives, and false negatives.

    Parameters:
    y_true (array-like): Ground truth (correct) target values.
    y_pred (array-like): Estimated targets as returned by a classifier.
    amount_tp (float): The amount associated with true positives.
    amount_tn (float, optional): The amount associated with true negatives.
    Default is 0.
    amount_fp (float, optional): The amount associated with false positives.
    Default is 0.
    amount_fn (float, optional): The amount associated with false negatives.
    Default is 0.

    Returns:
    float: The calculated business metric.
    """

    mask_true_positive = (y_true == 1) & (y_pred == 1)
    mask_true_negative = (y_true == 0) & (y_pred == 0)
    mask_false_positive = (y_true == 0) & (y_pred == 1)
    mask_false_negative = (y_true == 1) & (y_pred == 0)

    metric = (
        mask_true_positive.sum() * amount_tp
        + mask_true_negative.sum() * amount_tn
        + mask_false_positive.sum() * amount_fp
        + mask_false_negative.sum() * amount_fn
    )

    return metric


def compute_metrics(
    y_true: List[float], y_score: List[float], y_pred: List[float]
) -> Dict[str, float]:
    """
    Computes evaluation metrics based on true labels, target scores and
    predicted labels.

    Parameters:
    y_true (List[float]): True binary labels.
    y_score (List[float]): Target scores.
    y_pred (List[float]): Predicted binary labels.

    Returns:
    Dict[str, float]: A dictionary containing the evaluation metrics
    """

    return {
        "roc_score": roc_auc_score(y_true, y_score),
        "ks_statistic": binary_ks_curve(y_true, y_score)[3],
        "pr_score": average_precision_score(y_true, y_score),
        "accuracy_score": accuracy_score(y_true, y_pred),
        "balanced_accuracy_score": balanced_accuracy_score(y_true, y_pred),
        "precision_score": precision_score(y_true, y_pred, zero_division=0),
        "precision_score_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_score": recall_score(y_true, y_pred, zero_division=0),
        "recall_score_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "f1_score_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "business_metric": business_metric(
            y_true,
            y_pred,
            amount_tp=30,
            amount_tn=8,
            amount_fp=-12,
            amount_fn=-2
        ),
    }


def compute_roc_optimal_cutoff(
    y_true: List[float], y_score: List[float]
) -> float:
    """
    Compute the optimal cutoff threshold based on the Receiver Operating
    Characteristic (ROC) curve (Youden's index).
    """

    fpr, tpr, thresh = roc_curve(y_true, y_score)
    idx = np.arange(len(tpr))
    roc = pd.DataFrame(
        {
            "tf": pd.Series(tpr - (1 - fpr), index=idx),
            "threshold": pd.Series(thresh, index=idx),
        }
    )
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return float(roc_t["threshold"].values[0])


def compute_ks_optimal_cutoff(
    y_true: List[float], y_score: List[float]
) -> float:
    """
    Compute the optimal cutoff threshold based on the Kolmogorov-Smirnov
    (KS) statistic.
    """
    return binary_ks_curve(y_true, y_score)[4]


def compute_optimal_cutoff(
    y_true: List[float], y_score: List[float], method: str = "ks"
) -> float:
    """
    Compute the optimal cutoff threshold based on the specified method
    and test date.
    """
    if method == "roc":
        return compute_roc_optimal_cutoff(y_true, y_score)
    elif method == "ks":
        return compute_ks_optimal_cutoff(y_true, y_score)
    else:
        raise ValueError(f"Unsupported method: {method}")


def plot_ks(
    y_true: List[float],
    y_score: List[float],
    figsize: tuple = (10, 6),
    test_date: str = "N/D",
):
    """
    Plot the Kolmogorov-Smirnov (KS) statistic.
    """
    pred_scores = np.column_stack((1 - y_score, y_score))

    plot_ks_statistic(
        y_true,
        pred_scores,
        title=f"Estadístico KS - {test_date}",
        figsize=figsize,
    )
    plt.show()


def plot_feature_importance(model, top_n: int = 10, figsize: tuple = (10, 6)):
    """
    Plot the top feature importances.
    """

    try:
        if hasattr(model, "feature_names_in_"):
            feature_importances = pd.Series(
                model.feature_importances_ / model.feature_importances_.sum(),
                index=model.feature_names_in_,
            )
        elif hasattr(model, "feature_name_"):
            feature_importances = pd.Series(
                model.feature_importances_ / model.feature_importances_.sum(),
                index=model.feature_name_,
            )
        elif hasattr(model, "feature_names_"):
            feature_importances = pd.Series(
                model.feature_importances_ / model.feature_importances_.sum(),
                index=model.feature_names_,
            )
        else:
            feature_importances = pd.Series(
                model.feature_importances_ / model.feature_importances_.sum(),
                index=[str(i) for i in range(len(model.feature_importances_))],
            )
    except AttributeError:
        print(
            (
                "Cannot calculate feature importance. "
                "Is your model a decision tree object?"
            )
        )
    feature_importances = feature_importances.sort_values(ascending=False)
    top_features = feature_importances[:top_n]

    plt.figure(figsize=figsize)
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.title(f"Top {top_n} - Importancia de Variables")
    plt.show()


def plot_confusion_matrix(
    y_true: List[float],
    y_score: List[float],
    test_date="N/D",
    threshold=0.5,
    display_labels=None,
    figsize=(10, 6),
    normalize=None,
):
    """
    Plot the confusion matrix.
    """

    test_target = y_true
    pred_target = np.where(y_score > threshold, 1, 0)

    disp = ConfusionMatrixDisplay.from_predictions(
        test_target,
        pred_target,
        display_labels=display_labels,
        cmap="Blues",
        values_format=".0f" if normalize is None else None,
        normalize=normalize,
        colorbar=False,
    )
    disp.ax_.set_title(f"Matriz de Confusión - {test_date}")
    disp.figure_.set_size_inches(figsize)
    plt.show()


def plot_calibration_curve(
    y_true: List[float], y_score: List[float], test_date="N/D", figsize=(10, 6)
):
    """
    Plot the calibration curve.
    """

    plt.figure(figsize=figsize)
    disp = CalibrationDisplay.from_predictions(y_true, y_score)
    disp.ax_.set_title(f"Curva de Calibración - {test_date}")

    handles, labels = disp.ax_.get_legend_handles_labels()
    disp.ax_.legend(handles, labels, loc="best")

    plt.show()


def plot_shap_importance(
    model, train_data, top_n: int = 10, figsize: tuple = (10, 6)
):
    """
    Plot the top feature importances.
    """
    explainer = shap.TreeExplainer(
        model=model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    shap_values = explainer.shap_values(train_data)

    # Check if shap_values is a list (multi-output) or a single array 
    # (single-output)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[
            1
        ]  # Assuming binary classification, use the second class
    else:
        shap_values_to_plot = shap_values  # Single-output model

    shap.summary_plot(
        shap_values_to_plot,
        train_data,
        plot_type="violin",
        max_display=top_n,
        plot_size=figsize,
        show=False,
    )
    plt.title(f"Importancia SHAP - Top {top_n}", fontsize=16, y=1.05)
    plt.show()


def compute_ntile_table(
    y_true: List[float], y_score: List[float], quantiles=10
):
    """
    Compute ntile tables for the given inputs
    """

    predictions = pd.DataFrame(y_true)
    predictions.columns = ["TARGET"]
    predictions["SCORE"] = y_score

    segments = pd.qcut(predictions["SCORE"], q=quantiles)
    summary_table = (
        predictions.groupby(segments, observed=False)["TARGET"]
        .agg(
            [
                "count",
                "sum",
            ]
        )
        .rename(columns={"count": "TOTAL", "sum": "TARGET"})
        .sort_values("SCORE", ascending=False)
    )
    summary_table = summary_table.reset_index()

    score_thresholds = summary_table["SCORE"].apply(lambda x: x.left).tolist()

    total_vp = np.sum(predictions["TARGET"] == 1)
    total_vn = np.sum(predictions["TARGET"] == 0)

    confusion_matrices = []

    for threshold in score_thresholds:
        threshold_predictions = np.where(
            predictions["SCORE"] >= threshold, 1, 0
        )
        c_m = confusion_matrix(predictions["TARGET"], threshold_predictions)
        true_positive = c_m[1, 1]
        false_positive = c_m[0, 1]
        false_negative = c_m[1, 0]

        percent_true_positive = true_positive / total_vp
        percent_false_positive = false_positive / total_vn
        percent_false_negative = false_negative / total_vp
        confusion_matrices.append(
            [
                threshold,
                true_positive,
                percent_true_positive,
                false_positive,
                percent_false_positive,
                false_negative,
                percent_false_negative,
            ]
        )
    confusion_df = pd.DataFrame(
        confusion_matrices,
        columns=[
            "SCORE",
            "VERDADERO POSITIVO",
            "% VP",
            "FALSO POSITIVO",
            "% FP",
            "FALSO NEGATIVO",
            "% FN",
        ],
    )
    confusion_df.insert(1, "CANTIDAD REGISTROS", summary_table["TOTAL"])
    return confusion_df
