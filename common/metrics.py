
import numpy as np
from sklearn import metrics


def sensitivity_specificity(conf_matrix):
    true_positives = conf_matrix[1, 1]
    false_negatives = conf_matrix[1, 0]
    true_negatives = conf_matrix[0, 0]
    false_positives = conf_matrix[0, 1]

    if true_positives + false_negatives == 0:
        true_positives += 1
    if true_negatives + false_positives == 0:
        true_negatives += 1

    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    return sensitivity, specificity


def select_threshold_by_sensitivity(y_true, y_prob, min_sens=0.90):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)

    # 满足 sensitivity 约束
    valid = tpr >= min_sens
    if not np.any(valid):
        raise ValueError(f"No threshold achieves sensitivity >= {min_sens}")

    idx_valid = np.where(valid)[0]

    # 在 valid 中选 FPR 最小的点（specificity 最大）
    best_idx = idx_valid[np.argmin(fpr[valid])]

    t_star = thresholds[best_idx]

    # 避免 inf 阈值（全判阴性）
    if np.isinf(t_star):
        cand = thresholds[idx_valid]
        cand = cand[~np.isinf(cand)]
        best_idx = idx_valid[np.argmax(cand)]
        t_star = thresholds[best_idx]

    fpr_star = fpr[best_idx]
    tpr_star = tpr[best_idx]

    return t_star, fpr_star, tpr_star


def select_threshold_by_balanced_accuracy(y_true, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)

    bacc = (tpr + (1 - fpr)) / 2
    best_idx = np.argmax(bacc)

    t_star = thresholds[best_idx]
    fpr_star = fpr[best_idx]
    tpr_star = tpr[best_idx]

    return t_star, fpr_star, tpr_star


def get_roc_point_from_threshold(y_true, y_prob, t_star):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)

    # 找到与 t_star 最接近的阈值索引
    idx = np.argmin(np.abs(thresholds - t_star))

    return fpr[idx], tpr[idx]


def compute_metrics(y_trues, y_probs, threshold=0.5, with_ci=False):
    y_pred = (y_probs >= threshold).astype(int)

    auc = metrics.roc_auc_score(y_trues, y_probs)
    bac = metrics.balanced_accuracy_score(y_trues, y_pred)
    conf_matrix = metrics.confusion_matrix(y_trues, y_pred)
    sen, spe = sensitivity_specificity(conf_matrix)

    tn, fp, fn, tp = conf_matrix.ravel()
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    if with_ci:
        ci = bootstrap_ci(y_trues, y_probs, n_bootstrap=1000, threshold=threshold)
        return {
            "auc": (auc, ci["auc"]['ci_lower'], ci["auc"]['ci_upper']),
            "bac": (bac, ci["bac"]['ci_lower'], ci["bac"]['ci_upper']),
            "sen": (sen, ci["sen"]['ci_lower'], ci["sen"]['ci_upper']),
            "spe": (spe, ci["spe"]['ci_lower'], ci["spe"]['ci_upper']),
            "ppv": (ppv, ci["ppv"]['ci_lower'], ci["ppv"]['ci_upper']),
            "npv": (npv, ci["npv"]['ci_lower'], ci["npv"]['ci_upper']),
        }

    return {"auc": auc, "bac": bac, "sen": sen, "spe": spe, "ppv": ppv, "npv": npv}


""" DCA (Decision Curve Analysis)
Decision Curve Analysis (DCA) is a method used to evaluate the clinical usefulness 
of diagnostic tests or prediction models. It calculates the net benefit of using 
a model to make decisions across a range of threshold probabilities (pt). 
The net benefit considers the trade-off between true positives and false positives, 
weighted by the threshold probability.
Usage:
    import matplotlib.pyplot as plt

    thresholds = np.linspace(0.01, 0.99, 100)
    
    nb_model = [net_benefit(y_true, y_prob, pt) for pt in thresholds]
    nb_all = [treat_all(y_true, pt) for pt in thresholds]
    nb_none = [0 for _ in thresholds]

    plt.plot(thresholds, nb_model, label='Model')
    plt.plot(thresholds, nb_all, linestyle='--', label='Treat All')
    plt.plot(thresholds, nb_none, linestyle=':', label='Treat None')

    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.legend()
    plt.show()
"""
def net_benefit(y_true, y_prob, pt):
    y_pred = (y_prob >= pt).astype(int)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    N = len(y_true)
    return TP / N - FP / N * (pt / (1 - pt))

def treat_all(y_true, pt):
    N = len(y_true)
    TP = np.sum(y_true == 1)
    FP = np.sum(y_true == 0)
    return TP / N - FP / N * (pt / (1 - pt))

def treat_none(y_true, pt):
    return 0


def bootstrap_ci(y_trues, y_probs, n_bootstrap=1000, threshold=0.5, random_state=42):
    # 返回每个指标的 (mean, lower, upper)
    rng = np.random.RandomState(random_state)
    n = len(y_trues)

    stats = {"auc": [], "bac": [], "sen": [], "spe": [], "ppv": [], "npv": []}

    for _ in range(n_bootstrap):
        # subject-level bootstrap
        indices = rng.choice(n, size=n, replace=True)

        y_true_b = y_trues[indices]
        y_probs_b = y_probs[indices]

        # AUC requires both classes present
        if len(np.unique(y_true_b)) < 2:
            continue

        m = compute_metrics(y_true_b, y_probs_b, threshold)

        for k in stats:
            stats[k].append(m[k])

    ci = {}
    for k, values in stats.items():
        values = np.array(values)
        ci[k] = {
            "mean": np.mean(values),
            "ci_lower": np.percentile(values, 2.5),
            "ci_upper": np.percentile(values, 97.5)
        }

    return ci


def bootstrap_diff_ci(y_trues, y_probs1, y_probs2, n_bootstrap=1000, threshold=0.5, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(y_trues)

    diffs = {"auc": [], "bac": [], "sen": [], "spe": [], "ppv": [], "npv": []}

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)

        y_true_b = y_trues[indices]
        p1 = y_probs1[indices]
        p2 = y_probs2[indices]

        if len(np.unique(y_true_b)) < 2:
            continue

        m1 = compute_metrics(y_true_b, p1, threshold)
        m2 = compute_metrics(y_true_b, p2, threshold)

        for k in diffs:
            diffs[k].append(m1[k] - m2[k])

    ci = {}
    for k, values in diffs.items():
        values = np.array(values)
        ci[k] = {
            "mean_diff": np.mean(values),
            "ci_lower": np.percentile(values, 2.5),
            "ci_upper": np.percentile(values, 97.5),
            "significant": not (
                np.percentile(values, 2.5) <= 0 <= np.percentile(values, 97.5)
            )
        }

    return ci


class FullMetrics:
    def __init__(self, n_classes, class_labels=None, n_bootstrap=1000, alpha=0.95):
        self.n_classes = n_classes
        self.class_labels = class_labels if class_labels is not None else [str(i) for i in range(n_classes)]
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha

    def compute(self, y_true, y_pred, y_probs):

        class_report = metrics.classification_report(y_true, y_pred, target_names=self.class_labels, output_dict=True)
        
        all_auroc = []
        all_auprc = []
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        precision = dict()
        recall = dict()
        average_precision = dict()
        
        for i in range(self.n_classes):
            class_label = self.class_labels[i]
            fpr[class_label], tpr[class_label], _ = metrics.roc_curve((y_true == i).astype(int), y_probs[:, i])
            roc_auc[class_label] = metrics.auc(fpr[class_label], tpr[class_label])
            
            precision[class_label], recall[class_label], _ = metrics.precision_recall_curve((y_true == i).astype(int), y_probs[:, i])
            average_precision[class_label] = metrics.average_precision_score((y_true == i).astype(int), y_probs[:, i])

        # Bootstrap for AUROC and AUPRC
        for i in range(self.n_classes):
            auroc_scores = []
            auprc_scores = []

            for _ in range(self.n_bootstrap):
                indices = np.random.choice(len(y_true), len(y_true), replace=True)
                auc_score = metrics.roc_auc_score((y_true[indices] == i).astype(int), y_probs[indices, i])
                auroc_scores.append(auc_score)

                prc_score = metrics.average_precision_score((y_true[indices] == i).astype(int), y_probs[indices, i])
                auprc_scores.append(prc_score)

            auroc_scores.sort()
            auprc_scores.sort()

            lower_idx = int((1 - self.alpha) / 2 * self.n_bootstrap)
            upper_idx = int((self.alpha + (1 - self.alpha) / 2) * self.n_bootstrap)

            lower_auroc = round(auroc_scores[lower_idx], 3)
            upper_auroc = round(auroc_scores[upper_idx], 3)

            lower_auprc = round(auprc_scores[lower_idx], 3)
            upper_auprc = round(auprc_scores[upper_idx], 3)

            all_auroc.append((lower_auroc, upper_auroc))
            all_auprc.append((lower_auprc, upper_auprc))

            # Assign scores to class_report
            auroc = round(roc_auc[self.class_labels[i]], 3)
            auprc = round(average_precision[self.class_labels[i]], 3)
            class_report[self.class_labels[i]]['auroc'] = f"{auroc} ({lower_auroc}, {upper_auroc})"
            class_report[self.class_labels[i]]['auprc'] = f"{auprc} ({lower_auprc}, {upper_auprc})"

        macro_auroc = np.mean([np.mean(ss) for ss in all_auroc])
        # macro_auroc_ci = np.percentile([np.mean(ss) for ss in all_auroc], [100 * (1 - alpha / 2), 100 * alpha / 2])

        macro_auprc = np.mean([np.mean(ss) for ss in all_auprc])
        # macro_auprc_ci = np.percentile([np.mean(ss) for ss in all_auprc], [100 * (1 - alpha / 2), 100 * alpha / 2])

        weighted_auroc = np.average([np.mean(ss) for ss in all_auroc], weights=[np.sum(y_true == i) for i in range(self.n_classes)])
        weighted_auprc = np.average([np.mean(ss) for ss in all_auprc], weights=[np.sum(y_true == i) for i in range(self.n_classes)])

        class_report['macro avg']['auroc'] = macro_auroc
        class_report['weighted avg']['auroc'] = weighted_auroc
        class_report['macro avg']['auprc'] = macro_auprc
        class_report['weighted avg']['auprc'] = weighted_auprc

        # print(f"Classification Report:\n {class_report}")

        return class_report
    

if __name__ == "__main__":

    y_trues = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    y_probs1 = np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
    y_probs2 = np.array([.2, .5, .1, .4, .9, .8, .7, .5, .9, .8])
    
    ci_model1 = bootstrap_ci(y_trues, y_probs1)
    ci_model2 = bootstrap_ci(y_trues, y_probs2)

    print("Model 1:")
    for k, v in ci_model1.items():
        print(f"{k}: {v['mean']:.3f} [{v['ci_lower']:.3f}, {v['ci_upper']:.3f}]")

    print("\nModel 2:")
    for k, v in ci_model2.items():
        print(f"{k}: {v['mean']:.3f} [{v['ci_lower']:.3f}, {v['ci_upper']:.3f}]")

    print("\nDifference (Model 1 - Model 2):")
    diff_ci = bootstrap_diff_ci(y_trues, y_probs1, y_probs2)

    for k, v in diff_ci.items():
        sig = "YES" if v["significant"] else "NO"
        print(
            f"{k}: Δ={v['mean_diff']:.3f} "
            f"[{v['ci_lower']:.3f}, {v['ci_upper']:.3f}] "
            f"significant={sig}"
        )
