"""
=============================================================
MODULE 7: EVALUATION
Project: Tomato Leaf Disease Detection using CNN from Scratch
Tools: Pure Python + NumPy only
=============================================================
"""

import numpy as np
import os
from module1_image_loader import load_image, resize_image, TOMATO_CLASSES
from module2_preprocessing import preprocess, get_batches
from module4_forward_pass import TomatoCNN
from module8_rejection import TomatoRejector


# ─────────────────────────────────────────────
# 1. CONFUSION MATRIX
# ─────────────────────────────────────────────

def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def print_confusion_matrix(cm, class_names):
    print("\n[Confusion Matrix]  Rows=True | Cols=Predicted\n")
    short = [c.replace("Tomato__","").replace("Tomato_","")[:10] for c in class_names]
    header = f"{'':15}" + "".join(f"{s:12}" for s in short)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        print(f"{short[i]:15}" + "".join(f"{v:12}" for v in row))


# ─────────────────────────────────────────────
# 2. PRECISION / RECALL / F1
# ─────────────────────────────────────────────

def compute_metrics(cm, class_names):
    metrics = {}
    for i, cls in enumerate(class_names):
        tp = cm[i][i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precision = tp / (tp + fp + 1e-7)
        recall    = tp / (tp + fn + 1e-7)
        f1        = 2 * precision * recall / (precision + recall + 1e-7)
        metrics[cls] = {
            'precision': round(float(precision), 4),
            'recall'   : round(float(recall),    4),
            'f1'       : round(float(f1),        4),
            'support'  : int(np.sum(cm[i, :]))
        }
    return metrics


def print_metrics(metrics):
    print(f"\n{'Class':45} {'Precision':10} {'Recall':10} {'F1':10} {'Support':8}")
    print("-" * 88)
    ps, rs, fs = [], [], []
    for cls, m in metrics.items():
        name = cls.replace("Tomato__","").replace("Tomato_","")
        print(f"  {name:43} {m['precision']:10.4f} {m['recall']:10.4f} {m['f1']:10.4f} {m['support']:8}")
        ps.append(m['precision']); rs.append(m['recall']); fs.append(m['f1'])
    print("-" * 88)
    print(f"  {'AVERAGE':43} {np.mean(ps):10.4f} {np.mean(rs):10.4f} {np.mean(fs):10.4f}")


# ─────────────────────────────────────────────
# 3. EVALUATE ON TEST SET
# ─────────────────────────────────────────────

def evaluate_test_set(model, test_X, test_y, class_names, batch_size=32):
    model.set_training(False)
    all_preds, all_trues = [], []
    total_loss, num_batches = 0.0, 0

    print("\n[Evaluation] Running on test set...")
    for batch_X, batch_y in get_batches(test_X, test_y, batch_size):
        probs = model.forward(batch_X)
        total_loss  += model.compute_loss(probs, batch_y)
        all_preds.extend(np.argmax(probs,   axis=1).tolist())
        all_trues.extend(np.argmax(batch_y, axis=1).tolist())
        num_batches += 1

    model.set_training(True)
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    accuracy  = float(np.mean(all_preds == all_trues))
    cm        = compute_confusion_matrix(all_trues, all_preds, len(class_names))
    metrics   = compute_metrics(cm, class_names)

    return {
        'accuracy' : accuracy,
        'loss'     : total_loss / num_batches,
        'cm'       : cm,
        'metrics'  : metrics,
        'all_preds': all_preds,
        'all_trues': all_trues
    }


# ─────────────────────────────────────────────
# 4. PREDICT SINGLE IMAGE (WITH REJECTION)
# ─────────────────────────────────────────────

def predict_single_image(model, image_path, class_names, image_size=(64, 64), rejector=None):
    """
    Predict with full rejection pipeline.
    Returns (predicted_class, confidence, status)
    status: "ACCEPTED" | "REJECTED_COLOR" | "REJECTED_CONFIDENCE" | "REJECTED_MODEL" | "ERROR"
    """
    if rejector is None:
        rejector = TomatoRejector(confidence_threshold=0.75)

    # ── Load ──
    try:
        img = load_image(image_path)
        img = resize_image(img, image_size)
        img = img.astype(np.float32) / 255.0
        img = img[np.newaxis, :]
    except Exception as e:
        print(f"\n  [ERROR] Cannot load image: {e}")
        return None, 0.0, "ERROR"

    # ── Layer 1: Color/texture prefilter ──
    is_plausible, reason, green_ratio = rejector.prefilter(img)
    if not is_plausible:
        print(f"\n  ❌ REJECTED (color filter): {reason}")
        print(rejector.get_rejection_message(reason, green_ratio))
        return None, 0.0, "REJECTED_COLOR"

    # ── Layer 2: Model prediction ──
    model.set_training(False)
    probs      = model.forward(img)
    model.set_training(True)
    class_id   = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs,  axis=1)[0])
    predicted  = class_names[class_id]

    # ── Layer 3: Model predicted non_tomato class ──
    if predicted == "non_tomato":
        print(f"\n  ❌ REJECTED (model): Classified as non_tomato ({confidence*100:.1f}%)")
        print(rejector.get_rejection_message("Model identified this as not a tomato leaf"))
        return None, confidence, "REJECTED_MODEL"

    # ── Layer 4: Confidence threshold ──
    should_reject, reject_reason = rejector.should_reject_by_confidence(confidence, predicted)
    if should_reject:
        print(f"\n  ❌ REJECTED (confidence): {reject_reason}")
        print(rejector.get_rejection_message(reject_reason, confidence=confidence))
        return None, confidence, "REJECTED_CONFIDENCE"

    # ── Accepted ──
    short = predicted.replace("Tomato__","").replace("Tomato_","").replace("_"," ")
    print(f"\n  ✅ ACCEPTED")
    print(f"  Disease    : {short}")
    print(f"  Confidence : {confidence*100:.2f}%")

    if confidence < 0.80:
        print(rejector.get_warning_message(confidence, predicted))

    return predicted, confidence, "ACCEPTED"


# ─────────────────────────────────────────────
# 5. SAVE REPORT
# ─────────────────────────────────────────────

def save_report(results, class_names, filepath="saved_weights/evaluation_report.csv"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("OVERALL METRICS\n")
        f.write(f"Test Accuracy,{results['accuracy']*100:.4f}%\n")
        f.write(f"Test Loss,{results['loss']:.6f}\n\n")
        f.write("PER CLASS METRICS\n")
        f.write("Class,Precision,Recall,F1,Support\n")
        for cls, m in results['metrics'].items():
            f.write(f"{cls},{m['precision']},{m['recall']},{m['f1']},{m['support']}\n")
        f.write("\nCONFUSION MATRIX\n")
        f.write("," + ",".join(class_names) + "\n")
        for i, row in enumerate(results['cm']):
            f.write(class_names[i] + "," + ",".join(map(str, row)) + "\n")
    print(f"\n[Report] Saved → {filepath}")


# ─────────────────────────────────────────────
# 6. FULL EVALUATION PIPELINE
# ─────────────────────────────────────────────

def evaluate(
    dataset_dir  = "dataset",
    weights_path = "saved_weights/best_model.npz",
    image_size   = (64, 64),
    max_per_class = None,
    batch_size   = 32
):
    print("=" * 55)
    print("  TOMATO CNN — EVALUATION")
    print("=" * 55)

    data = preprocess(dataset_dir=dataset_dir, image_size=image_size,
                      max_per_class=max_per_class)

    model = TomatoCNN(num_classes=data['num_classes'])
    model.load_weights(weights_path)

    results = evaluate_test_set(
        model, data['test_X'], data['test_y'], data['classes'], batch_size
    )

    print(f"\n{'='*55}")
    print(f"  TEST ACCURACY : {results['accuracy']*100:.2f}%")
    print(f"  TEST LOSS     : {results['loss']:.4f}")
    print(f"{'='*55}")

    print_confusion_matrix(results['cm'], data['classes'])
    print_metrics(results['metrics'])
    save_report(results, data['classes'])

    print("\n[Evaluation] Complete ✓")
    return model, results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        if len(sys.argv) < 3:
            print("Usage: python module7_evaluation.py predict <image_path>")
        else:
            model = TomatoCNN(num_classes=11)
            model.load_weights("saved_weights/best_model.npz")
            rejector = TomatoRejector(confidence_threshold=0.75)
            predict_single_image(model, sys.argv[2], TOMATO_CLASSES, rejector=rejector)
    else:
        evaluate(
            dataset_dir   = "dataset",
            weights_path  = "saved_weights/best_model.npz",
            image_size    = (64, 64),
        )
        print("\n  MODULE 7 TEST PASSED ✓")