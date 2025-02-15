import os


def load_annotations(file_path):
    """Načítanie YOLO anotácií len zo správnych .txt súborov."""
    annotations = []
    # Skontroluj, či má súbor príponu `.txt`
    if not file_path.endswith('.txt'):
        return annotations  # Vráti prázdny zoznam, ak to nie je .txt súbor

    # Otvor len textové súbory
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            # Skontroluj, či riadok obsahuje presne 5 hodnôt (class_id, x, y, width, height)
            if len(parts) != 5:
                continue
            # Skontroluj, či sú všetky hodnoty číselné (žiaden text ako 'person')
            if not all(part.replace('.', '', 1).isdigit() for part in parts):
                continue
            # Preveď hodnoty na float
            class_id, x_center, y_center, width, height = map(float, parts)
            annotations.append([class_id, x_center, y_center, width, height])

    return annotations


def calculate_iou(box1, box2):
    #"""Vypočíta Intersection over Union (IoU) pre dva bounding boxy."""
    # Konverzia YOLO formátu na súradnice [x1, y1, x2, y2]
    x1_box1 = box1[1] - box1[3] / 2
    y1_box1 = box1[2] - box1[4] / 2
    x2_box1 = box1[1] + box1[3] / 2
    y2_box1 = box1[2] + box1[4] / 2

    x1_box2 = box2[1] - box2[3] / 2
    y1_box2 = box2[2] - box2[4] / 2
    x2_box2 = box2[1] + box2[3] / 2
    y2_box2 = box2[2] + box2[4] / 2

    # Výpočet prieniku
    x1 = max(x1_box1, x1_box2)
    y1 = max(y1_box1, y1_box2)
    x2 = min(x2_box1, x2_box2)
    y2 = min(y2_box1, y2_box2)

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    box2_area = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def evaluate(pred_dir, gt_dir, iou_threshold=0.5):
    """Porovná predikcie a ground truth a vypočíta štatistiky."""
    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))

    tp, fp, fn = 0, 0, 0

    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)

        predictions = load_annotations(pred_path)
        ground_truth = load_annotations(gt_path)

        matched = set()

        # Porovnaj predikcie s ground truth
        for pred_box in predictions:
            best_iou = 0
            best_gt_idx = -1
            for idx, gt_box in enumerate(ground_truth):
                if idx not in matched:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            if best_iou >= iou_threshold:
                tp += 1
                matched.add(best_gt_idx)
            else:
                fp += 1

        # False negatives: nezachytené ground truth boxy
        fn += len(ground_truth) - len(matched)

    # Výpočet metrík
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score, tp, fp, fn

# Cesty k priečinkom
pred_dir = "video_frames"  # Zmeň na svoju cestu
gt_dir = "video_frames_groundtruth"          # Zmeň na svoju cestu

# Vyhodnotenie
precision, recall, f1_score, tp, fp, fn = evaluate(pred_dir, gt_dir)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")
print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
