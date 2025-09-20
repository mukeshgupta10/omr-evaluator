# omr/processor.py
import cv2
import numpy as np
from PIL import Image
import io
import math

# ---------- helpers ----------
def read_image_file(file_like):
    """file_like can be a path string or a file-like object (streamlit uploaded file)."""
    if isinstance(file_like, str):
        img = cv2.imread(file_like)
        return img
    data = file_like.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def resize_by_width(image, width=1000):
    h, w = image.shape[:2]
    if w == width:
        return image
    ratio = width / float(w)
    return cv2.resize(image, (width, int(h * ratio)))

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts, width=1000):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    maxWidth = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl), width))
    maxHeight = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# ---------- paper detection & warp ----------
def find_paper_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2)
    return None

def preprocess_and_warp(image, target_width=1000):
    image = resize_by_width(image, width=1200)
    paper = find_paper_contour(image)
    if paper is not None:
        warped = four_point_transform(image, paper, width=target_width)
    else:
        # fallback: central crop/resize
        warped = resize_by_width(image, width=target_width)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # adaptive threshold to handle lighting
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 10)
    return warped, gray, thresh

# ---------- bubble detection ----------
def detect_bubble_contours(thresh_img, min_area=700, max_area=4000, debug=False):
    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    bubbles = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        ar = w/float(h)
        if ar < 0.6 or ar > 1.4:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4*math.pi*area/(peri*peri)
        if circularity < 0.45:  # tuned threshold
            continue
        bubbles.append((c, (x,y,w,h)))
    # sort by y then x
    bubbles = sorted(bubbles, key=lambda b: (b[1][1], b[1][0]))
    if debug:
        print(f"Detected {len(bubbles)} candidate bubbles")
    return bubbles

# ---------- grouping into questions ----------
def group_bubbles_into_questions(bubble_contours, num_questions, choices_per_question):
    """
    Simple and robust: sort by y coordinate and split into `num_questions` groups.
    Each group is then sorted by x -> map to choices A,B,C...
    Works reliably for standard grid OMR sheets where each question's bubbles are aligned.
    """
    n = len(bubble_contours)
    expected = num_questions * choices_per_question
    # If counts mismatch, np.array_split will still try to split evenly
    idxs = np.arange(n)
    # split indices top-to-bottom into num_questions chunks
    chunks = np.array_split(idxs, num_questions)
    groups = []
    bboxes = [b[1] for b in bubble_contours]
    for chunk in chunks:
        if len(chunk) == 0:
            groups.append([])
            continue
        # sort each chunk by x coordinate
        sorted_chunk = sorted(list(chunk), key=lambda i: bboxes[i][0])
        groups.append(sorted_chunk)
    return groups

def bubble_fill_ratio(thresh_img, bbox):
    x,y,w,h = bbox
    roi = thresh_img[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    filled = cv2.countNonZero(roi)
    total = roi.shape[0]*roi.shape[1]
    return filled/float(total)

# ---------- evaluation ----------
def evaluate_image(image_or_file, num_questions, choices_per_question, answer_key,
                   fill_threshold=0.5, min_area=700, max_area=4000, debug=False):
    """
    Returns: score, total, per_q_results(list), annotated_image (BGR numpy)
    per_q_results: list of dicts {question, selected, correct, is_correct, ratios}
    answer_key: dict mapping 1..n -> 'A'|'B'...  (keys can be int or str)
    """
    # Read image
    if isinstance(image_or_file, (str, bytes)) or hasattr(image_or_file, "read"):
        image = read_image_file(image_or_file)
    else:
        image = image_or_file
    if image is None:
        raise ValueError("Could not read image")

    warped, gray, thresh = preprocess_and_warp(image)
    bubble_contours = detect_bubble_contours(thresh, min_area=min_area, max_area=max_area, debug=debug)
    groups = group_bubbles_into_questions(bubble_contours, num_questions, choices_per_question)

    opts = [chr(ord('A') + i) for i in range(choices_per_question)]
    score = 0
    per_q = []
    annotated = warped.copy()

    for q_idx, group in enumerate(groups, start=1):
        ratios = []
        # compute ratio for each detected bubble in this group
        for i in range(len(group)):
            idx = group[i]
            bbox = bubble_contours[idx][1]
            r = bubble_fill_ratio(thresh, bbox)
            ratios.append(r)
        selected = None
        if len(ratios) > 0:
            max_r = max(ratios)
            sel_idx = ratios.index(max_r)
            if max_r >= fill_threshold:
                selected = opts[sel_idx]
        # map correct
        correct = answer_key.get(str(q_idx), answer_key.get(q_idx))
        is_correct = (selected is not None and correct is not None and str(selected).upper() == str(correct).upper())
        if is_correct:
            score += 1
        per_q.append({
            "question": q_idx,
            "selected": selected,
            "correct": correct,
            "is_correct": is_correct,
            "ratios": ratios
        })
        # annotate: draw boxes and letters
        for i, idx in enumerate(group):
            bbox = bubble_contours[idx][1]
            x,y,w,h = bbox
            cx = int(x + w/2); cy = int(y + h/2)
            # color
            color = (200,200,200)  # gray
            if selected is not None and i == sel_idx:
                color = (0,255,0) if is_correct else (0,0,255)
            # rectangle + label
            cv2.rectangle(annotated, (x,y), (x+w, y+h), color, 2)
            cv2.putText(annotated, opts[i], (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        # draw question number
        if len(group) > 0:
            first_bbox = bubble_contours[group[0]][1]
            qx, qy = first_bbox[0]-40, first_bbox[1]+int(first_bbox[3]/2)
            cv2.putText(annotated, str(q_idx), (max(5,qx), qy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,50,50), 2)

    total = num_questions
    return score, total, per_q, annotated, thresh

# ---------- batch helpers ----------
def evaluate_batch(file_list, **kwargs):
    all_results = []
    for f in file_list:
        score, total, per_q, annotated, _ = evaluate_image(f, **kwargs)
        all_results.append({"file": getattr(f, "name", "image"), "score": score, "total": total, "per_q": per_q, "annotated": annotated})
    return all_results
