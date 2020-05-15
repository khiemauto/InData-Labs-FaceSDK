import cv2

points_colors = [
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 0, 0)
]


def draw_boxes(img, boxes, conf_threshold=0., color=(0, 0, 255), thickness=2):
    font_scale = float(max(img.shape) * 0.001)
    print(f"font_scale = {font_scale}")
    for box in boxes:
        conf = box[4]
        if conf < conf_threshold:
            continue
        text = f"{conf:.4f}"
        b = list(map(int, box))
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color=color, thickness=thickness)
        tx = b[0]
        ty = b[1] + 12
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255))


def draw_landmarks(img, landmarks):
    thickness = int(max(img.shape) * 0.01)
    print(f"thickness = {thickness}")
    for landm in landmarks:
        landm = list(map(int, landm))
        l = iter(landm)
        for i, p in enumerate(zip(l, l)):
            cv2.circle(img, (p[0], p[1]), 1, points_colors[i], thickness)
