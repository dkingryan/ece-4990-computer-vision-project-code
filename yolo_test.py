import argparse
import cv2
import numpy as np


def load_yolo_model(cfg_path: str, weights_path: str, names_path: str):
    classes = []
    with open(names_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net, classes


def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except Exception:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_predictions(image, class_id, confidence, box, classes):
    x, y, w, h = box
    label = f"{classes[class_id]}: {confidence:.2f}"
    color = (0, 255, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x, y - text_size[1] - 8), (x + text_size[0] + 8, y), color, cv2.FILLED)
    cv2.putText(image, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def detect_objects(image, net, output_layers, conf_threshold=0.5, nms_threshold=0.4):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return [(class_ids[i], confidences[i], boxes[i]) for i in indices.flatten()] if len(indices) > 0 else []


def process_image(input_path, output_path, net, classes, conf_threshold, nms_threshold):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read input image: {input_path}")

    output_layers = get_output_layers(net)
    detections = detect_objects(image, net, output_layers, conf_threshold, nms_threshold)

    for class_id, confidence, box in detections:
        draw_predictions(image, class_id, confidence, box, classes)

    cv2.imwrite(output_path, image)
    print(f"Saved result to {output_path}")
    cv2.imshow("YOLO Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Simple YOLO object detection with OpenCV")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--version", default="yolov3", help="YOLO version for default config/weights file names (e.g. yolov3, yolov4, yolov3-tiny)")
    parser.add_argument("--cfg", default=None, help="Path to YOLO config file; if omitted uses <version>.cfg")
    parser.add_argument("--weights", default=None, help="Path to YOLO weights file; if omitted uses <version>.weights")
    parser.add_argument("--names", default="coco.names", help="Path to COCO class names file")
    parser.add_argument("--output", default="output.jpg", help="Path to save annotated output")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="Non-maxima suppression threshold")
    args = parser.parse_args()

    cfg_path = args.cfg if args.cfg else f"{args.version}.cfg"
    weights_path = args.weights if args.weights else f"{args.version}.weights"
    net, classes = load_yolo_model(cfg_path, weights_path, args.names)

    process_image(args.input, args.output, net, classes, args.conf, args.nms)

if __name__ == "__main__":
    main()
