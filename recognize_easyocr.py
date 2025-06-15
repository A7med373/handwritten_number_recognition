# recognize_easyocr.py

import numpy as np
import cv2
import easyocr

def recognize_digits(image_path, conf_thresh=0.3):

    if img is None:
        raise FileNotFoundError(f"Cannot load '{image_path}'")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(
        gray,
        detail=1,
        paragraph=False,
        allowlist='0123456789',
        text_threshold=0.4,    # lower => detect fainter text
        low_text=0.3           # lower => keep more low‐conf detections
    )

    filtered = [(bbox, txt, conf)
                for bbox, txt, conf in results
                if conf >= conf_thresh]

    filtered.sort(key=lambda r: min(pt[0] for pt in r[0]))

    number = ''.join(txt for _, txt, _ in filtered)
    return number, filtered

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Recognize a multi‐digit handwritten number with EasyOCR")
    parser.add_argument('image', help="Path to your number image")
    parser.add_argument('--conf', type=float, default=0.3,
                        help="Minimum confidence threshold (0–1)")
    args = parser.parse_args()

    number, results = recognize_digits(args.image, conf_thresh=args.conf)
    print("Recognized number:", number)
    print("Raw results (bbox, text, confidence):")
    for bbox, txt, conf in results:
        print(f"  {txt!r} @ {conf:.2f}")

    # OPTIONAL: draw boxes & show image
    img = cv2.imread(args.image)
    for bbox, txt, conf in results:
        pts = np.array(bbox, np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,255,0), 2)
        org = (int(bbox[0][0]), int(bbox[0][1]) - 10)
        cv2.putText(img, f"{txt} ({conf:.2f})", org,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("OCR result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()