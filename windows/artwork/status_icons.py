"""Status icons for the setup's requirements page: a filled circle with a white symbol,
drawn at 4x and scaled down for smooth edges; 64x64 RGBA PNGs.

  python status_icons.py [out folder, default windows/]

Needs NumPy and OpenCV, like wizard_images.py."""
import os, sys, numpy as np, cv2
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
N, K = 64, 4
S = N * K
def icon(name, bgr, draw):
    img = np.zeros((S, S, 4), np.uint8)
    cv2.circle(img, (S // 2, S // 2), S // 2 - K, (*bgr, 255), -1, cv2.LINE_AA)
    draw(img)
    small = cv2.resize(img, (N, N), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{OUT}/{name}.png", small)
W = (255, 255, 255, 255)
t = S // 11
icon("ok", (60, 160, 40), lambda im: cv2.polylines(im, [np.array([[S*0.28, S*0.52], [S*0.44, S*0.68], [S*0.73, S*0.36]], np.int32)], False, W, t, cv2.LINE_AA))
icon("missing", (50, 50, 210), lambda im: [cv2.line(im, (int(S*a), int(S*b)), (int(S*c), int(S*d)), W, t, cv2.LINE_AA)
                                           for a, b, c, d in ((0.33, 0.33, 0.67, 0.67), (0.67, 0.33, 0.33, 0.67))])
def arrow(im):
    cv2.line(im, (S // 2, int(S * 0.26)), (S // 2, int(S * 0.70)), W, t, cv2.LINE_AA)
    cv2.polylines(im, [np.array([[S*0.32, S*0.52], [S*0.5, S*0.71], [S*0.68, S*0.52]], np.int32)], False, W, t, cv2.LINE_AA)
icon("install", (200, 120, 20), arrow)
def info(im):
    cv2.circle(im, (S // 2, int(S * 0.30)), int(t * 0.75), W, -1, cv2.LINE_AA)
    cv2.line(im, (S // 2, int(S * 0.46)), (S // 2, int(S * 0.74)), W, t, cv2.LINE_AA)
icon("info", (140, 130, 120), info)
def warn(im):
    cv2.line(im, (S // 2, int(S * 0.24)), (S // 2, int(S * 0.56)), W, t, cv2.LINE_AA)
    cv2.circle(im, (S // 2, int(S * 0.72)), int(t * 0.75), W, -1, cv2.LINE_AA)
icon("warn", (0, 150, 230), warn)
