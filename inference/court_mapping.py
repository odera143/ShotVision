import cv2
import numpy as np
import math

img_pts = np.array([
 [461, 339], [431, 362], [283, 468], [64, 620], [874, 389], [927, 509], [799, 665], [445, 826]
], dtype=np.float32)

court_pts = np.array([
 [25, -5],    
 [22, -5],   
 [8, -5],
 [-8,-5],
 [22, 9],
 [8, 14],
 [-8, 14],
 [-22, 9]
], dtype=np.float32)

H, inliers = cv2.findHomography(img_pts, court_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
H_inv = np.linalg.inv(H)

def img_to_court(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)[0,0]
    return float(mapped[0]), float(mapped[1])

def court_to_img(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H_inv)[0,0]
    return int(mapped[0]), int(mapped[1])

img = cv2.imread("notebooks\\image1.png")

if img is None:
    raise RuntimeError("Image failed to load. Check the path.")

hx, hy = court_to_img(0,0)

cv2.circle(img, (hx, hy), 10, (0,0,255), -1)
cv2.putText(img, "HOOP", (hx+10, hy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

for x in range(-25, 26, 5):
    for y in range(-5, 47, 5):

        px, py = court_to_img(x, y)

        cv2.circle(img, (px, py), 3, (0,255,0), -1)

for angle in range(-70, 71, 5):
    r = 23.75
    x = r * math.sin(math.radians(angle))
    y = r * math.cos(math.radians(angle))

    px, py = court_to_img(x, y)

    cv2.circle(img, (px, py), 2, (255,0,0), -1)

# player foot pos
cv2.circle(img, (1326, 779), 2, (255,0,225), -1)

print(img_to_court(1326, 779))

cv2.imshow("homography check", img)
cv2.waitKey(0)
cv2.destroyAllWindows()