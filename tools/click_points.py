import cv2

points = []

img = cv2.imread("notebooks\\image1.png")

if img is None:
    raise RuntimeError("Image failed to load. Check the path.")

def click(event, x, y, flags, param):
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"{x}, {y}")
        points.append([x, y])
        cv2.circle(img, (x,y), 5, (0,0,255), -1)
        cv2.imshow("image", img)
        cv2.putText(img, str(len(points)), (x+5, y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imshow("image", img)
cv2.setMouseCallback("image", click)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nCollected points:")
print(points)