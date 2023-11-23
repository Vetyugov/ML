import cv2
import numpy
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import tensorflow

image = cv2.imread("people.jpg")
box, label, count = cv.detect_common_objects(image)
output = draw_bbox(image, box, label, count)

print(label)
print(f"Колличество объектов на картинке: {label.count('person')}")

plt.imshow(output)
plt.show()