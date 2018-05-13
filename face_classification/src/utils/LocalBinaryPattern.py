from skimage import feature, io
import numpy as np
import cv2


class LocalBinaryPatterns:
    def __init__(self, radius):
        self.radius = radius
        self.numPoints = radius * 8

    def describe(self, image_path, eps=1e-7):
        im = cv2.imread(image_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        lbp = feature.local_binary_pattern(im, self.numPoints,
                                           self.radius, method="uniform")

        print(lbp)
        cv2.imwrite("images.png", lbp)


        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist

# radius = 10
# lbp = LocalBinaryPatterns(radius)
# lbp.describe("../images/predicted_test_image.png")