import cv2
import numpy as np
#intervalSpecifier specifies how big the interval is, opencv doc recomends 10
def getHSVcolorRangeFromBGR(bgr_color, intervalSpecifier):
    # bgrcolor = np.uint8([[[bgr_color]]])
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)
    hue = int(hsv_color[0][0][0])

    # Create lower and upper HSV bounds
    lower_bound = np.array([max(hue-intervalSpecifier, 0), 50, 50])
    upper_bound = np.array([min(hue+intervalSpecifier, 179), 255, 255])
    return lower_bound, upper_bound


def getDarkHSVcolorRangeFromBGR(bgr_color):

    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)

    # Extract HSV components
    hue = int(hsv_color[0][0][0])
    saturation = int(hsv_color[0][0][1])
    value = int(hsv_color[0][0][2])

    # Define ranges for dark color
    lower_hue = max(0, hue - 10)
    upper_hue = min(179, hue + 10)
    lower_saturation = max(0, saturation - 30)  # Relax saturation for dark colors
    upper_saturation = min(255, saturation + 50)
    lower_value = max(0, value - 30)  # Relax value for dark colors
    upper_value = min(255, value + 50)

    # Set lower and upper bounds
    lower_bound = np.array([lower_hue, lower_saturation, lower_value])
    upper_bound = np.array([upper_hue, upper_saturation, upper_value])

    return lower_bound, upper_bound
# color = np.uint8([[[5, 19, 76]]])
# print(getHSVcolorRangeFromBGR(color))