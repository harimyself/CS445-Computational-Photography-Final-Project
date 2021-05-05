import cv2


def write_on_frame(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.putText(image,
                text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
