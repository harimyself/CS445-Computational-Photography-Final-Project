import cv2


def clean(image):
    scale_percent = 50

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim)

    print('Resized Dimensions : ', resized.shape)

    resized = resized[71:-60, int(width / 2):]

    return resized
