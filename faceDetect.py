def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B].
    Return:
        detection_results: python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            It should be formed as [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    detection_results: List[List[float]] = []

    imageHeight = len(img)
    imageWidth = len(img[0])
    
    imgGray = np.zeros([imageHeight, imageWidth], dtype=np.uint8)
    for i in range(imageHeight):
        for j in range(imageWidth):
            imgGray[i, j] =  int((0.299+img[i][j][0] + 0.587+img[i][j][1] + 0.114+img[i][j][2]) / 3)

    haarCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    dim = haarCascade.detectMultiScale(imgGray, 1.1, 3)
    
    for x, y, width, height in dim:
        detection_results.append([float(x), float(y), float(width), float(height)])


    return detection_results
