import dlib
import cv2

# speed: Faster than CNN-based models, but slower than Haar cascades
# accuracy: More accurate than Haar cascades, good for general-purpose face detection.
def hog_svm_face_detection(image_path):
    # Load the pre-trained dlib HOG + SVM face detector
    detector = dlib.get_frontal_face_detector()

    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    # Draw rectangle around faces
    for rect in faces:
        x, y, w, h = (rect.left(), rect.top(), rect.width(), rect.height())
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    resized_image = cv2.resize(img, (800, 600))

    cv2.imshow("Detected Faces", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
hog_svm_face_detection('boboxa_1.jpg')
