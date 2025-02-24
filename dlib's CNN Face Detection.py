import dlib
import cv2

# speed: Slower than HOG + SVM, requires more resources (especially GPU acceleration).
# accuracy: Highly accurate, detects faces well even under challenging conditions (e.g., occlusion, low lighting).
# Use Case: High accuracy for offline batch processing, not ideal for real-time unless using a GPU.
def cnn_face_detection(image_path):
    # Load the pre-trained dlib CNN face detector (requires dlib with CUDA support)
    detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    # Draw rectangle around faces
    for face in faces:
        x, y, w, h = (face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height())
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    resized_image = cv2.resize(img, (800, 600))
    cv2.imshow("Detected Faces", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
cnn_face_detection('boboxa.jpg')
