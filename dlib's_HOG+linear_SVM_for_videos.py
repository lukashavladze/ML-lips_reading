import dlib
import cv2

# speed: Faster than CNN-based models, but slower than Haar cascades
# accuracy: More accurate than Haar cascades, good for general-purpose face detection.
def hog_svm_face_detection(video_path):
    # Load the pre-trained dlib HOG + SVM face detector
    detector = dlib.get_frontal_face_detector()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # Read the current frame
        ret, img = cap.read()

        if not ret:
            print("Error: Failed to read frame from video.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        # Draw rectangle around faces
        for rect in faces:
            x, y, w, h = (rect.left(), rect.top(), rect.width(), rect.height())
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Resize the image for better visibility in window
        resized_image = cv2.resize(img, (800, 600))

        # Display the frame with faces detected
        cv2.imshow("Detected Faces", resized_image)

        # Check for exit condition (press 'q' to exit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()


# Example usage
hog_svm_face_detection('test_eng_vid.mp4')
