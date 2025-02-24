import cv2

# works Tested!

# speed: VERY FAST
# accuracy: Lower accuracy, but works well for real-time detection
def haar_face_detection(video_path):
    # Load the pre-trained Haar Cascade model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangle around faces
        for (x, y, w, h) in faces:
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
haar_face_detection('test_vid1.mp4')
