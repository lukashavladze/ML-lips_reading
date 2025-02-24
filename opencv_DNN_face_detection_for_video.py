import cv2

#works tested!

# speed: Can run in real-time with CPU, but speed can be improved with GPU.
# accuracy: Good balance between speed and accuracy.
# Use Case: Real-time face detection with better accuracy than Haar cascades, suitable for moderate-performance systems.
def dnn_face_detection(video_path):
    # Load the pre-trained OpenCV DNN face detector model
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

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

        h, w = img.shape[:2]

        # Prepare the image for input to the network
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False,
                                     crop=False)

        # Feed the image to the network
        net.setInput(blob)
        detections = net.forward()

        # Draw rectangle around faces
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

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
dnn_face_detection('test_vid1.mp4')
