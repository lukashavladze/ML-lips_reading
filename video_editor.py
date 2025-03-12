import cv2

# Input and output video paths
input_video = "C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\m1m1.mp4"
output_video = "C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\trimmed_video.mp4"

# Open the video file
cap = cv2.VideoCapture(input_video)

# Get original video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 output

# Initialize video writer
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened() and frame_count < 75:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends before 75 frames

    # Write the frame to the output video
    out.write(frame)

    frame_count += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved trimmed video as {output_video} with 75 frames.")
