import cv2

# Step 1: Access the Camera
# 0 is usually the default webcam. If you have multiple, try 1 or 2.
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Camera opened. Press 'q' to quit.")

while True:
    # Step 2: Read a frame
    # ret is a boolean (True if frame read correctly)
    # frame is the image array (Matrix)
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Step 3: Image Manipulation (Since you know this!)
    # Let's flip it so it acts like a mirror
    frame = cv2.flip(frame, 1)

    # Step 4: Display the frame
    cv2.imshow('Sign Language Interpreter - Camera Test', frame)

    # Step 5: Handle Input
    # Wait 1ms for a key press. If 'q' is pressed, break loop.
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
