import cv2

# Capture the video stream
cap = cv2.VideoCapture("volleyball_match.mp4")

# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Loop over the video stream
while True:

    # Capture the next frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply the background subtractor
    fg_mask = bg_subtractor.apply(gray)

    # Apply morphological operations
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, None)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, None)

    # Find contours in the foreground mask
    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()