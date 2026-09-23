import cv2
import math

model = "face_detection_yunet_2023mar.onnx"

detector = cv2.FaceDetectorYN.create(
    model,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access camera")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame")
        break

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            # Face box
            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            # Five facial landmarks
            landmarks = face[4:14].reshape(5, 2)

            for point in landmarks:

                px, py = point.astype(int)

                cv2.circle(
                    frame,
                    (px, py),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Mouth landmarks
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]

            # Distance between mouth points
            mouth_width = math.dist(
                left_mouth,
                right_mouth
            )

            # Face width
            face_width = w

            # Normalize mouth size
            mouth_ratio = mouth_width / face_width

            if mouth_ratio > 0.35:

                expression = "MOUTH OPEN"

            else:

                expression = "MOUTH CLOSED"

            cv2.putText(
                frame,
                expression,
                (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("OpenCV Expression Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()