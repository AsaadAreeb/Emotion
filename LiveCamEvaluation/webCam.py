import cv2
from model.deep_emotion import DeepEmotion
import torch
from torch.nn.functional import F
import numpy as np
from utils.config import checkpoint_path, haarcascade_path



# Load the Haar Cascade classifier for face detection
faceCascade = cv2.CascadeClassifier(haarcascade_path)

# Initialize the model for emotion recognition (replace 'model' with your actual model)
num_classes = 7
model = DeepEmotion(num_classes)
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
model.eval()

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, change to 1 if you have an additional camera

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Emotion recognition (process the ROI)
        gray_face = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        final_image = cv2.resize(gray_face, (48, 48))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0

        torchTensor = torch.from_numpy(final_image)
        torchTensor = torchTensor.type(torch.FloatTensor)

        output = model(torchTensor)
        pred = F.softmax(output, dim=1)

        index_pred = torch.argmax(pred)

        # Draw bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw emotion label on the frame
        emotion_label = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        emotion = emotion_label[index_pred.item()]
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
