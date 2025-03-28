import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained TensorFlow model (MobileNetV2 as a placeholder for face recognition)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Camera Intrinsic Parameters (to be calibrated for your camera)
FOCAL_LENGTH = 700  # in pixels (example value, needs calibration for your camera)
PRINCIPAL_POINT = (320, 240)  # Center of the image (example for 640x480 resolution)

# Real-world size of the reference object (e.g., a marker or known object)
REFERENCE_OBJECT_WIDTH = 0.2  # meters (adjust to your object size)

# Preprocessing function for input image (resize and normalization)
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # MobileNet expects 224x224 input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Function to calculate real-world 3D coordinates
def pixel_to_real_world(x, y, width, depth):
    # Convert pixel (x, y) to real-world coordinates using camera intrinsics
    real_x = (x - PRINCIPAL_POINT[0]) * depth / FOCAL_LENGTH
    real_y = (y - PRINCIPAL_POINT[1]) * depth / FOCAL_LENGTH
    return real_x, real_y, depth

# Function to calculate depth based on the apparent size of the detected face
def estimate_depth(face_width_in_pixels):
    # Using the known real-world width of the reference object and its size in pixels to estimate depth
    depth = (REFERENCE_OBJECT_WIDTH * FOCAL_LENGTH) / face_width_in_pixels
    return depth

# Detect faces and recognize them with zoom calibration
def detect_and_recognize_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    face_locations = []  # To store real-world locations of detected faces

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Estimate the depth (Z-axis) based on the face width in pixels
        depth = estimate_depth(w)
        
        # Convert pixel coordinates to real-world coordinates
        real_x, real_y, real_z = pixel_to_real_world(x + w // 2, y + h // 2, w, depth)
        
        # Store the real-world coordinates
        face_locations.append((real_x, real_y, real_z))
        
        # Extract the face area from the frame
        face = frame[y:y + h, x:x + w]
        
        # Preprocess the face for recognition
        face_input = preprocess_image(face)
        
        # Perform prediction using the pre-trained model
        predictions = model.predict(face_input)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
        
        # Display the predicted class on the image
        label = decoded_predictions[0][0][1]  # Top prediction label
        confidence = decoded_predictions[0][0][2]  # Confidence of the prediction
        #cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Display real-world coordinates on the frame
        cv2.putText(frame, f"Real-world (X, Y, Z): ({real_x:.2f}, {real_y:.2f}, {real_z:.2f}) meters",
                    (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, face_locations

# Video capture for real-time face detection and recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect and recognize faces in the frame
    frame_with_faces, face_locations = detect_and_recognize_faces(frame)

    # Display the output frame
    cv2.imshow('Face Recognition with 3D Location', frame_with_faces)
    
    # Print real-world locations of detected faces
    if face_locations:
        print("Real-world locations:", face_locations)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
