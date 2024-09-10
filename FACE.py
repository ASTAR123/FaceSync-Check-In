import cv2
from tensorflow.keras.models import load_model
from skimage import io, color, transform
import glob
import os
import numpy as np
from skimage import io, color, transform
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

def load_face_data(directory):
    Faces = []
    Names = []
    for k, i in enumerate(glob.glob(os.path.join(directory, "*"))):
        imagei = io.imread(i)
        Names.append(os.path.basename(i).split('_')[0])

        imagei_gray = color.rgb2gray(imagei)
        
        image_rescaled = transform.resize(imagei_gray, output_shape=(128, 128), anti_aliasing=False)
        
        mean_value = np.mean(image_rescaled)
        std_dev = np.std(image_rescaled)
        normalized_image = (image_rescaled - mean_value) / std_dev
        Faces.append(normalized_image.flatten())

    return Faces, Names
def SVD_PCA(faces, h=128, w=128, variance_ratio=0.99):
    mean_face = np.mean(faces, axis=0)
    centered_faces = faces - mean_face
    U, singular_values, Vt = np.linalg.svd(centered_faces, full_matrices=False)
    total_variance = np.sum(singular_values**2)
    k_value_with_90_ratio = None

    for k in range(20, 1000):
        explained_variance_ratio = np.sum(singular_values[:k]**2) / total_variance
        if explained_variance_ratio >= 0.90:
            k_value_with_90_ratio = k
            print(f"For explained variance ratio >= 90%, choose k={k_value_with_90_ratio}")
            break
    principal_components = Vt[:k_value_with_90_ratio]
    projected_data = np.dot(centered_faces, principal_components.T)
    return projected_data, principal_components

def face():
    recognized_faces = []
    faces, names = load_face_data("FaceData-2")

    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(names)

    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_labels = integer_labels.reshape(len(integer_labels), 1)
    labels = onehot_encoder.fit_transform(integer_labels)


    faces_PCA, principal_components = SVD_PCA(faces)


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    model_path = "./saved_model/best_model_final_2024-02-28_15-27-36.keras"
    model = load_model(model_path)

    cap = cv2.VideoCapture(0)
    while True:
        # Capture video frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection 
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Face extraction
            face = frame[y:y+h, x:x+w]
            
            # Convert to grayscale image
            imagei_gray = color.rgb2gray(face)
            
            # Resize image to a fixed size
            image_rescaled = transform.resize(imagei_gray, output_shape=(128, 128), anti_aliasing=False)
            
            # Centering and Normalization
            mean_value = np.mean(image_rescaled)
            std_dev = np.std(image_rescaled)
            normalized_image = (image_rescaled - mean_value) / std_dev

            # Flatten the normalized image
            face = normalized_image.flatten()
            
            face = np.dot(face, principal_components.T)
            
            # Reshape the input to match the desired shape of the model
            face = face.reshape(1, -1)  # The shapes required by the model are (1, features)
            
            # Face recognition 
            prediction = model.predict(face)

            # Converts the result of One-Hot encoding back to an integer label
            predicted_integer = np.argmax(prediction, axis=1)

            # Use LabelEncoder's inverse_transform method to convert the integer label back to the original label
            predicted_names = label_encoder.inverse_transform(predicted_integer)

            # The prediction is displayed on the face box
            cv2.putText(frame, str(predicted_names[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            recognized_faces.append(predicted_names[0])
            
        # Display result frame
        cv2.imshow('Face Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Freeing camera resources
    cap.release()
    cv2.destroyAllWindows()

    if len(recognized_faces) > 0:
        return recognized_faces 
    else:
        return None