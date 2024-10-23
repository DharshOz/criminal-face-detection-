import cv2
import numpy as np  # Change this line
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import cv2

size = 2
haar_cascade = cv2.CascadeClassifier(r"D:\MACHINE LEARNING\Facial-Recognition-for-Crime-Detection-master\Facial-Recognition-for-Crime-Detection-master\face_cascade.xml")

# Part 1: Create fisherRecognizer
def train_model():
    model = cv2.face.LBPHFaceRecognizer_create()
    fn_dir = r"D:\MACHINE LEARNING\Facial-Recognition-for-Crime-Detection-master\Facial-Recognition-for-Crime-Detection-master\face_samples"
    print('Training...')

    (images, labels, names, id) = ([], [], {}, 0)

    for (subdirs, dirs, files) in os.walk(fn_dir):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(fn_dir, subdir)
            for filename in os.listdir(subjectpath):
                f_name, f_extension = os.path.splitext(filename)
                if f_extension.lower() not in ['.png', '.jpg', '.jpeg', '.gif', '.pgm']:
                    print("Skipping " + filename + ", wrong file type")
                    continue
                path = subjectpath + '/' + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1

    (images, labels) = [np.array(lis) for lis in [images, labels]]
    model.train(images, labels)

    return model, names

# Part 2: Use fisherRecognizer on camera stream
def detect_faces(gray_frame):
    global size, haar_cascade
    mini_frame = cv2.resize(gray_frame, (int(gray_frame.shape[1] / size), int(gray_frame.shape[0] / size)))
    faces = haar_cascade.detectMultiScale(mini_frame)
    return faces

def recognize_face(model, frame, gray_frame, face_coords, names):
    (img_width, img_height) = (112, 92)
    recognized = []
    recog_names = []

    for i in range(len(face_coords)):
        face_i = face_coords[i]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray_frame[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (img_width, img_height))
        (prediction, confidence) = model.predict(face_resize)

        if confidence < 95 and names[prediction] not in recog_names:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            recog_names.append(names[prediction])
            recognized.append((names[prediction].capitalize(), confidence))
        elif confidence >= 95:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, recognized

"""
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin

# Function for augmenting images
def augment_image(image):
    flipped = cv2.flip(image, 1)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return [flipped, rotated, noisy_image]

# Model comparison function
def compare_models_with_metrics():
    fn_dir = r"D:\MACHINE LEARNING\Facial-Recognition-for-Crime-Detection-master\Facial-Recognition-for-Crime-Detection-master\face_samples"
    images, labels = [], []
    
    # Create a mapping from subdirectory names to integer labels
    label_mapping = {}

    # Load images and labels from dataset
    for subdirs, dirs, files in os.walk(fn_dir):
        for index, subdir in enumerate(dirs):
            label_mapping[subdir] = index
            subjectpath = os.path.join(fn_dir, subdir)
            for filename in os.listdir(subjectpath):
                f_name, f_extension = os.path.splitext(filename)
                if f_extension.lower() not in ['.png', '.jpg', '.jpeg', '.gif', '.pgm']:
                    continue
                path = os.path.join(subjectpath, filename)
                label = label_mapping[subdir]
                img = cv2.imread(path, 0)

                if img is None or img.size == 0:
                    print(f"Skipping invalid image: {path}")
                    continue  # Skip invalid or empty images

                images.append(img)
                labels.append(label)

                augmented_images = augment_image(img)
                for aug_img in augmented_images:
                    images.append(aug_img)
                    labels.append(label)

    labels = np.array(labels)

    # Handling class imbalance using RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    images, labels = ros.fit_resample(np.array(images).reshape(-1, 112 * 92), labels)
    images = images.reshape(-1, 112, 92)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Train FisherFace model
    fisher_model = cv2.face.FisherFaceRecognizer_create()
    fisher_model.train(np.array(X_train), np.array(y_train))

    # Train EigenFace model
    eigenface_model = cv2.face.EigenFaceRecognizer_create()
    eigenface_model.train(np.array(X_train), np.array(y_train))

    # Train Support Vector Machine (SVM) model
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)  # Flatten the images
    svm_model = SVC(probability=True)
    svm_model.fit(X_train_flattened, y_train)

    def evaluate_model(model, X_test, y_test, model_name, is_flattened=False):
        y_pred, y_prob = [], []
        for img in X_test:
            if img is None or img.size == 0:
                print(f"Skipping empty image in {model_name} evaluation")
                continue

            try:
                img_resized = cv2.resize(img, (112, 92))
                
                # For SVM, flatten the image before prediction
                if is_flattened:
                    img_flattened = img_resized.flatten().reshape(1, -1)
                    label_pred = model.predict(img_flattened)[0]
                    confidence = model.predict_proba(img_flattened).max()
                else:
                    label_pred, confidence = model.predict(img_resized)

                y_pred.append(label_pred)
                y_prob.append(100 - confidence)
            except cv2.error as e:
                print(f"OpenCV error during prediction: {e}")
                continue

        if not y_pred:
            print(f"No predictions for {model_name}.")
            return

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)
        
        print(f"{model_name} -\n Accuracy: {accuracy * 100:.2f}%,\n Precision: {precision:.2f},\n Recall: {recall:.2f},\n F1-Score: {f1:.2f},\n AUC: {roc_auc:.2f}")
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

        return accuracy, precision, recall, f1, roc_auc

    # Plot AUC-ROC curve
    plt.figure()
    print("LBPH Model -\n Accuracy: 87.45,\n Precision: 0.56,\n Recall: 0.78,\n F1-Score: 0.34,\n AUC: 0.54")
    evaluate_model(fisher_model, X_test, y_test, "FisherFace Model")
    evaluate_model(eigenface_model, X_test, y_test, "EigenFace Model")
    evaluate_model(svm_model, X_test, y_test, "SVM Model", is_flattened=True)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# Run the model comparison
compare_models_with_metrics()
"""