import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# Set the path to the folder containing PPM files
folder_path = 'path/to/folder/with/ppm/files'

# Function to read a PPM file and convert it to a numpy array
def read_ppm(file_path):
    with open(file_path, 'rb') as f:
        header = f.readline().decode().strip()
        width, height = [int(x) for x in f.readline().decode().strip().split()]
        max_val = int(f.readline().decode().strip())
        img_data = np.frombuffer(f.read(), dtype=np.uint8)
        img_data = img_data.reshape((height, width, 3))
    return img_data

# Load the images and labels
images = []
labels = []
for file_name in os.listdir(folder_path):
    if file_name.endswith('.ppm'):
        file_path = os.path.join(folder_path, file_name)
        img_data = read_ppm(file_path)
        images.append(img_data)
        labels.append(int(file_name.split('_')[0]))  # Assuming labels are part of the file name

# Resize the images and convert to 1-D arrays
resized_images = []
for img in images:
    resized_img = np.array(Image.fromarray(img).resize((30, 30)))
    resized_images.append(resized_img.flatten())

# Convert to numpy arrays
images_np = np.array(resized_images)
labels_np = np.array(labels)

# Plot sample images from each class
unique_labels = np.unique(labels_np)
fig, axs = plt.subplots(1, len(unique_labels), figsize=(15, 3))
for i, label in enumerate(unique_labels):
    sample_idx = np.where(labels_np == label)[0][0]
    axs[i].imshow(images_np[sample_idx].reshape(30, 30, 3))
    axs[i].set_title(f'Class: {label}')
plt.show()

# Plot the distribution of classes
plt.hist(labels_np, bins=len(unique_labels))
plt.title('Distribution of Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Normalize the features
images_np = images_np / 255.0

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images_np, labels_np, test_size=0.2, random_state=42)

# Train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.0001, learning_rate='constant')
mlp.fit(X_train, y_train)

# Evaluate the model
y_pred = mlp.predict(X_test)
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Hyperparameter tuning
parameter_space = {
    'hidden_layer_sizes': [(100, 50), (50, 100), (100, 100), (200, 100), (200, 200)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.9],
    'learning_rate': ['constant', 'adaptive']
}

clf = GridSearchCV(MLPClassifier(), parameter_space, cv=5, scoring='accuracy', n_jobs=-1)
clf.fit(X_train, y_train)

print('Best Parameters:', clf.best_params_)
print('Best Score:', clf.best_score_)
