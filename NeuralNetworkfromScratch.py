import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import OneHotEncoder

# Data Processing

folder_path = r"D:\DigitRecognition"
target_size = (20,20)

def load_and_process_images(img_path, target_size):
    # Load image
    img = Image.open(img_path)
    img = img.resize(target_size).convert('L')  # Convert to grayscale
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    return img

def load_images_from_folder(folder_path, target_size):
    images = []
    labels = []
    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = load_and_process_images(img_path, target_size)
                images.append(img)
                labels.append(label_folder)  # Assuming folder name is the label
    return np.array(images), np.array(labels)

images, labels = load_images_from_folder(folder_path, target_size)

# Flatten each image
images = images.reshape(images.shape[0], -1)
labels = labels.reshape(-1, 1)

print(f"First image data: {images[0]}, Label: {labels[0]}")
print(f"Shape of images array: {np.shape(images)}")
print(f"Shape of labels array: {np.shape(labels)}")

encoder = OneHotEncoder()
one_hot_labels = encoder.fit_transform(labels)
one_hot_labels = one_hot_labels.toarray().astype('float64')


def relu(z):
    return np.maximum(0, z)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_probs = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
    return np.sum(log_probs) / m

class NeuralNetwork:

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        self.w1 = np.random.uniform(low=-0.5, high=0.5, size=(input_size, hidden_size_1))
        self.b1 = np.random.uniform(low=-0.5, high=0.5, size=(1, hidden_size_1))
        self.w2 = np.random.uniform(low=-0.5, high=0.5, size=(hidden_size_1, hidden_size_2))
        self.b2 = np.random.uniform(low=-0.5, high=0.5, size=(1, hidden_size_2))
        self.w3 = np.random.uniform(low=-0.5, high=0.5, size=(hidden_size_2, output_size))
        self.b3 = np.random.uniform(low=-0.5, high=0.5, size=(1, output_size))

    def forward_propagation(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = relu(self.z2)
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(self.z3)
        return self.a3

    def backward_propagation(self, X, y_true, y_predicted, learning_rate):
        m = y_true.shape[0]

        dZ3 = y_predicted - y_true

        dW3 = np.dot(self.a2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, self.w3.T)
        dZ2 = dA2 * relu_derivative(self.a2)
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.w2.T)
        dZ1 = dA1 * relu_derivative(self.a1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.w3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.w2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def compute_accuracy(self, X, y):
        y_pred = self.forward_propagation(X)
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward_propagation(X)
            loss = cross_entropy_loss(y, y_pred)
            self.backward_propagation(X, y, y_pred, learning_rate)
            if (epoch + 1) % 1000 == 0:
                accuracy = self.compute_accuracy(X, y)
                print(f"Epoch: {epoch + 1}, Loss: {loss}, Accuracy: {accuracy}")

    def print_params(self):
        print(f"W1: {self.w1}")
        print(f"b1: {self.b1}")
        print(f"W2: {self.w2}")
        print(f"b2: {self.b2}")
        print(f"W3: {self.w3}")
        print(f"b3: {self.b3}")

    def predict(self,input):
        return self.forward_propagation(input)


# Example training
nn = NeuralNetwork(400, 128, 64, 3)
nn.train(images, one_hot_labels, 10000, 0.009)
nn.print_params()

for i in range(10):
    j=np.random.randint(0,6000)
    print(f"Actual digit: {np.argmax(one_hot_labels[j])}")
    prediction=nn.predict(images[j])
    print(prediction) 
    print(f"Predicted Digit :{np.argmax(prediction)}")

