import matplotlib.pyplot as plt
import numpy as np

filepath="./data/preprocessed/mnist_preprocessed.npz"
data = np.load(filepath)
training_data = list(zip(
    [x.reshape((784,1)) for x in data['train_x']],
    [y.reshape((10,1)) for y in data['train_y']]
))
validation_data = list(zip(
    [x.reshape((784,1)) for x in data['val_x']],
    data['val_y']
))
test_data = list(zip(
    [x.reshape((784,1)) for x in data['test_x']],
    data['test_y']
))

example_train = training_data[0] # You can choose any (in range ofc)
example_val = validation_data[0]
example_test = test_data[0]

print(f"Train data size: {len(training_data)}")
print(f"Validation data size: {len(validation_data)}")
print(f"Test data size: {len(test_data)}")

print(f"First training input shape: {example_test[0].shape}")
print(f"First training label shape: {example_test[1].shape}")
print(f"First training label vector (one-hot):\n{example_test[1]}")

print(f"First validation label: {example_val[1]}")
print(f"First test label: {example_train[1]}")
print(f"Vstupní data: {example_train[0]}")

# Visualization

train_img = example_train[0].reshape(28,28)
val_img = example_val[0].reshape(28,28)
test_img = example_test[0].reshape(28,28)

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(train_img, cmap='gray')
plt.title(f"Label: Train")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(val_img, cmap='gray')
plt.title(f"Label: Validation")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(test_img, cmap='gray')
plt.title(f"Label: Test")
plt.axis('off')

plt.tight_layout()
plt.show()