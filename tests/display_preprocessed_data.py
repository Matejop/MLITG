import matplotlib.pyplot as plt
import orjson
import gzip

filepath="c:/Users/matej/source/MLITG/data/mnist_preprocessed.gz"
data = orjson.loads(gzip.open(filepath).read())
training_data = list(zip(
    data["training_x"],
    data["training_y"]
))
validation_data = list(zip(
    data['validation_x'],
    data['validation_y']
))
test_data = list(zip(
    data['testing_x'],
    data['testing_y']
))

example_train = training_data[0] # You can choose any (in range ofc)
example_val = validation_data[0]
example_test = test_data[0]

print(f"Train data size: {len(training_data)}")
print(f"Validation data size: {len(validation_data)}")
print(f"Test data size: {len(test_data)}")

#print(f"First training input shape: {example_test[0].shape}")
#print(f"First training label shape: {example_test[1].shape}")
#print(f"First training label vector (one-hot):\n{example_test[1]}")

print(f"First validation label: {example_val[1]}")
print(f"First test label: {example_train[1]}")
print(f"Vstupní data: {example_train[0]}")

# Visualization
train_img = []
val_img = []
test_img = []
for i in range(28):
    train_img.append(example_train[0][28 * i: 28 * (i + 1)])
    val_img.append(example_test[0][28 * i: 28 * (i + 1)])
    test_img.append(example_val[0][28 * i: 28 * (i + 1)])

#train_img = example_train[0].reshape(28,28)
#val_img = example_val[0].reshape(28,28)
#test_img = example_test[0].reshape(28,28)

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