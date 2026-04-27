from MLL.model import Model
import tkinter as tk
from PIL import Image, ImageTk
from typing import List
import numpy as np

IMAGE_X = 28
IMAGE_Y = 28

class Display:
    def __init__(self, root, model: Model, images: List[List[int]], data: List[List[float]], index: int = 0):
        self.root = root
        self.root.title("Digit Recognizer")
        self.model = model
        self.images = images
        self.data = data        
        self.index = index
        self.current_img = None

        self.load_img()
        self.label_img = tk.Label(root, image=self.current_img) 
        self.label_img.pack()

        self.label_text = tk.Label(root, text="Click predict")
        self.label_text.pack()

        tk.Button(root, text="Predict", command=self.predict).pack(side="left")
        tk.Button(root, text="Next", command=self.next).pack(side="left")         

    def load_img(self):
        img_array = np.array(self.images[self.index], dtype='uint8').reshape(28, 28)
        img = Image.fromarray(img_array, mode="L")
        img = img.resize((280, 280), Image.NEAREST)
        self.current_img = ImageTk.PhotoImage(img)        

    def next(self):
        self.index = (self.index + 1) % len(self.images)
        self.load_img()
        self.label_img.config(image=self.current_img)
        self.label_text.config(text="Click predict")

    def predict(self):
        pred = self.model.classify(self.data[self.index])
        digit = 0
        confidence = pred[0]
        for i in range(len(pred)):
            if pred[i] > confidence:                
                digit = i
                confidence = pred[i]
        self.label_text.config(text=f"Prediction: {digit} ({confidence * 100}%)")

if __name__ == "__main__":    
    images = []
    data = []
    with open("c:/Users/matej/source/MLITG/src/tests/MNIST/mnist_test.csv") as f:
        file = f.read()
        file_lines = file.split("\n")
        file_lines.pop()
        for line in file_lines:
            split_line = line.split(",")
            images.append(split_line[1:])
            data.append([float(pixel) / 255 for pixel in split_line[1:]])
    model = Model.load("c:/Users/matej/source/MLITG/models/Model_57120b0b14.json")
    root = tk.Tk()
    app = Display(root, model, images, data)
    root.mainloop()