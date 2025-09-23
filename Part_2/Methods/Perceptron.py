"""

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename

class Layer:
    def __init__(self, number_of_nodes, inputs, activation, learning_rate=0.001):
        self.weights = np.random.rand(number_of_nodes)
        self.biases = np.random.rand(number_of_nodes)
        self.activ = activation
        self.lr = learning_rate
        self.inputs = inputs

    def forward(self):
        z = self.inputs @ self.weights + self.biases
        return Z

class Data_options:

    def __init__(self,dataset="Training"):
        self.dataset = dataset.lower()
        assert (self.dataset == "training") or (self.dataset == "live"), "Unknown data"
        data_path = "../Data/"
        if self.dataset == "training":
            self.train_x = pd.read_csv(data_path+"train_in.csv")
            self.train_y = pd.read_csv(data_path+"train_out.csv")
            self.test_x = pd.read_csv(data_path+"test_in.csv")
            self.test_y = pd.read_csv(data_path+"test_out.csv")
        else:
            user_file = askopenfilename()
            self.test_x = pd.read_csv(user_file)

    def transform_to_image(self,input,output,
                           prediction=None,prediction_probability=None,
                           index=0):
        """
        Input assumed to be flattened row by 256 columns
        Default index is first in list, otherwise you can set it to any particular index you know of or "Random"
        need both input and output to see what it should be
        you can add the predicted category as well as its likelihood.
        If a predicted value is given the output can be set to None
        """
        if index.lower() == "random":
            index = np.random.randint(input.shape[0])
        row = input.iloc[index].values
        image = row.reshape(16,16)
        plt.imshow(image,cmap="gray")
        if prediction == None:
            plt.title(f"Number {output.iloc[index].values[0]}")
        else:
            plt.title(f"Predicted number {prediction.iloc[index].values[0]}, "
                      f"at {prediction_probability.iloc[index].values[0]}% probability")
        plt.show()

    def get_data(self):
        if self.dataset == "training":
            return zip(self.train_x, self.train_y, self.test_x, self.test_y)
        else:
            return self.test_x

class functions:
    def __init__(self):
        h=0

    def relu(self,input):
        output = max(0,input)
        return output

    def loss(self,prediction,true_value):
        loss = abs(true_value-prediction)
        return loss

    def train(self):
        h=0

    def fit(self):
        h=0

    def predict(self):
        h=0

def main():
    func=functions()
    data_class=Data_options()
    data_class.transform_to_image()
    number_of_layers=2
    model = 0
    for layer_num in range(number_of_layers):
        model.add(Layer(num_nodes,inputs=model[-1], activation = func.relu()))
    func.fit(model)


main()