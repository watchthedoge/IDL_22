"""

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename



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
            return {"Train_in" : self.train_x, "Train_out" : self.train_y, "Test_in" : self.test_x, "Test_out" : self.test_y}
        else:
            return {"Test_in": self.test_x}

class Functions:



    def accuracy(self, predictions, true_classes):

        total_num = len(predictions)
        total_accuracy=0
        for predicted_class_idx, predicted_class in enumerate(predictions):

            if predicted_class == true_classes[predicted_class_idx]:
                total_accuracy += 1

        accuracy = total_accuracy/total_num * 100
        print(f"Accuracy is : {accuracy:.2f}%")
        return accuracy

    def loss(self,prediction,true_value):
        loss = abs(true_value-prediction)
        return loss



class Perceptron(object):

    def __init__(self, learning_rate = 0.01, epochs = 10):
        self.lr = learning_rate
        self.epochs = epochs

    def relu(self,input):
        output = np.maximum(0,input)
        return output

    def weighted_sum(self,input):
        return np.dot(input,self.weights.T) + self.bias

    def fit(self,X,y):
        self.n_classes = int(y.max()) + 1
        self.weights = np.random.rand(self.n_classes,X.shape[1])
        self.bias = np.random.rand(self.n_classes)
        self.errors = []

        for _ in range(self.epochs):
            error = 0
            for xi, yi in zip(X,y):
                y_pred = self.predict(xi)
                if y_pred != yi:
                    self.weights[yi] += self.lr * xi
                    self.bias[yi] += self.lr
                    self.weights[y_pred] -= self.lr * xi
                    self.bias[y_pred] -= self.lr


    def predict(self,input):
        return np.argmax(self.weighted_sum(input))

def main():
    func=Functions()
    data_class=Data_options()
    data = data_class.get_data()
    # data_class.transform_to_image()
    number_of_layers=1
    percep = Perceptron()
    X = data["Train_in"].to_numpy(dtype=float)
    y = data["Train_out"].to_numpy(dtype=int)
    percep.fit(X,y)
    test_x = data["Test_in"].to_numpy(dtype=float)
    test_y = data["Test_out"].to_numpy(dtype=float)
    y_pred = np.array([percep.predict(x) for x in test_x])
    func.accuracy(y_pred,test_y)
    print(np.c_[y_pred,test_y])


main()