"""

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename



class Data_options:

    def __init__(self,dataset="Training"):
        self.dataset = dataset.lower()
        if self.dataset != "training" and self.dataset != "live":
            raise ValueError("Unknown data")
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

    def softmax(self,x, temperature=1.0):
        e = np.array(x) / temperature
        e -= e.max(axis=1, keepdims=True)
        e = np.exp(e)
        dist = e / np.sum(e, axis=1, keepdims=True)
        return dist

    def accuracy(self, predictions, true_classes):
        matrix = np.c_[predictions,true_classes,np.zeros(len(predictions))]
        mask = matrix[:,0] == matrix[:,1]
        matrix[mask,2] = 1
        accuracy = sum(matrix[:,2]==1)/len(matrix[:,2]) * 100
        # print(f"Accuracy is : {accuracy:.2f}%")
        return round(accuracy,4)

    def multiclass_cross_entropy_loss(self,scores,true_value):

        num_classes = scores.shape[1]
        num_samples = scores.shape[0]
        scores = self.softmax(scores)
        epsilon = 1e-10
        NLL = -np.log(scores[np.arange(num_samples),true_value.flatten()]+epsilon)
        loss = np.mean(NLL)
        return round(loss,4)



class Perceptron(object):

    def __init__(self, learning_rate = 0.01, epochs = 50):
        self.lr = learning_rate
        self.epochs = epochs

    def relu(self,input):
        output = np.maximum(0,input)
        return output

    def weighted_sum(self,input):
        return np.dot(input,self.weights.T) + self.bias

    def fit(self,X,y,verbose=True,update="gd",weight_initializer="random",bias_initializer="random", **kwargs):

        initializers = ["random", "ones", "zeros", "normal"]
        # assert update rule is perceptron rule pr or gradient descent gd
        if update != "pr" and update != "gd":
            raise ValueError("Pick update rule == gd or pr")
        if weight_initializer not in initializers:
            raise ValueError("Weight initializer is invalid")
        if bias_initializer not in initializers:
            raise ValueError("Bias initializer is invalid")
        self.n_classes = int(y.max()) + 1
        if weight_initializer == "random":
            self.weights = np.random.rand(self.n_classes,X.shape[1])
        elif weight_initializer == "ones":
            self.weights = np.ones(self.n_classes,X.shape[1])
        elif weight_initializer == "zeros":
            self.weights = np.zeros(self.n_classes,X.shape[1])
        elif weight_initializer == "normal":
            self.weights = np.random.randn(self.n_classes,X.shape[1])
        if bias_initializer == "random":
            self.bias = np.random.rand(self.n_classes)
        elif bias_initializer == "ones":
            self.bias = np.ones(self.n_classes)
        elif bias_initializer == "zeros":
            self.bias = np.zeros(self.n_classes)
        elif bias_initializer == "normal":
            self.bias = np.random.randn(self.n_classes)

        history={}
        func=Functions()
        train_loss = []
        train_acc = []
        for _ in range(self.epochs):
            scores, predictions = self.predict(X)
            scores = func.softmax(scores)
            for xi, yi, score, y_pred in zip(X,y,scores,predictions):
                if update == "pr":
                    if y_pred != yi:
                        self.weights[yi] += self.lr * xi
                        self.bias[yi] += self.lr
                        self.weights[y_pred] -= self.lr * xi
                        self.bias[y_pred] -= self.lr
                elif update == "gd":
                    self.weights[yi] -= self.lr*(score[yi]-1)*xi
                    self.bias[yi] -= self.lr*(score[yi]-1)

            loss = func.multiclass_cross_entropy_loss(scores,y)
            accuracy = func.accuracy(predictions,y)
            train_loss.append(loss)
            train_acc.append(accuracy)
            if verbose == True:
                print(f"Epoch {_} \n"
                      f"Accuracy: {accuracy}\n"
                      f"Loss: {loss}")
        history.update({"Training Loss": train_loss, "Training Accuracy": train_acc})
        return history

    def predict(self,input):
        scores = self.weighted_sum(input)
        if scores.ndim ==1:
            return scores, (np.argmax(scores))
        return scores, np.argmax(scores,axis=1)

def plot(history,title=None):
    fig, ax1 = plt.subplots()
    color = "tab:blue"
    ax1.plot(history["Training Loss"], color=color, label="Training Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    color = "tab:red"
    ax2 = ax1.twinx()
    ax2.plot(history["Training Accuracy"], color=color, label="Training Accuracy")
    ax2.set_ylabel("Accuracy", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    plt.title(title)
    fig.tight_layout()
    plt.show()

def plot_multiple_histories(histories,title=None):
    for history in histories:
        fig, ax1 = plt.subplots()
        color = "tab:blue"
        ax1.plot(history["Training Loss"], color=color, label="Training Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        color = "tab:red"
        ax2 = ax1.twinx()
        ax2.plot(history["Training Accuracy"], color=color, label="Training Accuracy")
        ax2.set_ylabel("Accuracy", color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        fig.tight_layout()
        plt.title
        plt.show()

def main(verbose=False,grid_search=False):

    func=Functions()
    data_class=Data_options()
    data = data_class.get_data()

    # data_class.transform_to_image()

    initializers=["random","ones","zeros","normal"]
    updaters=["pr","gd"]
    learning_rates=[1,0.1,0.001,0.0001,0.00001]
    epochs=[5,10,25,50,100,250,500]
    number_of_runs = 10

    X = data["Train_in"].to_numpy(dtype=float)
    y = data["Train_out"].to_numpy(dtype=int)
    test_x = data["Test_in"].to_numpy(dtype=float)
    test_y = data["Test_out"].to_numpy(dtype=int)

    if grid_search:
        for bias_init in initializers:
            for weight_init in initializers:
                for updater in updaters:
                    for lr in learning_rates:
                        for epoch in epochs:
                            percep = Perceptron(learning_rate=lr,epochs=epoch)
                            history = percep.fit(X, y,
                                                 update=updater,
                                                 weight_initializer = weight_init,
                                                 bias_initializer = bias_init)
                            predictions_batched = [percep.predict(x) for x in test_x]
                            scores = np.array([r[0] for r in predictions_batched])
                            y_pred = np.array([r[1] for r in predictions_batched])
                            loss = func.multiclass_cross_entropy_loss(scores, test_y)
                            accuracy = func.accuracy(y_pred, test_y)
                            if verbose == True:
                                print(f"Test set : \n"
                                      f"Accuracy : {accuracy}\n"
                                      f"Loss : {loss}")

    else:
        histories=[]
        for num_run in range(number_of_runs):
            #best parameters
            epoch=50
            lr=0.01
            weight_init="random"
            bias_init = "random"
            updater = "gd"
            percep = Perceptron(learning_rate=lr, epochs=epoch)
            history = percep.fit(X=X, y=y,
                                 update=updater,
                                 weight_initializer=weight_init,
                                 bias_initializer=bias_init)
            predictions_batched = [percep.predict(x) for x in test_x]
            scores = np.array([r[0] for r in predictions_batched])
            y_pred = np.array([r[1] for r in predictions_batched])
            loss = func.multiclass_cross_entropy_loss(scores, test_y)
            accuracy = func.accuracy(y_pred, test_y)
            if verbose == True:
                print(f"Test set : \n"
                      f"Accuracy : {accuracy}\n"
                      f"Loss : {loss}")
            histories.append(history)
        plot_multiple_histories(histories,title="Best parameters 10 runs")


main()