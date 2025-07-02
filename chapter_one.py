"""
Given the number of reservations, Find the Linear regression between that and the 
number of pizzas that will be ordered
"""
import numpy as np


def predict(x_array: np.array, weight: float) -> np.array:
    """Makes a series of predictions based on the input array and weight."""
    return x_array * weight


def loss(x_array: np.array, y_array: np.array, weight: float) -> float:
    """
    Calculate the loss/error between the predicted and actual values.
    The y_array are the ground truth values we are trying to predict.
    """
    error = predict(x_array, weight) - y_array
    squared_error = error**2
    mean_squared_error = np.mean(squared_error)
    return mean_squared_error


def train(
    x_array: np.array, y_array: np.array, iterations: int, learning_rate: float
) -> float:
    """NOTE:
    The learning rate is a small number that determines how much the weight
    is updated during the training.
    """
    weight = 0
    for i in range(iterations):
        current_loss = loss(x_array, y_array, weight)
        print(f"Current Iteration: {i} => Current Loss: {current_loss:.4f}")
        if loss(x_array, y_array, weight + learning_rate) < current_loss:
            weight += learning_rate
        elif loss(x_array, y_array, weight - learning_rate) < current_loss:
            weight -= learning_rate
        else:
            return weight
    raise TimeoutError(f"Couldn't converge within {iterations} iterations")


def main():
    x_array, y_array = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
    iterations = 186
    learning_rate = 0.01
    num_reservations = input("Enter the number of reservations to predict pizzas for: \n")
    weight = train(x_array, y_array, iterations, learning_rate)
    print(f"Found the linear regression weight: {weight:.4f}")
    print(f"Predicted number of pizzas are: {(float(num_reservations) * weight):.4f}")

if __name__ == "__main__":
    main()