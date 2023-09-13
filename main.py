# Data Model
import numpy as np


def nn_for_or_gate():
    X1 = np.array([0, 0, 1, 1])
    X2 = np.array([0, 1, 0, 1])
    Y = np.array([0, 1, 1, 1])
    W = np.array([0.1, 0.3])

    neurons = 2
    epoch = 4
    learning_rate = 0.2
    threshold = 0.5
    number_of_test_case = 4

    for i in range(epoch):
        for j in range(0, number_of_test_case):
            k = X1[j] * W[0] + X2[j] * W[1]
            output = 0
            w_del_1 = 0
            w_del_2 = 0
            if k > threshold:
                output = 1
            if output != Y[j]:
                w_del_1 = learning_rate * X1[j] * (Y[j] - output)
                w_del_2 = learning_rate * X2[j] * (Y[j] - output)
                W[0] += w_del_1
                W[1] += w_del_2
            print("the stat of iteration no. ", j + 1, ": ", k, output, w_del_1, w_del_2)

        print("Epoch no. ", i + 1, " -- ", W[0], W[1])


def multiply_matrices(x, w, test_case):
    ans = 0
    for i in range(3):
        if test_case == 0:
            ans += x[i] * w[i]
        else:
            ans -= x[i] * w[i]

    return ans


def perception_model():
    # here Wo = -10 and the w = [ 1, 1]
    w = np.array([-10, 1, 1])

    # we have 1 to all the class elements to include the constant value the matrix
    x_class_1 = np.array([[1, 1, 8], [1, 6, 7], [1, 8, 5], [1, 11, 4]])
    x_class_2 = np.array([[1, 2, 4], [1, 3, 3], [1, 6, 2], [1, 11, 2]])

    learning_rate = 0.2

    epoch = 4
    iterations = 4

    for i in range(epoch):
        error = np.array([0, 0, 0])
        misclassified_element = 0;
        for j in range(iterations):
            if multiply_matrices(x_class_1[j], w, 0) < 0:
                misclassified_element += 1
                for k in range(3):
                    error[k] += x_class_1[j][k]

        for j in range(iterations):
            if multiply_matrices(x_class_2[j], w, 1) < 0:
                misclassified_element += 1
                for k in range(3):
                    error[k] -= x_class_2[j][k]

        for j in range(3):
            w[j] += learning_rate * error[j]

        print("For epoch no.", i + 1, " : ", "Wo = ", w[0], " W1 = ", w[1], " W2 = ", w[2], " no. of mis-classified "
                                                                                            "element: ",
              misclassified_element)


def multiply(x_class, w, h):
    ans = 0
    for i in range(2):
        ans += w[i][h] * x_class[h][i]
    return ans


def back_propagation_neural_network():
    x_class = np.array([[0.4, 0.3, 0.6], [-0.7, -0.5, 0.1]])
    # x_class_2 = np.array([-0.7, -0.5, 0.1])
    target = np.array([0.1, 0.05, 0.1])

    w = np.array([[0.1, 0.2], [0.2, 0.4]])
    # w_2 = np.array([0.2, 0.4])
    v = np.array([0.2, 0.4])
    y = 0

    learning_rate = 3
    epoch = 1
    iteration = 3

    for i in range(epoch):
        for j in range(iteration):
            # for O_h_1
            l_1 = multiply(x_class, w, 0)
            l_2 = multiply(x_class, w, 1)

            o_h = np.array([1 / (1 + np.exp(-l_1)), 1 / (1 + np.exp(-l_2))])

            y = 1 / (1 + np.exp(-(o_h[0] * v[0] + o_h[1] * v[1])))
            print()
            print("Actual Value = ", y, " target  = ", target[j], " diff = ", y - target[j])
            if y != target[j]:
                v[0] += learning_rate * y * (1 - y) * (target[j] - y) * o_h[0]
                v[1] += learning_rate * y * (1 - y) * (target[j] - y) * o_h[1]

                w[0][0] += learning_rate * y * (1 - y) * (target[j] - y) * v[0] * o_h[0] * (1 - o_h[0]) * x_class[0][0]
                w[0][1] += learning_rate * y * (1 - y) * (target[j] - y) * v[0] * o_h[0] * (1 - o_h[0]) * x_class[1][0]

                w[1][0] += learning_rate * y * (1 - y) * (target[j] - y) * v[1] * o_h[1] * (1 - o_h[1]) * x_class[0][0]
                w[1][1] += learning_rate * y * (1 - y) * (target[j] - y) * v[1] * o_h[1] * (1 - o_h[1]) * x_class[1][0]

            print("V11 = ", v[0], "V21 = ", v[1])
            print("W11 = ", w[0][0], "W12 = ", w[0][1], "W21 = ", w[1][0], "W22 = ", w[1][1])

            l_1 = multiply(x_class, w, 0)
            l_2 = multiply(x_class, w, 1)

            o_h = np.array([1 / (1 + np.exp(-l_1)), 1 / (1 + np.exp(-l_2))])

            y = 1 / (1 + np.exp(-(o_h[0] * v[0] + o_h[1] * v[1])))
            print("After Correction: ")
            print("Actual Value = ", y, " target  = ", target[j], " diff = ", y - target[j])


# perception_model()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def back_propagation_neural_network_1():
    x_class = np.array([[0.4, -0.7], [0.3, 0.5], [0.6, 0.1]])
    target = np.array([[0.1], [0.05], [0.1]])

    # Initialize weights
    w = np.array([[0.1, 0.2], [0.2, 0.4]])
    v = np.array([0.2, 0.3])

    learning_rate = 1
    epochs = 1

    for epoch in range(epochs):
        total_error = 0

        for i in range(len(x_class)):
            # Forward propagation
            l_1 = np.dot(x_class[i], w)
            o_h = sigmoid(l_1)
            y = sigmoid(np.dot(o_h, v))

            # Calculate error
            error = target[i] - y
            total_error += error

            # Backpropagation
            delta_y = error * sigmoid_derivative(y)
            delta_o_h = delta_y * v * sigmoid_derivative(o_h)

            v += learning_rate * delta_y * o_h
            w += learning_rate * np.outer(delta_o_h, x_class[i])

        # Print error for each epoch
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Error = {total_error[0]} : y = {y}")


    print("Training complete.")

    # Test the trained network
    for i in range(len(x_class)):
        l_1 = np.dot(x_class[i], w)
        o_h = sigmoid(l_1)
        y = sigmoid(np.dot(o_h, v))
        print(f"Input: {x_class[i]}, Predicted Output: {y}")


back_propagation_neural_network_1()
# back_propagation_neural_network()
