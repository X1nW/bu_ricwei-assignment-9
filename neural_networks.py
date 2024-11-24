import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1  # Input to hidden weights
        self.b1 = np.zeros((1, hidden_dim))  # Hidden layer biases
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1  # Hidden to output weights
        self.b2 = np.zeros((1, output_dim))  # Output layer biases

        # Placeholders for storing activations and gradients
        self.Z1 = None  # Pre-activation of hidden layer
        self.A1 = None  # Activation of hidden layer
        self.Z2 = None  # Pre-activation of output layer
        self.A2 = None  # Activation of output layer
        self.dW1 = None  # Gradient w.r.t W1
        self.db1 = None  # Gradient w.r.t b1
        self.dW2 = None  # Gradient w.r.t W2
        self.db2 = None  # Gradient w.r.t b2

    def activation(self, Z):
        if self.activation_fn == 'tanh':
            return np.tanh(Z)
        elif self.activation_fn == 'relu':
            return np.maximum(0, Z)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, A):
        if self.activation_fn == 'tanh':
            return 1 - A ** 2
        elif self.activation_fn == 'relu':
            return (A > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            return A * (1 - A)
        else:
            raise ValueError("Unsupported activation function")


    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activation(self.Z1)  # Hidden layer activation
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.tanh(self.Z2)  # Output layer activation (using tanh for binary classification)

        return self.A2

    def backward(self, X, y):
        m = X.shape[0]
        # TODO: compute gradients using chain rule
        dZ2 = (self.A2 - y) * (1 - self.A2 ** 2)
        # TODO: update weights with gradient descent

        # TODO: store gradients for visualization

        self.dW2 = np.dot(self.A1.T, dZ2) / m
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Propagate error back to hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_derivative(self.A1)

        # Compute gradients for W1 and b1
        self.dW1 = np.dot(X.T, dZ1) / m
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2
    
    def predict(self, X):
        A2 = self.forward(X)
        return np.sign(A2)

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.A1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title('Hidden Layer Feature Space')
    ax_hidden.set_xlabel('Neuron 1 Activation')
    ax_hidden.set_ylabel('Neuron 2 Activation')
    ax_hidden.set_zlabel('Neuron 3 Activation')
    # TODO: Hyperplane visualization in the hidden space
    h_range = np.linspace(-1, 1, 20)
    h1_grid, h2_grid, h3_grid = np.meshgrid(h_range, h_range, h_range)
    grid_hidden = np.c_[h1_grid.ravel(), h2_grid.ravel(), h3_grid.ravel()]
    Z2_grid = np.dot(grid_hidden, mlp.W2) + mlp.b2
    A2_grid = np.tanh(Z2_grid)
    boundary_indices = np.where(np.abs(A2_grid) < 0.05)[0]
    ax_hidden.scatter(grid_hidden[boundary_indices, 0],
                      grid_hidden[boundary_indices, 1],
                      grid_hidden[boundary_indices, 2],
                      c='green', alpha=0.1, marker='o')
    # TODO: Distorted input space transformed by the hidden layer

    # TODO: Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.predict(grid)
    Z = Z.reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], colors=['blue', 'red'], alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_title('Decision Boundary in Input Space')

    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    node_positions = {
        'x1': (0, 2),
        'x2': (0, -2),
        'h1': (2, 4),
        'h2': (2, 0),
        'h3': (2, -4),
        'y': (4, 0)
    }
    # Visualize gradients as arrows
    for node, (x, y_pos) in node_positions.items():
        ax_gradient.scatter(x, y_pos, s=1000, c='lightgray', edgecolors='black', zorder=3)
        ax_gradient.text(x, y_pos, node, fontsize=12, ha='center', va='center', zorder=4)
    grad_abs_W1 = np.abs(mlp.dW1)
    grad_abs_W2 = np.abs(mlp.dW2)
    max_grad = max(np.max(grad_abs_W1), np.max(grad_abs_W2), 1e-6)  # Avoid division by zero

    for i, input_node in enumerate(['x1', 'x2']):
        for j, hidden_node in enumerate(['h1', 'h2', 'h3']):
            x_start, y_start = node_positions[input_node]
            x_end, y_end = node_positions[hidden_node]
            grad = grad_abs_W1[i, j]
            grad_norm = grad / max_grad
            color_intensity = 1 - grad_norm  # Higher gradient, darker edge
            edge_color = (color_intensity, color_intensity, color_intensity)
            linewidth = 1 + 4 * grad_norm  # Line width between 1 and 5
            ax_gradient.plot([x_start, x_end], [y_start, y_end], c=edge_color, linewidth=linewidth)

    # Edges from hidden layer to output
    for i, hidden_node in enumerate(['h1', 'h2', 'h3']):
        x_start, y_start = node_positions[hidden_node]
        x_end, y_end = node_positions['y']
        grad = grad_abs_W2[i, 0]
        grad_norm = grad / max_grad
        color_intensity = 1 - grad_norm
        edge_color = (color_intensity, color_intensity, color_intensity)
        linewidth = 1 + 4 * grad_norm
        ax_gradient.plot([x_start, x_end], [y_start, y_end], c=edge_color, linewidth=linewidth)

    ax_gradient.set_xlim(-1, 5)
    ax_gradient.set_ylim(-5, 5)
    ax_gradient.set_title('Neural Network Gradients')
    ax_gradient.axis('off')


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)