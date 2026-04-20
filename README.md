# Neural Network Builder

A desktop GUI application for building, training, and analyzing feedforward neural networks from scratch using pure NumPy. No deep learning frameworks required.

---

## Features

- Design custom network architectures with any number of layers and neurons
- Train on randomly generated data or your own CSV files
- Three activation functions: ReLU, Sigmoid, Tanh
- Mini-batch gradient descent with backpropagation
- Live training progress and metrics (MSE, MAE)
- Visualize training loss curves and prediction plots
- Save and load trained networks as JSON

---

## Project Structure

```
PERCEPTRON/
├── perceptron.py        # Main application entry point
├── neural_network.py    # Core neural network implementation
├── gui_components.py    # Tkinter GUI tabs and widgets
└── data/
    └── diabetes.csv     # Sample dataset
```

---

## Requirements

- Python 3.8+
- numpy
- matplotlib
- tkinter (included with standard Python on most platforms)

Install dependencies:

```bash
pip install numpy matplotlib
```

> On Linux, if tkinter is missing: `sudo apt install python3-tk`

---

## How to Run

```bash
python perceptron.py
```

---

## Usage

The app has four tabs:

**1. Data**
Load a CSV file or generate random data. CSV format must have feature columns followed by a single integer label column (0-indexed classes).

**2. Network Designer**
Enter layer sizes as comma-separated values (e.g. `4,64,32,2`). The first number must match your feature count and the last must match your class count. Choose an activation function and learning rate, then click **Create Network**.

**3. Training**
Set the number of epochs and batch size, then click **Start Training**. Results (loss, MSE, MAE) are shown after training completes.

**4. Analysis**
Plot the training loss curve, view predictions vs actuals, or view a full performance summary. Save your trained network via **File → Save Network**.

---

## CSV Format

The last column must be the class label (integer, zero-indexed):

```
glucose,bmi,age,pregnancies,diabetes
148,33.6,50,6,1
85,26.6,31,0,0
```

---

## Save / Load Networks

Trained networks are saved as `.json` files containing the architecture, weights, biases, and activation type. Load them later via **File → Load Network** to resume evaluation without retraining.

---

## License

MIT
