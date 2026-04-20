import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from neural_network import NeuralNetwork


class DataFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        title = ttk.Label(self, text="Data Management", style="Title.TLabel")
        title.pack(pady=10)

        gen_frame = ttk.LabelFrame(self, text="Generate Sample Data", padding=10)
        gen_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(gen_frame, text="Samples:").grid(row=0, column=0, sticky="w")
        self.samples_var = tk.StringVar(value="1000")
        ttk.Entry(gen_frame, textvariable=self.samples_var, width=15).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(gen_frame, text="Input Features:").grid(row=1, column=0, sticky="w")
        self.features_var = tk.StringVar(value="10")
        ttk.Entry(gen_frame, textvariable=self.features_var, width=15).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(gen_frame, text="Output Classes:").grid(row=2, column=0, sticky="w")
        self.classes_var = tk.StringVar(value="3")
        ttk.Entry(gen_frame, textvariable=self.classes_var, width=15).grid(row=2, column=1, sticky="w", padx=5)

        ttk.Button(gen_frame, text="Generate Random Data", command=self.generate_data).grid(row=3, column=0, columnspan=2, pady=10)

        load_frame = ttk.LabelFrame(self, text="Load Data from File", padding=10)
        load_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(load_frame, text="Load CSV File", command=self.load_csv).pack(fill="x")

        info_frame = ttk.LabelFrame(self, text="Data Information", padding=10)
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.info_text = tk.Text(info_frame, height=12, width=50, bg="#f9f9f9")
        self.info_text.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        self.info_text.config(yscrollcommand=scrollbar.set)

        self.update_info()

    def generate_data(self):
        try:
            n_samples = int(self.samples_var.get())
            n_features = int(self.features_var.get())
            n_classes = int(self.classes_var.get())

            X = np.random.randn(n_samples, n_features)
            y = np.eye(n_classes)[np.random.randint(0, n_classes, n_samples)]

            split = int(0.8 * n_samples)
            self.app.training_data = (X[:split], y[:split])
            self.app.test_data = (X[split:], y[split:])

            self.app.status_var.set(f"Generated {n_samples} samples ({split} train, {n_samples - split} test)")
            self.update_info()
            messagebox.showinfo("Success", "Random data generated successfully!")
        except ValueError:
            messagebox.showerror("Error", "Invalid input values")

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            try:
                data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
                X = data[:, :-1]
                y_raw = data[:, -1].astype(int)

                n_classes = int(np.max(y_raw)) + 1
                y = np.eye(n_classes)[y_raw]

                split = int(0.8 * len(X))
                self.app.training_data = (X[:split], y[:split])
                self.app.test_data = (X[split:], y[split:])

                self.app.status_var.set(f"Loaded data: {len(X)} samples")
                self.update_info()
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")

    def update_info(self):
        self.info_text.delete(1.0, tk.END)

        if self.app.training_data is None:
            self.info_text.insert(tk.END, "No data loaded.\n\nGenerate or load data to begin.")
        else:
            X_train, y_train = self.app.training_data
            X_test, y_test = self.app.test_data

            info = f"Training Data:\n"
            info += f"  Samples: {len(X_train)}\n"
            info += f"  Features: {X_train.shape[1]}\n"
            info += f"  Classes: {y_train.shape[1]}\n\n"
            info += f"Test Data:\n"
            info += f"  Samples: {len(X_test)}\n"
            info += f"  Features: {X_test.shape[1]}\n"
            info += f"  Classes: {y_test.shape[1]}\n\n"
            info += f"Data Statistics:\n"
            info += f"  X mean: {np.mean(X_train):.4f}\n"
            info += f"  X std: {np.std(X_train):.4f}\n"
            info += f"  X min: {np.min(X_train):.4f}\n"
            info += f"  X max: {np.max(X_train):.4f}"

            self.info_text.insert(tk.END, info)


class NetworkDesignerFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        title = ttk.Label(self, text="Network Architecture Designer", style="Title.TLabel")
        title.pack(pady=10)

        config_frame = ttk.LabelFrame(self, text="Network Configuration", padding=10)
        config_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(config_frame, text="Layer Sizes (comma-separated):").grid(row=0, column=0, sticky="w")
        self.layers_var = tk.StringVar(value="10,64,32,3")
        ttk.Entry(config_frame, textvariable=self.layers_var, width=40).grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Label(config_frame, text="Activation Function:").grid(row=1, column=0, sticky="w")
        self.activation_var = tk.StringVar(value="relu")
        ttk.Combobox(config_frame, textvariable=self.activation_var, values=["relu", "sigmoid", "tanh"], state="readonly", width=37).grid(row=1, column=1, sticky="ew", padx=5)

        ttk.Label(config_frame, text="Learning Rate:").grid(row=2, column=0, sticky="w")
        self.lr_var = tk.StringVar(value="0.01")
        ttk.Entry(config_frame, textvariable=self.lr_var, width=40).grid(row=2, column=1, sticky="ew", padx=5)

        ttk.Button(config_frame, text="Create Network", command=self.create_network).grid(row=3, column=0, columnspan=2, pady=10)

        config_frame.columnconfigure(1, weight=1)

        info_frame = ttk.LabelFrame(self, text="Network Information", padding=10)
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.arch_text = tk.Text(info_frame, height=15, width=50, bg="#f9f9f9")
        self.arch_text.pack(fill="both", expand=True)

        self.refresh()

    def create_network(self):
        try:
            layers = [int(x.strip()) for x in self.layers_var.get().split(",")]
            activation = self.activation_var.get()
            lr = float(self.lr_var.get())

            if len(layers) < 2:
                messagebox.showerror("Error", "Network must have at least 2 layers")
                return

            self.app.network = NeuralNetwork(layers, activation, lr)
            self.app.status_var.set("Network created successfully")
            self.refresh()
            messagebox.showinfo("Success", "Network created!")
        except ValueError:
            messagebox.showerror("Error", "Invalid configuration values")

    def refresh(self):
        self.arch_text.delete(1.0, tk.END)

        if self.app.network is None:
            self.arch_text.insert(tk.END, "No network created yet.\n\nConfigure the parameters and click 'Create Network' to get started.")
        else:
            info = f"Network Architecture:\n{'=' * 40}\n\n"
            info += f"Activation: {self.app.network.activation_type}\n"
            info += f"Learning Rate: {self.app.network.learning_rate}\n\n"
            info += f"Layers:\n"

            total_params = 0
            for i, size in enumerate(self.app.network.architecture):
                info += f"  Layer {i}: {size} neurons\n"

                if i < len(self.app.network.architecture) - 1:
                    params = size * self.app.network.architecture[i + 1] + self.app.network.architecture[i + 1]
                    total_params += params

            info += f"\nTotal Parameters: {total_params}"
            self.arch_text.insert(tk.END, info)


class TrainingFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        title = ttk.Label(self, text="Network Training", style="Title.TLabel")
        title.pack(pady=10)

        config_frame = ttk.LabelFrame(self, text="Training Configuration", padding=10)
        config_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(config_frame, text="Epochs:").grid(row=0, column=0, sticky="w")
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(config_frame, textvariable=self.epochs_var, width=15).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(config_frame, text="Batch Size:").grid(row=1, column=0, sticky="w")
        self.batch_var = tk.StringVar(value="32")
        ttk.Entry(config_frame, textvariable=self.batch_var, width=15).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Button(config_frame, text="Start Training", command=self.train_network).grid(row=2, column=0, columnspan=2, pady=10)

        self.progress = ttk.Progressbar(config_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)

        config_frame.columnconfigure(1, weight=1)

        results_frame = ttk.LabelFrame(self, text="Training Results", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.results_text = tk.Text(results_frame, height=12, width=50, bg="#f9f9f9")
        self.results_text.pack(fill="both", expand=True)

    def train_network(self):
        if self.app.network is None:
            messagebox.showerror("Error", "Create a network first!")
            return

        if self.app.training_data is None:
            messagebox.showerror("Error", "Load or generate data first!")
            return

        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_var.get())

            X_train, y_train = self.app.training_data
            X_test, y_test = self.app.test_data

            self.progress.start()
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Training in progress...\n")
            self.update()

            losses = self.app.network.train(X_train, y_train, epochs, batch_size)

            train_mse, train_mae = self.app.network.evaluate(X_train, y_train)
            test_mse, test_mae = self.app.network.evaluate(X_test, y_test)

            self.app.results = {
                "losses": losses,
                "train_mse": train_mse,
                "train_mae": train_mae,
                "test_mse": test_mse,
                "test_mae": test_mae
            }

            self.progress.stop()

            results = f"Training Complete!\n{'=' * 40}\n\n"
            results += f"Epochs: {epochs}\n"
            results += f"Final Loss: {losses[-1]:.6f}\n\n"
            results += f"Training Metrics:\n"
            results += f"  MSE: {train_mse:.6f}\n"
            results += f"  MAE: {train_mae:.6f}\n\n"
            results += f"Test Metrics:\n"
            results += f"  MSE: {test_mse:.6f}\n"
            results += f"  MAE: {test_mae:.6f}"

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results)

            self.app.status_var.set(f"Training complete - Test MSE: {test_mse:.6f}")
            messagebox.showinfo("Success", "Training completed successfully!")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid training parameters:\n{str(e)}")
        finally:
            self.progress.stop()


class AnalysisFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        title = ttk.Label(self, text="Network Analysis & Visualization", style="Title.TLabel")
        title.pack(pady=10)

        button_frame = ttk.Frame(self)
        button_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(button_frame, text="Plot Loss Curve", command=self.plot_loss).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Plot Predictions", command=self.plot_predictions).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Show Summary", command=self.show_summary).pack(side="left", padx=5)

        self.plot_frame = ttk.LabelFrame(self, text="Visualization", padding=10)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = None

    def plot_loss(self):
        if not self.app.results or "losses" not in self.app.results:
            messagebox.showerror("Error", "Train a network first!")
            return

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8, 4))
        losses = self.app.results["losses"]
        ax.plot(losses, linewidth=2, color="#2E86AB")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss (MSE)", fontsize=10)
        ax.set_title("Training Loss Curve", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_predictions(self):
        if self.app.network is None or self.app.test_data is None:
            messagebox.showerror("Error", "Train a network first!")
            return

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        X_test, y_test = self.app.test_data
        predictions = self.app.network.predict(X_test)

        fig, ax = plt.subplots(figsize=(8, 4))
        samples = min(100, len(X_test))
        ax.scatter(range(samples), y_test[:samples, 0], alpha=0.6, label="Actual", s=40)
        ax.scatter(range(samples), predictions[:samples, 0], alpha=0.6, label="Predicted", s=40)
        ax.set_xlabel("Sample", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_title("Predictions vs Actual (First Output)", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_summary(self):
        if not self.app.results:
            messagebox.showerror("Error", "Train a network first!")
            return

        summary = "Network Performance Summary\n" + "=" * 50 + "\n\n"
        summary += f"Network Architecture: {self.app.network.architecture}\n"
        summary += f"Activation: {self.app.network.activation_type}\n"
        summary += f"Learning Rate: {self.app.network.learning_rate}\n\n"
        summary += f"Training Loss: {self.app.results['losses'][-1]:.6f}\n\n"
        summary += f"Train Metrics:\n"
        summary += f"  MSE: {self.app.results['train_mse']:.6f}\n"
        summary += f"  MAE: {self.app.results['train_mae']:.6f}\n\n"
        summary += f"Test Metrics:\n"
        summary += f"  MSE: {self.app.results['test_mse']:.6f}\n"
        summary += f"  MAE: {self.app.results['test_mae']:.6f}\n"

        messagebox.showinfo("Performance Summary", summary)