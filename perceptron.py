import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import json
import os
from datetime import datetime
from neural_network import NeuralNetwork
from gui_components import NetworkDesignerFrame, TrainingFrame, AnalysisFrame, DataFrame


class PerceptronApp(tk.Tk):
    """Main application window for the Perceptron neural network builder."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Perceptron - Neural Network Builder")
        self.geometry("1200x700")
        self.resizable(True, True)
        
        # Application state
        self.network = None
        self.training_data = None
        self.test_data = None
        self.results = {}
        
        # Configure style
        self.configure(bg="#f0f0f0")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Helvetica", 14, "bold"))
        style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create main container
        main_container = ttk.Frame(self)
        main_container.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill="both", expand=True)
        
        # Create tabs
        self.data_frame = DataFrame(self.notebook, self)
        self.notebook.add(self.data_frame, text="Data")
        
        self.designer_frame = NetworkDesignerFrame(self.notebook, self)
        self.notebook.add(self.designer_frame, text="Network Designer")
        
        self.training_frame = TrainingFrame(self.notebook, self)
        self.notebook.add(self.training_frame, text="Training")
        
        self.analysis_frame = AnalysisFrame(self.notebook, self)
        self.notebook.add(self.analysis_frame, text="Analysis")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken")
        status_bar.pack(side="bottom", fill="x", padx=5, pady=5)
        
    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Network", command=self.new_network)
        file_menu.add_command(label="Save Network", command=self.save_network)
        file_menu.add_command(label="Load Network", command=self.load_network)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        
    def new_network(self):
        """Create a new neural network."""
        self.network = None
        self.training_data = None
        self.test_data = None
        self.results = {}
        self.designer_frame.refresh()
        self.status_var.set("New network ready. Configure in Network Designer tab.")
        messagebox.showinfo("New Network", "Network configuration reset. Configure your network in the Network Designer tab.")
        
    def save_network(self):
        """Save the current network configuration and results."""
        if self.network is None:
            messagebox.showwarning("Warning", "No network to save. Create a network first.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            data = {
                "architecture": self.network.architecture,
                "weights": [w.tolist() for w in self.network.weights],
                "biases": [b.tolist() for b in self.network.biases],
                "activation": self.network.activation_type,
                "timestamp": datetime.now().isoformat()
            }
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.status_var.set(f"Network saved to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Network saved successfully!")
            
    def load_network(self):
        """Load a network configuration from file."""
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                self.network = NeuralNetwork(
                    layer_sizes=data["architecture"],
                    activation_type=data["activation"]
                )
                
                for i, (w, b) in enumerate(zip(data["weights"], data["biases"])):
                    self.network.weights[i] = np.array(w)
                    self.network.biases[i] = np.array(b)
                
                self.designer_frame.refresh()
                self.status_var.set(f"Network loaded from {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Network loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load network: {str(e)}")
    
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About Perceptron",
            "Perceptron v1.0\n\n"
            "A flexible artificial neural network (ANN) builder.\n"
            "Design, analyze, and optimize neural networks with ease.\n\n"
            "© 2025"
        )
    
    def show_documentation(self):
        """Show documentation."""
        messagebox.showinfo(
            "Documentation",
            "Perceptron User Guide:\n\n"
            "1. DATA: Load or generate training/test data\n"
            "2. NETWORK DESIGNER: Define network architecture\n"
            "3. TRAINING: Train your network with various optimizers\n"
            "4. ANALYSIS: Visualize results and performance metrics\n\n"
            "Activation Functions: ReLU, Sigmoid, Tanh\n"
            "Optimizers: SGD, Adam, RMSprop"
        )


if __name__ == "__main__":
    app = PerceptronApp()
    app.mainloop()
