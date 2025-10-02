# LoRA (Low-Rank Adaptation) Fine-Tuning Demo

This repository provides a comprehensive demonstration of Low-Rank Adaptation (LoRA) fine-tuning concepts using Ollama with small language models. The goal is to offer a clear, hands-on learning experience for understanding and implementing parameter-efficient fine-tuning (PEFT) techniques.

## Table of Contents
- [Introduction to LoRA](#introduction-to-lora)
- [Demo Overview](#demo-overview)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction to LoRA

**Low-Rank Adaptation (LoRA)** is a revolutionary technique for efficiently fine-tuning large language models (LLMs). Instead of retraining all the model's parameters (which is computationally expensive and memory-intensive), LoRA freezes the pre-trained model weights and injects small, trainable rank-decomposition matrices into each layer. This approach dramatically reduces the number of trainable parameters, often by over 99%, without sacrificing model performance.

> LoRA is based on the hypothesis that the change in weights during model adaptation has a low “intrinsic rank”. By decomposing the weight update matrix into two smaller matrices (ΔW = A × B), we can train a much smaller number of parameters while still achieving high-quality task-specific adaptation.

### Key Advantages

| Feature | Traditional Fine-Tuning | LoRA Fine-Tuning |
| :--- | :--- | :--- |
| **Trainable Parameters** | All model parameters (billions) | Only adapter matrices (millions) |
| **Memory Usage** | Very High (requires multiple GPUs) | Low (can run on consumer hardware) |
| **Storage Cost** | Full model copy for each task (GBs) | Small adapter file per task (MBs) |
| **Training Speed** | Slow (days or weeks) | Fast (hours) |
| **Inference Latency** | No extra latency | No extra latency (weights can be merged) |
| **Modularity** | Monolithic | Highly modular (swap adapters for tasks) |

## Demo Overview

This demo provides a simplified yet practical illustration of LoRA concepts. It includes:

- **Conceptual Explanations**: Clear, concise explanations of LoRA's mathematical foundation and benefits.
- **Practical Scripts**: A Python script (`src/simple_lora_demo.py`) that simulates the LoRA process and demonstrates parameter efficiency.
- **Jupyter Notebook**: An interactive notebook (`notebooks/01_lora_concepts.ipynb`) for a step-by-step conceptual walkthrough.
- **Ollama Integration**: Examples of how to create a custom Ollama model that mimics a LoRA-tuned model.
- **Setup Script**: An easy-to-use setup script (`setup.sh`) to prepare the environment.

## Getting Started

To get started with the demo, clone the repository and run the setup script.

```bash
# 1. Clone the repository
git clone https://github.com/kaljuvee/lora-demo.git
cd lora-demo

# 2. Run the setup script
# This will install Ollama, pull a base model, and run the concept demo.
./setup.sh
```

### Prerequisites

- **Linux/macOS**: The setup script is designed for Unix-like environments.
- **curl**: Required to download the Ollama installation script.
- **Python 3**: Required to run the demo scripts.

## Project Structure

```
lora-demo/
├── README.md                 # Main documentation
├── setup.sh                  # Environment setup script
├── requirements.txt          # Python dependencies (for reference)
├── notebooks/                # Jupyter demonstration notebooks
│   └── 01_lora_concepts.ipynb
├── src/                      # Source code
│   └── simple_lora_demo.py   # Core conceptual demo script
├── data/                     # Sample datasets
│   └── demo_dataset.json
├── models/                   # Directory for model files
│   └── Modelfile.lora-ml-tutor # Example Ollama Modelfile
└── ...
```

## How It Works

The core of this demonstration is the `simple_lora_demo.py` script, which provides a hands-on, text-based tour of LoRA concepts.

1.  **Conceptual Explanation**: It starts by printing a clear, high-level overview of LoRA.
2.  **Parameter Efficiency**: It calculates and compares the number of trainable parameters between full fine-tuning and LoRA, highlighting the massive reduction.
3.  **Simulated Training**: It walks through the steps of a simulated LoRA training process, from loading the base model to exporting the final adapter.
4.  **Ollama Modelfile**: It generates an example `Modelfile` that shows how you can use a custom system prompt and parameters to create a specialized model in Ollama, mimicking the outcome of a LoRA fine-tune.
5.  **Base Model Test**: It runs a quick test on the base Ollama model to ensure the environment is working correctly.

## Examples

### Running the Concept Demo

This is the easiest way to understand the core ideas.

```bash
python3 src/simple_lora_demo.py
```

### Using the Jupyter Notebook

For a more interactive and visual explanation, run the Jupyter notebook.

```bash
# Make sure you have jupyter installed (pip3 install jupyter)
jupyter notebook notebooks/01_lora_concepts.ipynb
```

### Creating a Custom Ollama Model

The demo generates a `Modelfile` that you can use to create your own specialized model with Ollama.

```bash
# Create the custom model using the generated Modelfile
ollama create lora-ml-tutor -f models/Modelfile.lora-ml-tutor

# Run your new custom model
ollama run lora-ml-tutor "Explain the difference between classification and regression."
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential changes or additions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

