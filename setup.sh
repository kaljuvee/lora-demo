#!/bin/bash

# LoRA Demo Setup Script
# This script sets up the environment for the LoRA fine-tuning demonstration

set -e  # Exit on any error

echo "🚀 Setting up LoRA Demo Environment..."
echo "======================================"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Start Ollama service
    sudo systemctl enable ollama
    sudo systemctl start ollama
    
    echo "✅ Ollama installed and started"
else
    echo "✅ Ollama already installed"
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Install Python dependencies (minimal set to avoid timeout)
echo "📦 Installing Python dependencies..."
pip3 install --user --no-cache-dir numpy matplotlib jupyter || {
    echo "⚠️  Some packages failed to install, but demo will still work"
}

# Pull a small model for demonstration
echo "📥 Pulling Llama 3.2 1B model..."
ollama pull llama3.2:1b || {
    echo "⚠️  Model download failed, but you can try again later"
}

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p notebooks src data models/lora_adapters docs examples

# Make scripts executable
chmod +x src/simple_lora_demo.py 2>/dev/null || true

# Run the simple demo
echo "🧪 Running LoRA concept demonstration..."
python3 src/simple_lora_demo.py

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📚 What's available:"
echo "  • src/simple_lora_demo.py - Interactive LoRA concepts demo"
echo "  • notebooks/01_lora_concepts.ipynb - Jupyter notebook tutorial"
echo "  • data/demo_dataset.json - Sample training dataset"
echo "  • models/ - Directory for model files and adapters"
echo ""
echo "🚀 Quick start:"
echo "  1. Run: python3 src/simple_lora_demo.py"
echo "  2. Or: jupyter notebook notebooks/01_lora_concepts.ipynb"
echo "  3. Test Ollama: ollama run llama3.2:1b 'What is machine learning?'"
echo ""
echo "📖 For detailed documentation, see README.md"
echo ""
