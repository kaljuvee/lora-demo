# LoRA Research Notes

## Key Concepts from IBM Article

### What is LoRA?
- **Low-rank adaptation (LoRA)** is a technique used to adapt machine learning models to new contexts
- Adds lightweight pieces to the original model rather than changing the entire model
- Originally published by Edward Hu, Yelong Shen and collaborators in "LoRA: Low-Rank Adaptation Of Large Language Models"

### How LoRA Works
- **Freezes original weights** and parameters of the model
- **Adds low-rank matrices** on top of the original model
- Low-rank matrix adjusts for the weights of the original model to match desired use case
- Leverages smaller matrices (low-rank matrices) to make training extremely efficient
- Updates matrices A and B to track changes in pretrained weights using smaller matrices of rank r

### Key Technical Details
- Built on understanding that large models have low-dimensional structure
- High-rank matrix can be decomposed into two low-rank matrices (e.g., 4x4 → 4x1 and 1x4)
- Low-rank matrices are updated through gradient descent during fine-tuning
- Final model = base model weights + multiplied change matrix

### Key Parameters (HuggingFace PEFT)
- **r**: rank of update matrices (lower rank = fewer trainable parameters)
- **target_modules**: modules to apply LoRA update matrices (e.g., attention blocks)
- **lora_alpha**: LoRA scaling factor

### Advantages
- **Efficiency**: Significantly reduces trainable parameters and GPU memory requirements
- **Modularity**: Base model can be shared, switch tasks by replacing LoRA weight matrices
- **Performance**: No inference latency compared to fully fine-tuned model
- **Flexibility**: Can be combined with other techniques like prefix-tuning

### Tradeoffs
- **Information loss** during matrix decomposition
- Some details lost when reducing full weight matrix into smaller components
- Models may become less accurate due to reduced redundancy/robustness
- Rank of update matrices needs tuning to balance efficiency vs accuracy

## Sources
- IBM Think Topics: https://www.ibm.com/think/topics/lora


## Unsloth and Ollama Integration

### What is Unsloth?
Unsloth makes fine-tuning LLMs like Llama-3, Mistral, Phi-3 and Gemma 2x faster, uses 70% less memory, and with no degradation in accuracy. It provides integration with Google Colab for free GPU access during training.

### Key Features
- **Performance**: 2x faster fine-tuning with 70% less memory usage
- **Accuracy**: No degradation in model accuracy
- **Integration**: Direct export to Ollama format
- **Accessibility**: Free GPU access via Google Colab

### Workflow Overview
1. **Fine-tune model** using Unsloth with LoRA adapters
2. **Export to GGUF format** (llama.cpp compatible format)
3. **Deploy to Ollama** for local inference
4. **Use interactively** like ChatGPT locally

### Export Process
- Convert fine-tuned model to GGUF format
- Popular quantization: Q8_0 (8-bit) and q4_k_m (4-bit)
- Automatic integration with Ollama
- Supports various quantization methods

### Available Notebooks
- **Ollama Llama-3 Alpaca**: Main tutorial notebook
- **CSV/Excel Ollama Guide**: For structured data fine-tuning

### Technical Requirements
- Google Colab account for free GPU access
- Ollama installation for local deployment
- Compatible with various model architectures (Llama, Mistral, Phi, Gemma)

## Sources
- Unsloth Documentation: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama
