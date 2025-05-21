# Using Ollama for Lyric Generation

As an alternative to running LLM models with GGUF format directly with `llama-cpp`
by using the Python wrapper `llama-cpp-python`, `llama-cpp` hosted 
by [Ollama](https://github.com/ollama) can be used for lyric generation.

The advantages are
- usage of Ollama LLM models
- usage of several GPUs out-of-the-box

## Prerequisites

An Ollama instance up and running with the intended Ollama LLM model.

Installing the Python wrapper for the Ollama REST API:
```
pip install ollama
```

## Usage

For using Ollama, add the following command line arguments:
- `--ollama`
- `--model_path <Ollama model name>`, e.g. `--model_path gemma3:12b-it-q4_K_M`
 