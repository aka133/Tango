import tensorrt as trt # type: ignore
import torch
from transformers import AutoTokenizer
import tensorrt_llm # type: ignore
import time
from pathlib import Path

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def setup_engine():
    """Initialize TensorRT-LLM engine"""
    print(f"Building TensorRT engine for {model_name}...")
    
    # Initialize builder and network
    builder = tensorrt_llm.Builder()
    network = builder.create_network()
    
    # Configure builder
    config = builder.create_builder_config(
        max_batch_size=8,
        max_input_len=2048,
        max_output_len=512,
        name="tinyllama"
    )
    
    # Load model configuration
    config.load_pretrained(model_name)
    
    # Build and save engine
    engine_path = Path("tinyllama_engine")
    if not engine_path.exists():
        print("Building engine (this may take a while)...")
        engine = builder.build_engine(network, config)
        engine.save(engine_path)
    else:
        print("Loading existing engine...")
        engine = tensorrt_llm.runtime.Engine.load(engine_path)
    
    return engine

def format_prompt(instruction):
    """Format prompt for TinyLlama chat model"""
    return f"<|system|>You are a helpful AI assistant.<|user|>{instruction}<|assistant|>"

def run_inference(engine, prompts):
    """Run inference with TensorRT-LLM"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Format and tokenize prompts
    formatted_prompts = [format_prompt(p) for p in prompts]
    input_ids = tokenizer(formatted_prompts, return_tensors="pt", padding=True).input_ids
    
    print("\nRunning TensorRT-LLM inference...")
    start_time = time.time()
    
    # Run inference
    outputs = engine.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        temperature=0.8
    )
    
    inference_time = time.time() - start_time
    
    # Decode outputs
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Calculate throughput
    total_tokens = sum(len(output) for output in outputs)
    throughput = total_tokens / inference_time
    
    print(f"Inference took {inference_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} tokens/second")
    
    return output_texts, inference_time, throughput

def main():
    # Test prompts
    prompts = [
        "Write a poem about hot sauce in the style of Punjabi rapper Karan Aujla",
        "Explain the biggest problems facing the field of whole-brain emulation in two sentences",
        "Write a short story about a robot learning to love"
    ]
    
    # Build/load engine and run inference
    start_time = time.time()
    engine = setup_engine()
    load_time = time.time() - start_time
    
    outputs, infer_time, throughput = run_inference(engine, prompts)
    
    # Print results
    print("\n=== TensorRT-LLM Performance ===")
    print(f"Engine build/load time: {load_time:.2f} seconds")
    print(f"Inference time: {infer_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} tokens/second")
    
    # Print sample output
    print("\n=== Sample Output ===")
    print(f"Prompt: {prompts[0]}")
    print(f"Generated: {outputs[0]}")

if __name__ == "__main__":
    main()