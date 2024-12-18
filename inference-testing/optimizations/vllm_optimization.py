from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os

def format_prompt(instruction):
    """Format prompt for TinyLlama chat model"""
    return f"<|system|>You are a helpful AI assistant.<|user|>{instruction}<|assistant|>"

def setup_vllm():
    """Initialize vLLM with a small model"""
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"Loading model {model_name}...")
    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=2048, 
            trust_remote_code=True
        )
        return llm
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def run_batch_inference(llm, prompts):
    """Run batch inference with vLLM"""
    # Format prompts properly
    formatted_prompts = [format_prompt(p) for p in prompts]
    
    sampling_params = SamplingParams(
        temperature=0.8,  # Slightly more creative
        max_tokens=512,  
        stop=["<|user|>", "<|system|>"] 
    )

    print("\nRunning batch inference...")
    start_time = time.time()

    try:
        # vLLM handles batching automatically
        outputs = llm.generate(formatted_prompts, sampling_params)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate tokens per second
        total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        tokens_per_second = total_output_tokens / total_time
        
        print(f"Batch inference took {total_time:.2f} seconds")
        print(f"Throughput: {tokens_per_second:.2f} tokens/second")
        
        return outputs, total_time, tokens_per_second
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise

def run_transformers_inference(prompts):
    """Run inference using regular transformers"""
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"\nLoading {model_name} with Transformers...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")

    formatted_prompts = [format_prompt(p) for p in prompts]
    outputs = []
    inference_start = time.time()

    for prompt in formatted_prompts:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
            )
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            outputs.append(output_text)
    
    inference_time = time.time() - inference_start
    total_tokens = sum(len(tokenizer.encode(text)) for text in outputs)
    trans_throughput = total_tokens / inference_time
    
    print(f"Inference took {inference_time:.2f} seconds")

    return outputs, load_time, inference_time, trans_throughput

def main():
    # Test prompts
    prompts = [
        "Write a poem about hot sauce in the style of Punjabi rapper Karan Aujla",
        "Explain the biggest problems facing the field of whole-brain emulation in two sentences",
        "Write a short story about a robot learning to love"
    ]

    # Run vLLM
    print("\n=== vLLM Performance ===")
    vllm_start = time.time()
    llm = setup_vllm()
    vllm_load = time.time() - vllm_start
    vllm_outputs, vllm_infer, vllm_throughput = run_batch_inference(llm, prompts)

    # Clear GPU memory
    del llm
    torch.cuda.empty_cache()

    # Run transformers
    print("\n=== Transformers Performance ===")
    trans_outputs, trans_load, trans_infer, trans_throughput = run_transformers_inference(prompts)

    # Print comparison
    print("\n=== Performance Comparison ===")
    print(f"Load Time: vLLM {vllm_load:.2f}s vs Transformers {trans_load:.2f}s")
    print(f"Inference Time: vLLM {vllm_infer:.2f}s vs Transformers {trans_infer:.2f}s")
    print(f"Throughput: vLLM {vllm_throughput:.2f} tokens/s vs Transformers {trans_throughput:.2f} tokens/s")

if __name__ == "__main__":
    main()