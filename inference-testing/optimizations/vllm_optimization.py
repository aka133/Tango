from vllm import LLM, SamplingParams
import time

def setup_vllm():
    """Initialize vLLM with a small model"""
    # Using a small model for quick testing
    model_name = "meta-llama/Llama-3.3-70B-Instruct"

    print(f"Loading model {model_name}...")
    llm = LLM(model = model_name)

    return llm

def run_batch_inference(llm, prompts):
    """Run batch inference with vLLM"""
    sampling_params = SamplingParams(
        temperature = 0.7,
        top_p = 0.95,
        max_tokens = 128
    )

    print("\nRunning batch inference...")
    start_time = time.time()

    # vLLM handles batching automatically
    outputs = llm.generate(prompts, sampling_params)

    end_time = time.time()
    print(f"Batch inference took {end_time - start_time:.2f} seconds")

    return outputs

def main():
    # Test prompts
    prompts = [
        "Write a poem about hot sauce in the style of Punjabi rapper Karan Aujla",
        "Explain the biggest problems facing the field of whole-brain emulation in two sentences",
        "Write a short story about a robot learning to love"
    ]

    # Initialize vLLM
    llm = setup_vllm()

    # Run inference
    outputs = run_batch_inference(llm, prompts)

    # Print results
    print("\nResults:")
    for prompt, output in zip(prompts, outputs):
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()