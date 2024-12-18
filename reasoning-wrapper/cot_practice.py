from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional

class Coconut:
    def __init__(self, 
                 model_name: str = "gpt2", # or any other model like tinyllama or microsoft/Phi-3.5-mini-instruct
                 num_thoughts: int = 3,
                 max_length: int = 256,
                 temperature: float = 0.3):
        """
        Initialize the Coconut implementation.

        Args:
            model_name: The name of the model to use.
            num_thoughts: Number of thinking steps before providing an answer.
            max_length: The maximum length of the generated text.
            temperature: The temperature of the model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        self.model.eval()

        self.num_thoughts = num_thoughts
        self.max_length = max_length
        self.temperature = temperature
    
    def _get_llm_response(self, 
                          prompt: str,
                          max_new_tokens: Optional[int] = None,
                          new_temperature: Optional[float] = None,
                          ) -> str:
        """
        Get a response from our local model.

        Args: 
            prompt: The input prompt for the LLM
            max_new_tokens: Optional override for the maximum number of tokens to generate.
            temperature: Optional override for the temperature of the model.
        
        Returns:
            The LLM's response as a string.  
        """

        formatted_prompt = f"<|system|>You are a helpful AI assistant. </s><|user|>{prompt}</s><|assistant|>"

        # Encode the prompt
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True
        )

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = max_new_tokens or self.max_length,
                temperature = new_temperature or self.temperature,
                do_sample = True, 
                top_p = 0.5,
                repetition_penalty = 1.2,
                pad_token_id = self.tokenizer.eos_token_id,
                eos_token_id = self.tokenizer.eos_token_id
            )
        
        # Decode the response, excluding the input prompt
        prompt_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][prompt_length:],
            skip_special_tokens=True
        )

        return response.strip()

    def solve(self, question: str) -> str:
        """
        Solve a question using the chain of continuous thought.

        Args:
            question: The question to answer
        
        Returns:
            The answer to the question
        """
        
        # Build initial prompt for first thought
        system_context = (
            "You are an AI assistant that solves problems step by step. "
            "For each step, provide a key insight without explaining fully. "
            "Focus on facts and relationships that help solve the problem."
        )

        thoughts = []
        current_context = question

        # Generate chain of thoughts
        for i in range(self.num_thoughts):
            # Create prompt that includes previous thoughts
            thought_prompt = f"Previous thoughts: {', '.join(thoughts) if thoughts else 'None'}\nQuestion: {current_context}\nNext key insight:"
            prompt = f"<|system|>{system_context}</s><|user|>{thought_prompt}</s><|assistant|>"

            # Get next thought
            thought = self._get_llm_response(prompt)
            thoughts.append(thought)

            # Update context with new thought
            current_context = f"{question}\nThought: {thought}"
        
        # Generate final answer using accumulated thoughts
        final_prompt = f"Based on these thoughts: {', '.join(thoughts)}\nQuestion: {question}\nProvide a concise final answer:"
        prompt = f"<|system|>{system_context}</s><|user|>{final_prompt}</s><|assistant|>"

        return self._get_llm_response(prompt)

def test_llm_response():
    # Initialize Coconut
    coconut = Coconut()

    # Test with a simple prompt
    prompt = "What is the capital of France?"
    print("Prompt: ", prompt)
    response = coconut._get_llm_response(prompt)
    print("Response: ", response)


def test_coconut():
    coconut = Coconut(num_thoughts=4)
    question = "If John has 5 apples and gives 2 to Mary, who then shares 1 with Peter, how many apples does Mary have?"
    print("\nQuestion:", question)
    print("\nAnswer:", coconut.solve(question))

if __name__ == "__main__":
    test_coconut()