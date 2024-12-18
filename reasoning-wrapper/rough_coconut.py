from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional, Mapping
import re

class Coconut:
    def __init__(self, 
                 model_name: str = "gpt2", # Changed from TinyLlama to GPT-2
                 num_thoughts: int = 3,
                 max_length: int = 256,
                 temperature: float = 0.7):
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

        # Add special tokens for latent reasoning
        self.tokenizer.add_special_tokens({'bos_token': "<bot>"})
        self.tokenizer.add_special_tokens({'eos_token': "<eot>"})
    
    # Resize model embeddings to account for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.eval()
        self.num_thoughts = num_thoughts
        self.max_length = max_length
        self.temperature = temperature

    def _get_hidden_state_output(self, input_ids):
        """
        Get the hidden state output from the model.
        """
        outputs = self.model(
            input_ids,
            output_hidden_states=True,
            use_cache=True
        )

        # Get the last hidden state of the last token
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        return last_hidden_state
    

    def _process_continuous_thought(self, current_hidden, context_size: int = 1):
        """
        Process a continuous thought in latent space.
        """ 
        # Create attention mask for the hidden state
        attention_mask = torch.ones((1, 1), dtype=torch.long)
        
        outputs = self.model(
            inputs_embeds=current_hidden.unsqueeze(1),
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True
        )

        # Get hidden state for next step
        hidden = outputs.hidden_states[-1][:, -1, :]
        return hidden, outputs.logits[:, -1, :]

    def _generate_final_answer(self, current_hidden):
        """
        Generate the final answer from the accumulated thought.
        """
        # Add <eot> token embedding to mark end of latent thought
        eot_token = torch.tensor(self.tokenizer.encode("<eot>")).unsqueeze(0)
        eot_embed = self.model.get_input_embeddings()(eot_token)
        
        # Combine hidden state with eot token
        final_context = torch.cat([
            current_hidden.unsqueeze(1),
            eot_embed
        ], dim=1)

        # Create attention mask for the combined context
        attention_mask = torch.ones((1, final_context.size(1)), dtype=torch.long)

        # Generate from the final context
        outputs = self.model.generate(
            inputs_embeds=final_context,
            attention_mask=attention_mask,
            max_new_tokens=self.max_length,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def solve(self, question: str) -> str:
        """
        Process a question through latent space reasoning.
        """
        # Add <bot> token to mark start of latent thought
        formatted_question = f"{question} <bot>"
        
        initial_inputs = self.tokenizer(
            formatted_question,
            return_tensors="pt",
            return_attention_mask=True
        )

        # Get initial hidden state
        current_hidden = self._get_hidden_state_output(initial_inputs.input_ids)
        
        print("\nInitial hidden state analysis:")
        print(f"Shape: {current_hidden.shape}")
        print(f"Mean: {current_hidden.mean().item()}")

        # Single accumulated thought for simplicity
        for step in range(self.num_thoughts):
            print(f"\nProcessing thought step {step + 1}")
            current_hidden, _ = self._process_continuous_thought(current_hidden)
            
        return self._generate_final_answer(current_hidden)

def test_coconut():
    coconut = Coconut(num_thoughts=4)
    question = "If John has 5 apples and gives 2 to Mary, who then shares 1 with Peter, how many apples does Mary have?"
    print("\nQuestion:", question)
    print("\nAnswer:", coconut.solve(question))

if __name__ == "__main__":
    test_coconut()