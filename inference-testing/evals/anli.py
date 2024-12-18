import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import time

def load_model():
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"  # Public model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

class BaseEvaluator:

    def __init__(self):
        self.load_datasets()

    def load_datasets(self):
        print("Loading ANLI R3 dataset...")
        anli_dataset = load_dataset("anli")
        self.anli = anli_dataset['test_r3'] # type: ignore
        
        # Print some stats to verify loading
        print(f"Loaded {len(self.anli)} ANLI examples")

    def evaluate_anli(self, model, tokenizer):
        correct = 0
        total = 0
        results = []
        
        print("Evaluating ANLI R3...")
        for example in self.anli:  # Loop through all 1200 examples
            inputs = tokenizer(
                example['premise'], # type: ignore
                example['hypothesis'], # type: ignore
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits / 2.0, dim=-1)
            predicted_class = int(probs.argmax().item())
            
            # Track accuracy
            if predicted_class == example['label']: # type: ignore
                correct += 1
            total += 1
            
            if total % 100 == 0:  # Progress update
                print(f"Processed {total}/1200 examples. Current accuracy: {(correct/total)*100:.2f}%")
        
        final_accuracy = (correct/total) * 100
        print(f"\nFinal ANLI R3 Accuracy: {final_accuracy:.2f}%")
        return final_accuracy

def main():
    evaluator = BaseEvaluator()
    model, tokenizer = load_model()
    
    # Run full evaluation on all 1200 examples
    evaluator.evaluate_anli(model, tokenizer)

if __name__ == "__main__":
    main()