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

        subjects = [
            "abstract_algebra",
            "anatomy",
            "virology",
            "high_school_physics"
        ]

        print("Loading MMLU datasets...")
        self.mmlu = []
        for subject in subjects:
            dataset = load_dataset("cais/mmlu", subject, streaming=False)
            self.mmlu.extend(dataset['test'])  # type: ignore
        
        print(f"Loaded {len(self.mmlu)} MMLU examples")

    def evaluate_mmlu(self, model, tokenizer):
        correct = 0
        total = 0
        
        print("\nEvaluating MMLU...")
        for example in self.mmlu:
            # MMLU is multiple choice (A,B,C,D)
            question = example['question']  # type: ignore
            choices = example['choices']    # type: ignore
            correct_answer = example['answer'] # type: ignore
            
            # Format as multiple choice
            prompt = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits / 2.0, dim=-1)
            predicted_class = int(probs.argmax().item())
            
            if predicted_class == correct_answer:
                correct += 1
            total += 1
            
            if total % 10 == 0:  # Progress update
                print(f"Processed {total}/{len(self.mmlu)} questions. Current accuracy: {(correct/total)*100:.2f}%")
        
        final_accuracy = (correct/total) * 100
        print(f"\nFinal MMLU Accuracy: {final_accuracy:.2f}%")
        return final_accuracy

def main():
    evaluator = BaseEvaluator()
    model, tokenizer = load_model()
    
    # Run full evaluation on all 1200 examples
    evaluator.evaluate_mmlu(model, tokenizer)

if __name__ == "__main__":
    main()