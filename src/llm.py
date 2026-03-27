import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LLMModel:
    def __init__(self, model_name="google/flan-t5-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt",truncation=True, max_length=512).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)