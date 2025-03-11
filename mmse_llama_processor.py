from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

class MMSE_LlamaProcessor:
    def __init__(self, model_name="FineTuning_Llama2_Chat_Model_with_step_Dementia_Detection_BestResult.ipynb"):
        """Initialize Llama 2 model for MMSE processing."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def ask_mmse_question(self, question):
        """Ask MMSE question to the model."""
        promt = f"Question: {question} \n\
                Answer:"
        response = self.pipeline(promt, max_length=50, do_sample=True)[0]["generated_text"]
        return response.split("Answer:")[-1].strip()
    
    def evaluate_response(self, question, expected_answer):
        """Compare Llama respons with the expected MMSE answer."""
        response = self.ask_mmse_question(question)
        score = 1 if response.lower() == expected_answer.lower() else 0
        return {"Question": question, "LLM Answer": response, "Expected": expected_answer, "Score": score}
    
    def run_mmse_test(self, test_cases):
        """Run a batch of MMSE questions through Llama and generate a score."""
        results = []
        for q, expected in test_cases:
            result = self.evaluate_response(q, expected)
            results.append(result)
        return results
    
# Example usage
mmse_test_cases = [
    ("What is the year?", "2023"),
    ("Name three objects (Ball, Flag, Tree)", "Ball, Flag, Tree"),
    ("Spell 'WORLD' backward", "DLROW")
]

llama_processor = MMSE_LlamaProcessor()
llama_processor.run_mmse_test(mmse_test_cases)
