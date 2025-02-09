from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B",trust_remote_code = True)
model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", trust_remote_code=True)
def calculate(outfit):
    initial = "Evaluate the outfit combination with the following items: [{outfit}]. Analyze the overall style, harmony in design and material, and suitability for different occasions. Provide a percentage-based rating for each criterion: overall style (XX%), outfit harmony (XX%), occasion appropriateness (XX%), balance between items (XX%), and total score (XX%). After the evaluation, suggest an optimized outfit combination if necessary, including clothing, accessories, and footwear to create a more stylish and well-balanced look"
    input = tokenizer(initial,return_tensors='pt')
    output = model(**input)
    return output