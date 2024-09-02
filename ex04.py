from transformers import pipeline
import torch

# pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

input_text = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds like an Italian chef",
    },
    {
        "role": "user",
        "content": "How to make a good sandwich?",
    },
]

prompt = pipe.tokenizer.apply_chat_template(input_text, tokenize=False, add_generation_prompt=True)
outputs = pipe(
    prompt,
    max_new_tokens=512,
    do_sample=True, # to start using the sampling method for generating text
    temperature=0.5, # to modulate the probabilities of sequential tokens
    top_k=50, # to control the diversity of the generated text, to broaden the range of tokens that the model can pick from
    top_p=0.9, # to exclude less probable tokens from generation
    no_repeat_ngram_size=3, # to prevent the model from repeating the same n-gram multiple times
)

print(outputs[0]["generated_text"])