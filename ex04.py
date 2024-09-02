from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
from tqdm import tqdm

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

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

prompt = tokenizer.apply_chat_template(input_text, tokenize=False, add_generation_prompt=True)

# Create a tqdm progress bar
progress_bar = tqdm(total=512, desc="Generating text", unit="tokens")

def update_progress(_, __, generated_tokens):
    progress_bar.update(len(generated_tokens[0]) - progress_bar.n)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.5,
    top_k=50,
    top_p=0.9,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
    streamer=streamer  # Pass the streamer instance here
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

progress_bar.close()

print(generated_text)