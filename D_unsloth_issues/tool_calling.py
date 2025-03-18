# %% [markdown]
# # Unsloth Tool Calling Demo
# 
# Fine-tune a model to handle structured tool calls using Unsloth's optimized training

# %% [code]
# Install dependencies
"""!pip install -q "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps "xformers<0.0.26" triton>=2.1.0
!pip install -q datasets transformers
"""
# %% [code]
from unsloth import FastLanguageModel
from datasets import Dataset
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
import json

# %% [markdown]
# ## 1. Prepare Tool Calling Dataset

# %% [code]
tool_calling_data = [
    {
        "prompt": "What's the weather like in Tokyo right now?",
        "response": json.dumps({"tool": "get_weather", "location": "Tokyo"})
    },
    {
        "prompt": "Book a table for 2 at 7pm at Chez Panisse",
        "response": json.dumps({
            "tool": "restaurant_booking",
            "name": "Chez Panisse",
            "time": "19:00",
            "party_size": 2
        })
    },
    {
        "prompt": "Check if the 3pm flight from SFO to JFK is delayed",
        "response": json.dumps({
            "tool": "flight_status",
            "origin": "SFO",
            "destination": "JFK",
            "time": "15:00"
        })
    },
    # Add more examples as needed
]

dataset = Dataset.from_dict({
    "prompt": [x["prompt"] for x in tool_calling_data],
    "response": [x["response"] for x in tool_calling_data],
})

# Split dataset
dataset = dataset.train_test_split(test_size=0.1)

# %% [markdown]
# ## 2. Configure Model with Unsloth

# %% [code]
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Set chat template
tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '### Instruction: ' + message['content'] + '\n' }}
{% else %}
{{ '### Response: ' + message['content'] + '\n' }}
{% endif %}
{% endfor %}"""

# %% [code]
# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
)

# %% [markdown]
# ## 3. Training Setup

# %% [code]
def formatting_prompts(examples):
    instructions = examples["prompt"]
    responses = examples["response"]
    
    texts = []
    for inst, resp in zip(instructions, responses):
        text = f"### Instruction: {inst}\n### Response: {resp}</s>"
        texts.append(text)
    return {"text" : texts,}

# %% [code]
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    dataset_text_field = "text",
    max_seq_length = 512,
    formatting_func = formatting_prompts,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 50,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        learning_rate = 2e-4,
        seed = 3407,
        output_dir = "outputs",
    ),
)

# %% [markdown]
# ## 4. Run Training

# %% [code]
trainer.train()

# %% [markdown]
# ## 5. Inference & Tool Parsing

# %% [code]
def generate_tool_call(query, max_new_tokens=64):
    prompt = f"### Instruction: {query}\n### Response:"
    
    inputs = tokenizer(
        [prompt],
        return_tensors = "pt",
        padding = True,
    ).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        pad_token_id = tokenizer.eos_token_id,
    )
    
    decoded = tokenizer.decode(outputs[0])
    response = decoded.split("### Response:")[-1].strip()
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"error": "Invalid tool call format"}

# %% [code]
# Test the model
test_query = "Check if the 9am flight from LAX to ORD is on time"
tool_call = generate_tool_call(test_query)
print("Generated Tool Call:", tool_call)

# %% [markdown]
# **Expected Output:**
# ```json
# {
#   "tool": "flight_status",
#   "origin": "LAX",
#   "destination": "ORD", 
#   "time": "09:00"
# }
# ```

# %% [markdown]
# ## Notes:
# 1. Add more training examples for better generalization
# 2. Implement output validation with Pydantic models
# 3. Add error handling for malformed JSON
# 4. Consider chain-of-thought prompting for complex tool calls
# 5. Monitor VRAM usage with `nvidia-smi`

# %% [code]
# Export to Google Drive (optional)
from google.colab import drive
drive.mount('/content/drive')

model.save_pretrained("/content/drive/MyDrive/unsloth_tool_calling_model")