# Import the required dependencies for this project
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

from datasets import Dataset, load_dataset
from collections import Counter, defaultdict
import random

# First load the entire dataset
ds = load_dataset('bitext/Bitext-travel-llm-chatbot-training-dataset', split="train")

# Group examples by intent
random.seed(42)
intent_groups = defaultdict(list)
for record in ds:
    intent = record["intent"]
    intent_groups[intent].append(record)

# Determine how many samples per intent
total_intents = len(intent_groups)
samples_per_intent = 100 // total_intents

# Sample from each intent
balanced_subset = []
for intent, examples in intent_groups.items():
    sampled = random.sample(examples, min(samples_per_intent, len(examples)))
    balanced_subset.extend(sampled)

total_num_of_records = 50    
travel_chat_ds = Dataset.from_list(balanced_subset[:total_num_of_records])

travel_chat_ds.to_pandas().head(3)

# Modify the dataset to include a conversation field
def merge_example(row):    
  row['conversation'] = f"Query: {row['instruction']}\nIntent: {row['intent']}\nResponse: {row['response']}"
  return row

travel_chat_ds = travel_chat_ds.map(merge_example)

print(travel_chat_ds[0]['conversation'])

# Load the Llama Model
model_name="TinyLlama/TinyLlama-1.1B-Chat-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Initialize SFTConfig
sftConfig = SFTConfig(
    max_steps=1,    
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=2e-3,
    max_grad_norm=0.3,
    save_steps=10,
    dataset_text_field='conversation',
    output_dir="/tmp",
)

# Initialize LoRA config
lora_config = LoraConfig(    
    r=12,    
    lora_alpha=32,    
    lora_dropout=0.05,    
    bias="none",    
    task_type="CAUSAL_LM",    
    target_modules=['q_proj', 'v_proj']
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=travel_chat_ds,
    peft_config=lora_config,
    args=sftConfig
)

# Kickstart fine-tuning process
trainer.train()

# Generate responses with fine-tuned model
inputs = tokenizer.encode("Query: I'm trying to book a flight", return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=20, max_length=20)
decoded_outputs = tokenizer.decode(outputs[0, inputs.shape[1]:], skip_special_tokens=True)
model_response = decoded_outputs