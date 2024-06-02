import torch
from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.models import PreTrainedModelWrapper
from peft import LoraConfig
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

model_id = "llava-hf/llava-1.5-7b-hf"

config = PPOConfig(
    model_name=model_id,
    learning_rate=1.41e-5,
    batch_size=1,
    mini_batch_size=1,
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

LLAVA_CHAT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. \
                        The assistant gives helpful, detailed, and polite answers to the user's questions. \
                        {% for message in messages %}{% if message['role'] == 'user' %}\
                        USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}\
                        {% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer = tokenizer

class LLavaDataset(Dataset):
    def __init__(self):
        return

    def __getitem__(self, examples):
        return[]
    
    def __len__(self):
        return 100000

dataset = LLavaDataset()

model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                    #   quantization_config=quantization_config,
                                                      torch_dtype=torch.float16).to("cuda:0")
# model = PreTrainedModelWrapper(model)

ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

epochs = 10
for epoch in tqdm(range(epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader): 
        # query_tensors = batch["input_ids"]
    
        #### Get response from SFTModel
        image = Image.fromarray(np.zeros([128, 128, 3], dtype=np.uint8), 'RGB')
        prompt = "USER: <image>\nJust say hi to me. ASSISTANT:"
        query_tensors = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    
        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards = [torch.tensor(0)]
    
        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

# ppo_trainer.train()
