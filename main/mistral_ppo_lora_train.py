import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset

tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

import retro
from street_fighter_custom_wrapper import StreetFighterCustomWrapper
from observer import Observer
from peft import LoraConfig

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
RESET_ROUND = True  # Whether to reset the round when fight is over. 
RENDERING = False    # Whether to render the game screen.
def make_env(game, state):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE
        )
        env = StreetFighterCustomWrapper(env, reset_round=RESET_ROUND, rendering=RENDERING)
        return env
    return _init
game = "StreetFighterIISpecialChampionEdition-Genesis"

def get_llm_actions(llm_output):
    llm_output = llm_output.lower()
    actions = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
    if "move left" in llm_output:
        actions[0][6] = 1.0
    if "move right" in llm_output:
        actions[0][7] = 1.0
    if "crouch" in llm_output:
        actions[0][5] = 1.0
    if "crouch right" in llm_output:
        actions[0][7] = 1.0
        actions[0][5] = 1.0
    if "crouch left" in llm_output:
        actions[0][6] = 1.0
        actions[0][5] = 1.0
    if "jump" in llm_output:
        actions[0][4] = 1.0
    if "jump right" in llm_output:
        actions[0][4] = 1.0
        actions[0][7] = 1.0
    if "jump left" in llm_output:
        actions[0][4] = 1.0
        actions[0][6] = 1.0
    if "punch" in llm_output:
        actions[0][11] = 1.0
        actions.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    if "kick" in llm_output:
        actions[0][8] = 1.0
        actions.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    return actions

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

config = PPOConfig(
    model_name=MODEL,
    learning_rate=1.41e-5,
    batch_size=1,
    mini_batch_size=1
)

sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}


class StreetFighterOnlineDataset(Dataset):
    def __init__(self, config):
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        self.env = make_env(game, state="Champion.Level12.RyuVsBison")()
        self.obs = self.env.reset()
        RYU_GREY = [168, 168, 168]
        GUILE_RED = [232, 32, 32]
        self.observer = Observer(RYU_GREY, GUILE_RED)

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        observation = {"frame": self.obs}
        self.observer.observe(observation)
        context = self.observer.context_prompt()
        action_descriptions = {
        "punch": "a quick attack that deals moderate damage to the opponent if close enough.",
        "kick": "a powerful attack with longer reach than a punch but slightly slower execution.",
        "move left": "shifts your position to the left, potentially avoiding an opponent's attack.",
        "move right": "shifts your position to the right, potentially avoiding an opponent's attack.",
        "crouch": "lowers your stance, dodging high attacks and preparing for defensive or low attacks.",
        "crouch right": "moves right while crouched, evading high attacks and positioning for a counter.",
        "crouch left": "moves left while crouched, evading high attacks and positioning for a counter.",
        "jump": "leaps upward, avoiding low attacks and creating opportunities for aerial assaults.",
        "jump right": "jumps to the right, covering distance and evading low threats.",
        "jump left": "jumps to the left, covering distance and evading low threats."
        }
        action_list_description = "Here are your possible movements and their effects: "
        for action, description in action_descriptions.items():
            action_list_description += f"\n- {action.capitalize()}: {description}"
        sample = {}
        # prompt = f"In Street Fighter, we want to be aggressive and defeat the enemy. Given {context} The available actions are: punch, kick, move left, move right, squat, jump, the best single action to take now is to"
        # sample["review"] = prompt
        # sample["input_ids"] = torch.as_tensor(tokenizer.encode(prompt)[: len(prompt)])
        # sample["query"] = tokenizer.decode(sample["input_ids"])
        # sample["label"] = torch.as_tensor(0)

        prompt = f"[INST] You are playing Street Fighter, your goal is to defeat the opponent. This is the current game state: {context} Your available movements are in this list: Your available movements are in this list: [punch, kick, move left, move right, crouch, crouch right, crouch left, jump, jump right, jump left]. {action_list_description}. Pick one best movement to take from the list. Just constrain your answer to one word such as 'punch'.[/INST]"
        sample = {}
        sample = tokenizer([prompt], return_tensors="pt").to("cuda")
        sample["query"] = tokenizer.batch_decode(sample["input_ids"])
        sample["input_ids"] = sample["input_ids"][0]
        sample["attention_mask"] = sample["attention_mask"][0]
        sample["query"] = sample["query"][0]

        return sample

dataset = StreetFighterOnlineDataset(config)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name,
                                                          torch_dtype=torch.bfloat16,
                                                          peft_config=lora_config)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name,
                                                              torch_dtype=torch.bfloat16,
                                                              peft_config=lora_config)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

generation_kwargs = {
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

output_min_length = 10
output_max_length = 20
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]
    query_texts = batch["query"]

    #### Get response from gpt2
    response_tensors = []
    for i,query in enumerate(query_tensors):
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        generation_kwargs["attention_mask"] = torch.unsqueeze(batch["attention_mask"][i], 0)
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute sentiment score
    reward = 0
    text = batch["response"][0]
    query_text = query_texts[0]

    # Hack to make the model learn to attack more
    if epoch < 10000 and ("punch" in text or "attack" in text or "hit" in text or "strike" in text or "kick" in text):
        reward += 0.1
    actions = get_llm_actions(text)
    for action in actions:
        obs, reward, done, info = dataset.env.step(action)
        if done:
            obs = dataset.env.reset()
            break
        dataset.obs = obs
        reward += reward
    rewards = [torch.tensor(reward)]
    if (reward != 0):
        print(reward)

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if epoch % 100 == 0:
        ppo_trainer.save_pretrained("mistral_finetune")

