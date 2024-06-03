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

MODEL = "gpt2"
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
    if "left" in llm_output:
        actions[0][6] = 1.0
    if "right" in llm_output:
        actions[0][7] = 1.0
    if "squat" in llm_output:
        actions[0][5] = 1.0
    if "jump" in llm_output:
        actions[0][4] = 1.0
    if "punch" in llm_output or "attack" in llm_output or "hit" in llm_output or "strike" in llm_output:
        actions[0][11] = 1.0
        actions.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    if "kick" in llm_output:
        actions[0][8] = 1.0
        actions.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    return actions

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
        sample = {}
        prompt = f"In Street Fighter, we want to be aggressive and defeat the enemy. Given {context} The available actions are: punch, kick, move left, move right, squat, jump, the best single action to take now is to"
        sample["review"] = prompt
        sample["input_ids"] = torch.as_tensor(tokenizer.encode(prompt)[: len(prompt)])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["label"] = torch.as_tensor(0)
        return sample

dataset = StreetFighterOnlineDataset(config)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]
    query_texts = batch["query"]

    #### Get response from gpt2
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute sentiment score
    reward = 0
    text = batch["response"][0]
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

    if epoch % 1000 == 0:
        ppo_trainer.save_pretrained("gpt2_finetune")
