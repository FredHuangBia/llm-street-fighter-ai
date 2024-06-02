# Copyright 2023 LIN Yi. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import time 

import retro
from stable_baselines3 import PPO

from street_fighter_custom_wrapper import StreetFighterCustomWrapper
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration


RESET_ROUND = True  # Whether to reset the round when fight is over. 
RENDERING = True    # Whether to render the game screen.

MODEL_NAME = r"llava" # Specify the model file to load. Model "ppo_ryu_2500000_steps_updated" is capable of beating the final stage (Bison) of the game.

# Model notes:
# ppo_ryu_2000000_steps_updated: Just beginning to overfit state, generalizable but not quite capable.
# ppo_ryu_2500000_steps_updated: Approaching the final overfitted state, cannot dominate first round but partially generalizable. High chance of beating the final stage.
# ppo_ryu_3000000_steps_updated: Near the final overfitted state, almost dominate first round but barely generalizable.
# ppo_ryu_7000000_steps_updated: Overfitted, dominates first round but not generalizable. 

RANDOM_ACTION = False
NUM_EPISODES = 30 # Make sure NUM_EPISODES >= 3 if you set RESET_ROUND to False to see the whole final stage game.
MODEL_DIR = r"trained_models/"

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
env = make_env(game, state="Champion.Level12.RyuVsBison")()
# model = PPO("CnnPolicy", env)

# if not RANDOM_ACTION:
#     model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)
if not RANDOM_ACTION:
    # 如果你使用的是linyiLYi提供的模型
    keys = ['high', 'low', 'bounded_above', 'bounded_below']
    setattr(env.observation_space, '_shape', (3,100,128))
    for k in keys:
        new_attr = getattr(env.observation_space, k).reshape(3,100,128)
        setattr(env.observation_space, k, new_attr)

    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf").to("cuda:0")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

def get_llm_actions(llm_output):
    llm_output = llm_output.split("ASSISTANT")[1]
    llm_output = llm_output.lower()
    actions = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
    if "left" in llm_output:
        actions[0][6] = 1.0
    if "right" in llm_output:
        actions[0][7] = 1.0
    if "squad" in llm_output:
        actions[0][5] = 1.0
    if "jump" in llm_output:
        actions[0][4] = 1.0
    if "punch" in llm_output:
        actions[0][11] = 1.0
        actions.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    if "kick" in llm_output:
        actions[0][8] = 1.0
        actions.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # jump = [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
    # punch = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.] -> [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    # quick punch = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.] -> [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    # heavy punch = [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.] -> [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    # kick = [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.] -> [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    # quick kick = [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.] -> [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    # heavy kick = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.] -> [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    return actions

obs = env.reset()
done = False

num_episodes = NUM_EPISODES
episode_reward_sum = 0
num_victory = 0

print("\nFighting Begins!\n")

for _ in range(num_episodes):
    done = False
    
    if RESET_ROUND:
        obs = env.reset()

    total_reward = 0

    while not done:
        timestamp = time.time()

        if RANDOM_ACTION:
            obs, reward, done, info = env.step(env.action_space.sample())
        else:
            image = Image.fromarray(obs.astype('uint8'), 'RGB')

            prompt = "USER: <image>\nYou are the white cloth player in street fighter game. Given the game state image, think about where is the opponent and where you are, what is the one best action from {left, right, squad, jump, punch, kick}? ASSISTANT:"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")

            output = model.generate(**inputs, max_new_tokens=100)
            llm_output = processor.decode(output[0], skip_special_tokens=True)
            print(llm_output)
            actions = get_llm_actions(llm_output)
            for action in actions:
                obs, reward, done, info = env.step(action)

        if reward != 0:
            total_reward += reward
            print("Reward: {:.3f}, playerHP: {}, enemyHP:{}".format(reward, info['agent_hp'], info['enemy_hp']))
        
        if info['enemy_hp'] < 0 or info['agent_hp'] < 0:
            done = True

    if info['enemy_hp'] < 0:
        print("Victory!")
        num_victory += 1

    print("Total reward: {}\n".format(total_reward))
    episode_reward_sum += total_reward

    if not RESET_ROUND:
        while info['enemy_hp'] < 0 or info['agent_hp'] < 0:
        # Inter scene transition. Do nothing.
            obs, reward, done, info = env.step([0] * 12)
            env.render()

env.close()
print("Winning rate: {}".format(1.0 * num_victory / num_episodes))
if RANDOM_ACTION:
    print("Average reward for random action: {}".format(episode_reward_sum/num_episodes))
else:
    print("Average reward for {}: {}".format(MODEL_NAME, episode_reward_sum/num_episodes))