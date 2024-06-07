import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset

tqdm.pandas()

from transformers import pipeline, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig
from trl.models.modeling_value_head import *
from trl.core import LengthSampler

import retro
from street_fighter_custom_wrapper import StreetFighterCustomWrapper
from PIL import Image
from peft import LoraConfig


class AutoModelForConditionLMWithValueHead(PreTrainedModelWrapper):
    r"""
    An autoregressive model with a value head in addition to the language model head.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained`, `push_to_hub` and `generate`. To call a method of the wrapped
    model, simply manipulate the `pretrained_model` attribute of this class.

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the `ValueHead` class. Currently, the supported args are:
            - **summary_dropout_prob** (`float`, `optional`, defaults to `None`) -- The dropout probability for the
                `ValueHead` class.
            - **v_head_initializer_range** (`float`, `optional`, defaults to `0.2`) -- The initializer range for the
                `ValueHead` if a specific initialization strategy is selected.
            - **v_head_init_strategy** (`str`, `optional`, defaults to `None`) -- The initialization strategy for the
                `ValueHead`. Currently, the supported strategies are:
                - **`None`** -- Initializes the weights of the `ValueHead` with a random distribution. This is the default
                    strategy.
                - **"normal"** -- Initializes the weights of the `ValueHead` with a normal distribution.

    """

    transformers_parent_class = LlavaForConditionalGeneration
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
        """
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)

        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)

        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        r"""
        Initializes the weights of the value head. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `v_head_init_strategy` argument
        when calling `.from_pretrained`. Supported strategies are:
        - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class. These arguments
                can contain the `v_head_init_strategy` argument as well as the `v_head_initializer_range`
                argument.
        """
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        value = self.v_head(last_hidden_state).squeeze(-1)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value)

    def generate(self, *args, **kwargs):
        r"""
        A simple wrapper around the `generate` method of the wrapped model.
        Please refer to the [`generate`](https://huggingface.co/docs/transformers/internal/generation_utils)
        method of the wrapped model for more information about the supported arguments.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed to the `generate` method of the wrapped model.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed to the `generate` method of the wrapped model.
        """
        return self.pretrained_model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            # if it is a peft model, only save the v_head
            pretrained_model_state_dict = {}

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        self.pretrained_model.v_head = self.v_head

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            first_device = list(set(self.pretrained_model.hf_device_map.values()))[0]
            if isinstance(first_device, int):
                if is_npu_available():
                    first_device = f"npu:{first_device}"
                elif is_xpu_available():
                    first_device = f"xpu:{first_device}"
                else:
                    first_device = f"cuda:{first_device}"
            self.v_head = self.v_head.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)

            self.is_sequential_parallel = True


MODEL = "llava-hf/llava-1.5-7b-hf"
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

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules="all-linear"
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
        processor = AutoProcessor.from_pretrained(config.model_name)
        self.env = make_env(game, state="Champion.Level12.RyuVsBison")()
        self.obs = self.env.reset()

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        image = Image.fromarray(self.obs.astype('uint8'), 'RGB')

        prompt = "USER: <image>\nYou are playing Street Fighter, your goal is to defeat the opponent. Please check the given game state image. Your available movements are in this list: [punch, kick, move-left, move-right, squat, jump]. Pick one best movement to take from the list. Just answer in one word. ASSISTANT:"
        sample = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")
        sample["query"] = tokenizer.batch_decode(sample["input_ids"])
        sample["input_ids"] = sample["input_ids"][0]
        sample["pixel_values"] = sample["pixel_values"][0]
        sample["query"] = prompt

        return sample

dataset = StreetFighterOnlineDataset(config)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

model = AutoModelForConditionLMWithValueHead.from_pretrained(config.model_name,
                                                          torch_dtype=torch.bfloat16,
                                                          peft_config=lora_config)
ref_model = AutoModelForConditionLMWithValueHead.from_pretrained(config.model_name,
                                                              torch_dtype=torch.bfloat16,
                                                              peft_config=lora_config)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
processor = AutoProcessor.from_pretrained(config.model_name)

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
        generation_kwargs["pixel_values"] = torch.unsqueeze(batch["pixel_values"][i], 0)
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute sentiment score
    reward = 0
    text = batch["response"][0]
    query_text = query_texts[0]
    print("Q:", query_text)
    print("A:", text)
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
        ppo_trainer.save_pretrained("llava_finetune")
