# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, interleave_datasets
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainerRefEMA
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from rouge_score import rouge_scorer

import random
from functools import partial
import math

from datasets import load_dataset, load_from_disk
import json


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    bonus_coefficient: Optional[float] = field(
        default=1.0,
        metadata={"help": "The coefficient for consistency bonus."}
    )
    freeze_visual_encoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the visual encoder."},    
    )
    use_care: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use consistency-aware reward enhancement."},    
    )
    confidence_upper_bound: Optional[float] = field(
        default=None,
        metadata={"help": "The upper bound value for clamping reference likelihood."},    
    )
    ref_ema_decay: Optional[float] = field(
        default=None,
        metadata={"help": "Decay used for updating the reference model using EMA"},    
    )
    ref_ema_update_every: Optional[float] = field(
        default=None,
        metadata={"help": "Step interval used for updating the reference model using EMA"},    
    )
    customized_top_p: Optional[float] = field(
        default=1.0,
        metadata={"help": "The top_p used for online sampling"},    
    )
    max_gen_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum generation length for control"},    
    )
    min_gen_len: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum generation length for control"},    
    )
    consistency_margin: Optional[float] = field(
        default=None,
        metadata={"help": "Do not punish the samples with reference likelihoods within a given margin below the mean confidence"},    
    )
   

def clean_text(text, exclue_chars=['\n', '\r']):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]
    
    for char in exclue_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    
    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()



def default_accuracy_reward(content, sol, **kwargs):
    reward = 0.0
    # Extract answer from solution if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    # print(f"Solution: {sol}")
    
    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()


    # If symbolic verification failed, try string matching or fuzzy matching
    if reward == 0.0:
        try: 
            final_pred = clean_text(student_answer)
            final_gt = clean_text(ground_truth)

            # print(f".  clean text   {final_pred}  vs.  {final_gt}")
            reward = 1.0 if final_pred == final_gt else 0.0
        except Exception:
            pass  # Keep reward as 0.0 if all methods fail

    return reward


def accuracy_reward(prompts, completions, solution, **kwargs):
   
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for prompt, content, sol in zip(prompts, contents, solution):
        reward = default_accuracy_reward(content, sol)  
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                # f.write(f"Problem ID: {problem_id}\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}



def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Format into conversation
    QUESTION_TEMPLATE = "{Question} First think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"


    all_data = []
    with open(script_args.dataset_name, 'r') as f:
        for line in f:
            item = json.loads(line)
            if 'image' in item:
                if isinstance(item['image'], str):
                    # Store image path instead of loading the image
                    item['image_path'] = os.path.join(os.environ.get('IMAGE_ROOT'), item['image'])
                    del item['image'] # remove the image column so that it can be loaded later
                elif isinstance(item['image'], list):
                    # if the image is a list, then it is a list of images (for multi-image input)
                    item['image_path'] = [os.path.join(image_folder, image) for image in item['image']]
                    del item['image'] # remove the image column so that it can be loaded later
                else:
                    raise ValueError(f"Unsupported image type: {type(item['image'])}")
            # Remove immediate image loading
            item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
            
            # Handle solution that could be a float or string
            solution_value = item['conversations'][1]['value']
            if isinstance(solution_value, str):
                item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
            else:
                # If it's a float or other non-string type, keep it as is
                item['solution'] = str(solution_value)
            
            del item['conversations']
            # item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method) # if accu_reward_method is in the data jsonl, use the value in the data jsonl, otherwise use the defined value
            all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        msg = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example['image_path'], "max_pixels": 802816},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example['problem'])},
                    ]
                },
            ],
            'solution': example['solution'],
            'problem_type': 'multiple choice'
        }

        return msg



    # Map the conversations
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    # Split dataset for validation if requested
    splits = {'train': dataset}


    trainer_cls = Qwen2VLGRPOTrainerRefEMA


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=splits['train'],
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

