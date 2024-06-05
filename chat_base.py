# CUDA_VISIBLE_DEVICES=8 python3 applications/chat_base.py --model_path /data2/models/Abel-7B-001/ --cllm_type gsm8k --chat --debug

# We compare default greedy decoding and prompt lookup decoding in this file

import torch
import argparse
import subprocess

import time, os
import random
from typing import Dict, Optional, Sequence, List, Tuple
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from transformers.cache_utils import Cache, DynamicCache
from transformers import LlamaModel,LlamaForCausalLM
from transformers.generation import GenerationConfig

from transformers.generation.utils import _crop_past_key_values

import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

def get_default_question(cllm_type):
    if cllm_type == 'sharegpt':
        return "Which methods did Socrates employ to challenge the prevailing thoughts of his time?"
    elif cllm_type == 'spider':
        return "The SQL database has table named vehicle with columns ['Vehicle_ID', 'Model', 'Build_Year', 'Top_Speed', 'Power', 'Builder', 'Total_Production'], table named driver with columns ['Driver_ID', 'Name', 'Citizenship', 'Racing_Series'], table named vehicle_driver with columns ['Driver_ID', 'Vehicle_ID'], Question: What are the vehicle ids and models which have been driven by more than 2 drivers or been driven by the driver named 'Jeff Gordon'?"
    elif cllm_type == 'python':
        return "Implement the Conway's Game of Life. You should start with a 2D grid initialized with some configuration of live and dead cells. 1 for live cell and -1 for dead cell. The simulation should update the grid state by applying the rules for each cell simultaneously: any live cell with fewer than two live neighbors dies, as if by underpopulation. Any live cell with two or three live neighbors lives on to the next generation. Any live cell with more than three live neighbors dies, as if by overpopulation. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction. initial_grid = [[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]"
    elif cllm_type == 'gsm8k':
        return "Poppy is solving a 1000-piece jigsaw puzzle. She places a quarter of the pieces on the board, then her mom places a third of the remaining pieces. How many jigsaw pieces are left to be placed?"
    else:
        return "Tell me a short story."

def get_system_prompt(cllm_type):
    if cllm_type == 'sharegpt':
        return "Answer in English unless other language is used. A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    elif cllm_type == 'spider':
        return "Could you translate the following question into SQL. Please only generate SQL, don't include explanation in the answer.\n"
    elif cllm_type == 'python':
        return "Please generate code based on the following doc:\n"
    elif cllm_type == 'gsm8k':
        return ""
    else:
        return "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"

def get_instruction_template(system_prompt, roles, model_input, cllm_type):
    if cllm_type == 'sharegpt':
        return system_prompt + f"{roles[0]}: " + f"{model_input}\n{roles[1]}: "
    if cllm_type == 'spider' or 'python':
        return f"### Instruction:\n" + system_prompt + f"{model_input}\n" + f"### Response:\n"
    if cllm_type == 'gsm8k':
        prompt_mapping = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
        return prompt_mapping.format(input=model_input)
    else:
        return system_prompt + f"{roles[0]}: " + f"{model_input}\n{roles[1]}: "
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0) 
    parser.add_argument("--model_path", type=str, help="model path", default="meta-llama/Llama-2-7b-chat-hf") #tiiuae/falcon-7b-instruct #"TheBloke/Falcon-180B-Chat-GPTQ" 
    parser.add_argument("--model_type", type=str, default="llama")
    parser.add_argument("--cllm_type", type=str, default="sharegpt")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="n-token sequence size",
    )
    parser.add_argument(
        "--num_gram",
        type=int,
        default=2,
        help="Number of gram to match",
    )

    parser.add_argument(
        "--max_new_seq_len",
        type=int,
        default=256,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()
    
    if args.dtype == "float16":
        args.dtype = torch.float16
    elif args.dtype == "bfloat16":
        args.dtype = torch.bfloat16
    
    #if args.use_ds:
    config = transformers.AutoConfig.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
    )
 
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        model_max_length=512,
        padding_side="right",
        use_fast = False
    )

    # some special id of tokenizer
    # pad_token_id 0
    # bos_token_id 1
    # eos_token_id 2

    # print(tokenizer.eos_token_id)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='cuda',
        #attn_implementation="flash_attention_2",
    )

    user_input = ""
    num_rounds = 0
    if args.model_type == "llama":  
        roles = ("USER", "ASSISTANT") #support vicuna
    else:
        assert False 

    user_input = ""
    if args.model_type == "llama":  
        system_prompt = get_system_prompt(args.cllm_type)
    else:
        raise NotImplementedError('Only LLaMA or LLaMA2 architecture is supported.')

    while True:
        num_rounds += 1
        if args.chat:
            model_input = input("USER: ")
        else:
            model_input = get_default_question(args.cllm_type)
            print("USER: " + model_input)

        new_inputs = get_instruction_template(system_prompt, roles, model_input, args.cllm_type)
        user_input += new_inputs

        print("ASSISTANT: " , flush=True, end="")
        inputs = tokenizer(user_input, return_tensors="pt").to(args.device)

        os.environ["CHAT"] = "1"

        # calling huggingface generate function
        # t0 = time.time()
        # generate_ids = model.generate(inputs.input_ids, max_length=args.max_new_seq_len, prompt_lookup_num_tokens=10)
        # generated_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(generated_output)
        # t1 = time.time()

        t0 = time.time()
        # greedy_output, avg_fast_forwward_count = jacobi_generate(inputs, model, tokenizer, args.max_new_tokens, args.max_new_seq_len)

        # codes modified from https://github.com/mit-han-lab/streaming-llm/blob/main/examples/run_streaming_llama.py

        outputs = model(
            input_ids=inputs.input_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        print(tokenizer.decode(pred_token_idx.item(),skip_special_tokens=True), flush=True, end=" ")      
        generated_ids = [pred_token_idx.item()]
        pos = 0
        for _ in range(args.max_new_seq_len):
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

            print(tokenizer.batch_decode(pred_token_idx,skip_special_tokens=True)[0], flush=True, end=" ")      
            generated_ids.append(pred_token_idx.item())
            # generated_text = (
            #     tokenizer.decode(
            #         generated_ids,
            #         skip_special_tokens=True,
            #         clean_up_tokenization_spaces=True,
            #         spaces_between_special_tokens=False,
            #     )
            #     .strip()
            #     .split(" ")
            # )

            # now = len(generated_text) - 1
            # if now > pos:
            #     # print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            #     pos = now

            if pred_token_idx == tokenizer.eos_token_id:
                break
        # print(" ".join(generated_text[pos:]), flush=True)
        
        avg_fast_forwward_count = 1 
        t1 = time.time()
        
        os.environ["CHAT"] = "0"

        if args.debug:
            generated_tokens = len(generated_ids)
            print()
            print("======================================Basline=======================================================")
            print("Generated tokens: ", generated_tokens,"Time: ", round(t1 - t0, 2), "s Throughput: ", round(generated_tokens / (t1 - t0), 2), "tokens/s", "Fast forwarding: ", round(avg_fast_forwward_count, 2), "tokens/step")
            print("====================================================================================================")
        
        # prompt lookup decoding by https://github.com/apoorvumang/prompt-lookup-decoding/

        # initilize the KV cache

        t0 = time.time()

        outputs = model(
            input_ids=inputs.input_ids,
            past_key_values=None,
            use_cache=True,
        )

        prompt_ids = inputs.input_ids
        initial_prompt_length = prompt_ids.shape[-1]

        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

        print(tokenizer.batch_decode(pred_token_idx,skip_special_tokens=True)[0], flush=True, end=" ")      
        prompt_ids = torch.cat((prompt_ids, pred_token_idx), dim=-1)
        
        total_fast_forward_cnt = 0
        total_count = 0

        while True:
            # lookup in the prompt
            success_matched = False
            len_prompt = prompt_ids.shape[-1] 
            end = len_prompt - args.num_gram - args.max_new_tokens

            for i in range(end, args.num_gram-2, -1):
                matched = 0
                for j in range(0,args.num_gram):
                    if prompt_ids[0,i-j] == prompt_ids[0,len_prompt-1-j]:
                        matched += 1
                    else:
                        break
                if matched == args.num_gram:
                    success_matched = True
                    guess = prompt_ids[:, i:i+args.max_new_tokens]

            if success_matched is False:    
                guess = prompt_ids[:, -1].unsqueeze(-1)

            outputs = model(
                input_ids=guess,
                past_key_values=past_key_values,
                use_cache=True,
            )

            pred_gram = outputs.logits.argmax(dim=-1)

            fast_forward_cnt = 1
            if success_matched is True:
                for j in range(guess.shape[-1] - 1):
                    if pred_gram[0, j] == guess[0, j+1]:
                        fast_forward_cnt += 1
                    else:
                        break

            # if fast_forward_cnt > 1:
            #     pred_token_idx = torch.cat( (guess[:,1:fast_forward_cnt-1], pred_gram[:,fast_forward_cnt-1].unsqueeze(-1)), dim=-1)   
            #     generated_tokens = tokenizer.batch_decode(pred_token_idx, skip_special_tokens=True)[0] 
            #     print(" ")
            #     print(generated_tokens)
            #     print(" ")                
            # else:
            
            pred_token_idx = pred_gram[:,:fast_forward_cnt]
            total_fast_forward_cnt += fast_forward_cnt
            total_count += 1

            #print(pred_token_idx)

            generated_tokens = tokenizer.batch_decode(pred_token_idx,skip_special_tokens=True)[0]
            
            print(generated_tokens, flush=True, end=" ")      
            prompt_ids = torch.cat((prompt_ids, pred_token_idx), dim=-1)

            # update the KV cache
            past_key_values = _crop_past_key_values(model, outputs.past_key_values, prompt_ids.shape[-1] - 1)

            has_eos_token = False
            for j in range(0,pred_token_idx.shape[-1]):
                if pred_token_idx[0,j].cpu() ==  tokenizer.eos_token_id:
                    has_eos_token = True
                    break        
            if has_eos_token is True or prompt_ids.shape[-1] - initial_prompt_length > args.max_new_seq_len:
                break
        
        t1 = time.time()
        if args.debug:
            generated_tokens = prompt_ids.shape[-1] - initial_prompt_length
            print()
            print("======================================Prompt Lookup Decoding=======================================================")
            print("Generated tokens: ", generated_tokens,"Time: ", round(t1 - t0, 2), "s Throughput: ", round(generated_tokens / (t1 - t0), 2), "tokens/s", "Fast forwarding: ", round(total_fast_forward_cnt/total_count, 2), "tokens/step")
            print("====================================================================================================")
        
        # re-initialize user input
        # TODO: support multi-turn conversation
        user_input = ""
        
        if not args.chat:
            break

