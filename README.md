
# Prompt Lookup Decoding

## Introduction

A simple implementation of PLD ([Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding)).

PLD copy the same pattern to be the draft in speculated decoding. It consists of three steps:
* Match the last n-gram prefix
* Copy the subsequent m tokens to be the draft
* Verify the draft 

## Expreiment Results

Compare naive greedy decoding and PLD on dataset GSM8k with [Abel-7B001](https://huggingface.co/GAIR/Abel-7B-001). 

```
CUDA_VISIBLE_DEVICES=0 python3 chat_base.py --model_path GAIR/Abel-7B-001/ --cllm_type gsm8k --chat --debug
```

PLD leads to a roughly 1.5x speed up.



