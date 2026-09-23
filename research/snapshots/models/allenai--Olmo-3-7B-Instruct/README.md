---
license: apache-2.0
base_model: allenai/Olmo-3-7B-Instruct-DPO
language:
- en
library_name: transformers
datasets:
- allenai/Dolci-Instruct-RL-7B
---

## Model Details
<img alt="Logo for Olmo 3 7B Instruct model" src="olmo-instruct.png" width="307px" style="margin-left:'auto' margin-right:'auto' display:'block'">


# Model Card for Olmo 3 7B Instruct

We introduce Olmo 3, a new family of 7B and 32B models both Instruct and Think variants. Long chain-of-thought thinking improves reasoning tasks like math and coding.

Olmo is a series of **O**pen **l**anguage **mo**dels designed to enable the science of language models. 
These models are pre-trained on the Dolma 3 dataset and post-trained on the Dolci datasets. We are releasing all code, checkpoints, logs (coming soon), and associated training details. 



The core models released in this batch include the following:

| **Stage**               | **Olmo 3 7B Think** | **Olmo 3 32B Think** | **Olmo 3 7B Instruct** |
|--------------------------|-----------------------|------------------------|---------------------------|
| **Base Model**           | [Olmo-3-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) | [Olmo-3-32B](https://huggingface.co/allenai/Olmo-3-1125-32B) | [Olmo-3-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) |
| **SFT**                  | [Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT) | [Olmo-3-32B-Think-SFT](https://huggingface.co/allenai/Olmo-3-32B-Think-SFT) | [Olmo-3-7B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT) |
| **DPO**                  | [Olmo-3-7B-Think-DPO](https://huggingface.co/allenai/Olmo-3-7B-Think-DPO) | [Olmo-3-32B-Think-DPO](https://huggingface.co/allenai/Olmo-3-32B-Think-DPO) | [Olmo-3-7B-Instruct-DPO](https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO) |
| **Final Models (RLVR)**  | [Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think) | [Olmo-3-32B-Think](https://huggingface.co/allenai/Olmo-3-32B-Think) | [Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) |          
             

## Installation

Olmo 3 is supported in transformers 4.57.0 or higher:
```bash
pip install transformers>=4.57.0
```

## Inference

You can use OLMo with the standard HuggingFace transformers library:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct")
message = [{"role": "user", "content": "Who would win in a fight - a dinosaur or a cow named Moo Moo?"}]
inputs = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt', return_dict=True)
# optional verifying cuda
# inputs = {k: v.to('cuda') for k,v in inputs.items()}
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.decode(response[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
>> 'This is a fun and imaginative question! Let’s break it down...'
```

For faster performance, you can quantize the model using the following method:
```python
AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Instruct", 
    torch_dtype=torch.float16, 
    load_in_8bit=True)  # Requires bitsandbytes
```
The quantized model is more sensitive to data types and CUDA operations. To avoid potential issues, it's recommended to pass the inputs directly to CUDA using:
```python
inputs.input_ids.to('cuda')
```

We have released checkpoints for these models. For post-training, the naming convention is `step_XXXX`. 


To load a specific model revision with HuggingFace, simply add the argument `revision`:
```bash
olmo = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Instruct", revision="step_300")
```

Or, you can access all the revisions for the models via the following code snippet:
```python
from huggingface_hub import list_repo_refs
out = list_repo_refs("allenai/Olmo-3-7B-Instruct")
branches = [b.name for b in out.branches]
```

## Chat template

## Default System Message
The default system prompt for this model is:
```
<|im_start|>system
You are a helpful function-calling AI assistant. 
You do not currently have access to any functions. <functions></functions><|im_end|>
```

## Chat Format

The chat template for this model is formatted as:
```
<|im_start|>system
You are a helpful function-calling AI assistant. 
You do not currently have access to any functions. <functions></functions><|im_end|>
<|im_start|>user
Who would win in a fight - a dinosaur or a cow named Moo Moo?<|im_end|>
<|im_start|>assistant
This is a fun and imaginative question! Let’s break it down...
Moo Moo the cow would certinaly win.
<|endoftext|>
```

### Model Description

- **Developed by:** Allen Institute for AI (Ai2)
- **Model type:** a Transformer style autoregressive language model.
- **Language(s) (NLP):** English
- **License:** This model is licensed under Apache 2.0. It is intended for research and educational use in accordance with Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use).
- **Contact:** Technical inquiries: `olmo@allenai.org`. Press: `press@allenai.org`
- **Date cutoff:** Dec. 2024.


### Model Sources

- **Project Page:** https://allenai.org/olmo
- **Repositories:**
    - Open-Instruct for DPO and RLVR: https://github.com/allenai/open-instruct
    - OLMo-Core for pre-training and SFT: https://github.com/allenai/OLMo-core
    - OLMo-Eval for evaluation: https://github.com/allenai/OLMo-Eval
- **Paper:** [TBD]
<!-- - **Technical blog post:** (URL)  -->
<!-- - **W&B Logs:** [SFT](()), [DPO](()), [RLVR](()) -->


## Evaluation

| **Skill** | **Benchmark** | **Olmo 3 Instruct 7B SFT** | **Olmo 3 Instruct 7B DPO** | **Olmo3 Instruct 7B** | **Qwen 3 8B (no reasoning)** | **Qwen 3 VL 8B Instruct** | **Qwen 2.5 7B** | **Olmo 2 7B Instruct** | **Apertus 8B Instruct** | **Granite 3.3 8B Instruct** |
|-----------|--------------|---------------------------|---------------------------|------------------------|------------------------------|----------------------------|-------------------|--------------------------|----------------------------|-------------------------------|
| **Math** | MATH | 65.1 | 79.6 | 87.3 | 82.3 | 91.6 | 71.0 | 30.1 | 21.9 | 67.3 |
|  | AIME 2024 | 6.7 | 23.5 | 44.3 | 26.2 | 55.1 | 11.3 | 1.3 | 0.5 | 7.3 |
|  | AIME 2025 | 7.2 | 20.4 | 32.5 | 21.7 | 43.3 | 6.3 | 0.4 | 0.2 | 6.3 |
|  | OMEGA | 14.4 | 22.8 | 28.9 | 20.5 | 32.3 | 13.7 | 5.2 | 5.0 | 10.7 |
| **Reasoning** | BigBenchHard | 51.0 | 69.3 | 71.2 | 73.7 | 85.6 | 68.8 | 43.8 | 42.2 | 61.2 |
|  | ZebraLogic | 18.0 | 28.4 | 32.9 | 25.4 | 64.3 | 10.7 | 5.3 | 5.3 | 17.6 |
|  | AGI Eval English | 59.2 | 64.0 | 64.4 | 76.0 | 84.5 | 69.8 | 56.1 | 50.8 | 64.0 |
| **Coding** | HumanEvalPlus | 69.8 | 72.9 | 77.2 | 79.8 | 82.9 | 74.9 | 25.8 | 34.4 | 64.0 |
|  | MBPP+ | 56.5 | 55.9 | 60.2 | 64.4 | 66.3 | 62.6 | 40.7 | 42.1 | 54.0 |
|  | LiveCodeBench v3 | 20.0 | 18.8 | 29.5 | 53.2 | 55.9 | 34.5 | 7.2 | 7.8 | 11.5 |
| **IF** | IFEval | 81.7 | 82.0 | 85.6 | 86.3 | 87.8 | 73.4 | 72.2 | 71.4 | 77.5 |
|  | IFBench | 27.4 | 29.3 | 32.3 | 29.3 | 34.0 | 28.4 | 26.7 | 22.1 | 22.3 |
| **Knowledge** | MMLU | 67.1 | 69.1 | 69.1 | 80.4 | 83.6 | 77.2 | 61.6 | 62.7 | 63.5 |
| **QA** | PopQA | 16.5 | 20.7 | 14.1 | 20.4 | 26.5 | 21.5 | 25.5 | 25.5 | 28.9 |
|  | GPQA | 30.0 | 37.9 | 40.4 | 44.6 | 51.1 | 35.6 | 31.3 | 28.8 | 33.0 |
| **Chat** | AlpacaEval 2 LC | 21.8 | 43.3 | 40.9 | 49.8 | 73.5 | 23.0 | 18.3 | 8.1 | 28.6 |
| **Tool Use** | SimpleQA | 74.2 | 79.8 | 79.3 | 79.0 | 90.3 | 78.0 | – | – | – |
|  | LitQA2 | 38.0 | 43.3 | 38.2 | 39.6 | 30.7 | 29.8 | – | – | – |
|  | BFCL | 48.9 | 49.6 | 49.8 | 60.2 | 66.2 | 55.8 | – | – | – |
| **Safety** | Safety | 89.2 | 90.2 | 87.3 | 78.0 | 80.2 | 73.4 | 93.1 | 72.2 | 73.7 |

## Model Details

#### Stage 1: SFT
- supervised fine-tuning on the Dolci-Think-SFT-7B dataset. This dataset consits of math, code, chat, and general knowledge queries.
- Datasets: [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/dolci-thinking-sft), [Dolci-Instruct-SFT-7B](https://huggingface.co/datasets/allenai/dolci-instruct-sft)

#### Stage 2:DPO
- direct preference optimization on the Dolci-Think-DPO-7B dataset. This dataset consits of math, code, chat, and general knowledge queries.
- Datasets: [Dolci-Think-DPO-7B](https://huggingface.co/datasets/allenai/dolci-thinking-dpo), [Dolci-Instruct-DPO-7B](https://huggingface.co/datasets/allenai/dolci-3-instruct-dpo-with-metadata)

#### Stage 3: RLVR
- reinforcement learning from verifiable rewards on the Dolci-Think-RL-7B dataset. This dataset consits of math, code, instruction-following, and general chat queries.
- Datasets: [Dolci-Think-RL-7B](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B), [Dolci-Instruct-RL-7B](https://huggingface.co/datasets/allenai/Dolci-Instruct-RL-7B)

## Inference & Recommended Settings
We evaluated our models on the following settings. We also recommend using them for generation:
- **temperature:** `0.6`
- **top_p:** `0.95`
- **max_tokens:** `32768`

### transformers Example
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "allenai/Olmo-3-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)

message = [{"role": "user", "content": "Who would win in a fight - a dinosaur or a cow named Moo Moo?"}]
inputs = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(model.device)

outputs = model.generate(
    **inputs,
    temperature=0.6,
    top_p=0.95,
    max_new_tokens=32768,
)

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### vllm Example
```python
from vllm import LLM, SamplingParams

model_id = "allenai/Olmo-3-7B-Instruct"
llm = LLM(model=model_id)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=32768,
)

message = [{"role": "user", "content": "Who would win in a fight - a dinosaur or a cow named Moo Moo?"}]
outputs = llm.chat(message, sampling_params)
print(outputs[0].outputs[0].text)
```


## Bias, Risks, and Limitations
Like any base language model or fine-tuned model without safety filtering, these models can easily be prompted by users to generate harmful and sensitive content. Such content may also be produced unintentionally, especially in cases involving bias, so we recommend that users consider the risks when applying this technology. Additionally, many statements from OLMo or any LLM are often inaccurate, so facts should be verified.

## License
This model is licensed under Apache 2.0. It is intended for research and educational use in accordance with [Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use).


## Citation

```
@misc{olmo2025olmo3,
title={Olmo 3},
author={Team Olmo and Allyson Ettinger and Amanda Bertsch and Bailey Kuehl and David Graham and David Heineman and Dirk Groeneveld and Faeze Brahman and Finbarr Timbers and Hamish Ivison and Jacob Morrison and Jake Poznanski and Kyle Lo and Luca Soldaini and Matt Jordan and Mayee Chen and Michael Noukhovitch and Nathan Lambert and Pete Walsh and Pradeep Dasigi and Robert Berry and Saumya Malik and Saurabh Shah and Scott Geng and Shane Arora and Shashank Gupta and Taira Anderson and Teng Xiao and Tyler Murray and Tyler Romero and Victoria Graf and Akari Asai and Akshita Bhagia and Alexander Wettig and Alisa Liu and Aman Rangapur and Chloe Anastasiades and Costa Huang and Dustin Schwenk and Harsh Trivedi and Ian Magnusson and Jaron Lochner and Jiacheng Liu and Lester James V. Miranda and Maarten Sap and Malia Morgan and Michael Schmitz and Michal Guerquin and Michael Wilson and Regan Huff and Ronan Le Bras and Rui Xin and Rulin Shao and Sam Skjonsberg and Shannon Zejiang Shen and Shuyue Stella Li and Tucker Wilde and Valentina Pyatkin and Will Merrill and Yapei Chang and Yuling Gu and Zhiyuan Zeng and Ashish Sabharwal and Luke Zettlemoyer and Pang Wei Koh and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
year={2025},
eprint={2512.13961},
archivePrefix={arXiv},
primaryClass={cs.CL},
url={https://arxiv.org/abs/2512.13961},
}
```

## Model Card Contact
For errors in this model card, contact `olmo@allenai.org`.