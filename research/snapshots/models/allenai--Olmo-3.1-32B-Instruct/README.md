---
license: apache-2.0
base_model: allenai/Olmo-3.1-32B-Instruct-DPO
language:
- en
library_name: transformers
datasets:
- allenai/Dolci-Instruct-RL
---

## Model Details
<img alt="Logo for Olmo 3.1 32B Instruct model" src="olmo-instruct.png" width="307px" style="margin-left:'auto' margin-right:'auto' display:'block'">



# Model Card for Olmo-3.1-32B-Instruct

We introduce Olmo 3, a new family of 7B and 32B models both Instruct and Think variants. Long chain-of-thought thinking improves reasoning tasks like math and coding.

Olmo is a series of **O**pen **l**anguage **mo**dels designed to enable the science of language models. 
These models are pre-trained on the Dolma 3 dataset and post-trained on the Dolci datasets. We are releasing all code, checkpoints, logs (coming soon), and associated training details. 



The core models released in this batch include the following:

| **Stage**               | **Olmo 3 7B Think** | **Olmo (3/3.1) 32B Think** | **Olmo 3 7B Instruct** | **Olmo 3.1 32B Instruct** |
|--------------------------|-----------------------|------------------------|---------------------------|----------------------------|
| **Base Model**           | [Olmo-3-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) | [Olmo-3-32B](https://huggingface.co/allenai/Olmo-3-1125-32B) | [Olmo-3-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) | [Olmo-3-32B](https://huggingface.co/allenai/Olmo-3-1125-32B) |
| **SFT**                  | [Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT) | [Olmo-3-32B-Think-SFT](https://huggingface.co/allenai/Olmo-3-32B-Think-SFT) | [Olmo-3-7B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT) | [Olmo-3.1-32B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct-SFT) |
| **DPO**                  | [Olmo-3-7B-Think-DPO](https://huggingface.co/allenai/Olmo-3-7B-Think-DPO) | [Olmo-3-32B-Think-DPO](https://huggingface.co/allenai/Olmo-3-32B-Think-DPO) | [Olmo-3-7B-Instruct-DPO](https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO) | [Olmo-3.1-32B-Instruct-DPO](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct-DPO) |
| **Final Models (RLVR)**  | [Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think) | [Olmo-3-32B-Think](https://huggingface.co/allenai/Olmo-3-32B-Think)<br>[Olmo-3.1-32B-Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think) | [Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) | [Olmo-3.1-32B-Instruct](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct) |
             

## Installation

Olmo 3 is supported in transformers 4.57.0 or higher:
```bash
pip install transformers>=4.57.0
```

## Inference

You can use OLMo with the standard HuggingFace transformers library:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3.1-32B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3.1-32B-Instruct")
message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# optional verifying cuda
# inputs = {k: v.to('cuda') for k,v in inputs.items()}
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
>> 'Language modeling is  a key component of any text-based application, but its effectiveness...'
```

For faster performance, you can quantize the model using the following method:
```python
AutoModelForCausalLM.from_pretrained("allenai/Olmo-3.1-32B-Instruct", 
    torch_dtype=torch.float16, 
    load_in_8bit=True)  # Requires bitsandbytes
```
The quantized model is more sensitive to data types and CUDA operations. To avoid potential issues, it's recommended to pass the inputs directly to CUDA using:
```python
inputs.input_ids.to('cuda')
```

We have released checkpoints for these models. For post-training, the naming convention is `step_XXXX`. 
**NOTE**: For this model, due to a checkpointing issue, we only are releasing the final few checkpoints. See our other RL jobs for more detailed intermediate checkpoint suite.

To load a specific model revision with HuggingFace, simply add the argument `revision`:
```bash
olmo = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3.1-32B-Instruct", revision="step_1375")
```

Or, you can access all the revisions for the models via the following code snippet:
```python
from huggingface_hub import list_repo_refs
out = list_repo_refs("allenai/Olmo-3.1-32B-Instruct")
branches = [b.name for b in out.branches]
```

### Fine-tuning
Model fine-tuning can be done from the final checkpoint (the `main` revision of this model) or many intermediate checkpoints. Two recipes for tuning are available.
1. Fine-tune with the OLMo-core repository:
```bash
torchrun --nproc-per-node=8 ./src/scripts/official/MODEL.py run01
```
You can override most configuration options from the command-line. For example, to override the learning rate you could launch the script like this:

```bash
torchrun --nproc-per-node=8 ./src/scripts/train/MODEL.py run01 --train_module.optim.lr=6e-3
```
For more documentation, see the [GitHub readme](https://github.com/allenai/OLMo-core).

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
- **Paper:**: https://allenai.org/papers/olmo3


## Evaluation

| Metric | **Olmo 3.1 32B Instruct SFT** | **Olmo 3.1 32B Instruct DPO** | **Olmo 3.1 32B Instruct** | Apertus 70B | Qwen 3 32B (No Think) | Qwen 3 VL 32B Instruct | Qwen 2.5 32B | Gemma 3 27B | Gemma 2 27B | OLMo 2 32B |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Math** | | | | | | | | | | |
| MATH | 74.4 | 86.6 | 93.4 | 36.2 | 84.3 | 95.1 | 80.2 | 87.4 | 51.5 | 49.2 |
| AIME 2024 | 12.7 | 35.2 | 67.8 | 0.31 | 27.9 | 75.4 | 15.7 | 28.9 | 4.7 | 4.6 |
| AIME 2025 | 8.2 | 23.3 | 57.9 | 0.1 | 21.3 | 64.2 | 13.4 | 22.9 | 0.9 | 0.9 |
| OMEGA | 15.5 | 33.3 | 42.2 | 5.6 | 23.4 | 44.0 | 19.2 | 24.0 | 9.1 | 9.8 |
| **Reasoning** | | | | | | | | | | |
| BigBenchHard | 69.0 | 82.1 | 84.0 | 57.0 | 80.4 | 89.0 | 80.9 | 82.4 | 66.0 | 65.6 |
| ZebraLogic | 30.6 | 51.1 | 61.7 | 9.0 | 28.4 | 86.7 | 24.1 | 24.8 | 17.2 | 13.3 |
| AGI Eval English | 71.7 | 79.4 | 79.5 | 61.6 | 82.4 | 89.4 | 78.9 | 76.9 | 70.9 | 68.4 |
| **Coding** | | | | | | | | | | |
| HumanEvalPlus | 80.8 | 85.7 | 86.7 | 42.9 | 83.9 | 89.3 | 82.6 | 79.2 | 67.5 | 44.4 |
| MBPP+ | 61.5 | 63.6 | 65.1 | 45.8 | 67.9 | 69.0 | 66.6 | 65.7 | 61.2 | 49.0 |
| LiveCodeBench v3 | 35.4 | 49.6 | 54.7 | 9.7 | 57.5 | 70.2 | 49.9 | 39.0 | 28.7 | 10.6 |
| **IF** | | | | | | | | | | |
| IFEval | 87.7 | 87.3 | 88.8 | 70.4 | 87.5 | 88.1 | 81.9 | 85.4 | 62.1 | 85.8 |
| IFBench | 29.7 | 36.3 | 39.7 | 26.0 | 31.3 | 37.2 | 36.7 | 31.3 | 27.8 | 36.4 |
| **Knowledge & QA** | | | | | | | | | | |
| MMLU | 79.0 | 81.9 | 80.9 | 70.2 | 85.8 | 88.7 | 84.6 | 74.6 | 76.1 | 77.1 |
| PopQA | 23.7 | 28.5 | 25.0 | 33.5 | 25.9 | 25.7 | 28.0 | 30.2 | 30.4 | 37.2 |
| GPQA | 41.3 | 47.9 | 48.6 | 27.9 | 54.4 | 61.4 | 44.6 | 45.0 | 39.9 | 36.4 |
| **Chat** | | | | | | | | | | |
| AlpacaEval 2 LC | 42.2 | 69.7 | 59.8 | 19.9 | 67.9 | 84.3 | 81.9 | 65.5 | 39.8 | 38.0 |
| **Safety** | 92.1 | 88.9 | 89.5 | 77.1 | 81.6 | 85.8 | 82.2 | 68.8 | 74.4 | 84.2 |


## Model Details

#### Stage 1: SFT
- supervised fine-tuning on the Dolci-Think-SFT-7B dataset. This dataset consits of math, code, chat, and general knowledge queries.
- Datasets: [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/dolci-thinking-sft), [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/dolci-instruct-sft)

#### Stage 2:DPO
- direct preference optimization on the Dolci-Think-DPO-7B dataset. This dataset consits of math, code, chat, and general knowledge queries.
- Datasets: [Dolci-Think-DPO-7B](https://huggingface.co/datasets/allenai/dolci-thinking-dpo), [Dolci-Instruct-DPO](https://huggingface.co/datasets/allenai/dolci-3-instruct-dpo-with-metadata)

#### Stage 3: RLVR
- reinforcement learning from verifiable rewards on the Dolci-Think-RL-7B dataset. This dataset consits of math, code, instruction-following, and general chat queries.
- Datasets: [Dolci-Think-RL-7B](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B), [Dolci-Instruct-RL](https://huggingface.co/datasets/allenai/Dolci-Instruct-RL-7B)


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