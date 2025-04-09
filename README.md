# Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs 

This repository contains the datasets and code for our paper "Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs"

![AFCE Cover](images/AFCE.png)

## Datasets

If you just want to explore our experiment dataset. We provide a simple csv version of the dataset `./Data`. The format for each 
task is as follows:

**Multiple Choices Questions**
1. **MMLU:** The dataset is cleaned and formatted as: [[question],[A],[B],[C],[D],[answer]] (The order of data is shuffled)
2. **GPQA:** The dataset is cleaned and formatted as: [[question],[A],[B],[C],[D],[answer]] (The order of data is shuffled)

**Open Questions**
1.  **NQ-open:** Coming soon.
2. **SimpleQA:** Coming soon.

## Click and Run Inference using our methods

We have created several simple examples for how to run our experiments, such as AFCE on MMLU, GPQA, SimpleQA, and NQ-open. The code is in `./code`.
To add and test more data please modify `dataloader.py`. To test different models please modify `llm_api_utils.py`.
In order to run our code, you first need to install environment from `requirements.txt`.
For different tasks, the `run_script.sh` file or use commands(examples attached below) contains instructions to run the experiments:

**Overconfidence**
1. AFCE: `AFCE.py`
2. Quiz-like: `quiz_like.py`
3. Top-k: `top_k.py`
4. Verbalized Confidence method (Vanilla): `vanilla.py`
5. Sampling based method: `sampling.py`
6. Probability based method: Coming soon

**Overplacement**
1. Overplacement: `overplacement.py`

**Demographic groups**
1. Demographic groups: `demographic.py`

**Ablation Study**
1. Number of questions: You can adjust the number via the `questions_per_quiz` parameter in `AFCE.py`.
2. Question order: You can change the order of questions by shuffling the dataset and then running `AFCE.py`.


## 🔧 Usage

Install dependencies before running:

```bash
pip install -r requirements.txt
```

Fill in API keys into `llm_api_utils.py`
```
# Replace with your actual API key
GROQ_API_KEY = ''
ANTHROPIC_API_KEY = ''
OPENAI_API_KEY = ''  
DEEP_SEEK_API_KEY = ''
```



To run the AFCE method on a dataset (e.g., MMLU) using GPT-4o, use the following command:

```bash
# 1. AFCE.py
python3 ./code/AFCE.py \
  --datasets "test" "test1" \
  --output_dir "results/" \
  --model "gpt-4o" \
  --questions_per_quiz 10

# 2. quiz_like.py
python3 ./code/quiz_like.py \
  --datasets "test" \
  --output_dir "results/" \
  --model "gpt-4o" \
  --questions_per_quiz 10

# 3. vanilla.py
python3 ./code/vanilla.py \
  --datasets "test" \
  --output_dir "results/" \
  --model "gpt-4o"

# 4. top_k.py
python3 ./code/top_k.py \
  --datasets "test" \
  --output_dir "results/" \
  --model "gpt-4o"

# 5. sampling.py
python3 ./code/sampling.py \
  --datasets "test" \
  --output_dir "results/" \
  --model "gpt-4o" \
  --temperature 0.7 \
  --num_samples 10

# 6. overplacement.py
python3 ./code/overplacement.py \
  --datasets "test" \
  --output_dir "results/" \
  --model "gpt-4o"

# 7. demographic.py
python3 ./code/demographic.py \
  --datasets "test" \
  --output_dir "results/" \
  --model "gpt-4o" \
  --race "White" \
  --age "Young adult (18--24)" \
  --gender "Male"
```
## Acknowledgement
We thank the anonymous reviewers for their feedback on our paper.

## Citation information

If you use this code, please cite our paper:

```
@misc{xu2025llmconfidence,
  title={Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs},
  author={Chenjun Xu, Bingbing Wen, Bin HAN, Robert Wolfe, Lucy Lu Wang, Bill Howe},
}
```
