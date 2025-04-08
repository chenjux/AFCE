
# Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs Official

This repository contains the datasets and codes for our paper "Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs"

![](imges/AFCE.pdf)

## Datasets

If you just want to explore our experiment dataset. We provide a simple json version of the dataset ``./Data``. The format for each 
task is as follows:

**Multiple Choices Questions**
1. **MMLU:** The dataset is cleaned and formatted as: [[question],[A],[B],[C],[D],[answer]] (The order of data is shuffled)
2. **GPQA:** The dataset is cleaned formatted as: [[question],[A],[B],[C],[D],[answer]] (The order of data is shuffled)

**Open Questions**
1.  **NQ-open:** To be coming soon.
2. **SimpleQA:** To be coming soon.

## Click and Run Inference using our methods

We have created a simple click-and-run example for how to use AFCE on MMLU, GPQA, SimpleQA, and NQ-open. The code are in ``./AFCE``.
In order to run our code, you first need to install environment from ``./ModelTraining/environment.yml``.
For different task, the runscript.sh file contain the instructions to run the experiments:

**Overconfidence**
1. AFCE: ``AFCE.py``
2. Quiz-like: ``quiz_like.py``
2. Top-k: ``top_k.py``
3. Verbalized Confidence method (Vanilla): ``vanilla.py``
5. Sampling based method: ``sampling.py``
6. Probability based method: To be coming soon

**Overplacement**
1. Overplacement:``overplacement.py``

**Demographic groups**
1. Demographic groups: ``demographic.py``

**Ablation Study**
1. Numbers of questions: It can be achieved by adjust parameters in ``AFCE.py``
2. Change order of questions: It can be achieved by shifting order of Data, then apply ``AFCE.py``

## Acknowledgement
We thank the anonymous reviewers for their feedback on our paper.

## Citation information

If you use this code, please cite our paper:

```
@inproceedings{...,
  title={Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs},
  author={...},
}
```
