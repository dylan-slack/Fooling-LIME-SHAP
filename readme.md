# Adversarial Attacks on Post Hoc Explanation

This is the code for our paper, "How can we fool LIME and SHAP? Adversarial Attacks on Post hoc Explanation Methods."

Read the [paper](https://arxiv.org/abs/1911.02508).

## Getting started

Setup virtual environment and install requirements:

```
conda create -n fooling_limeshap python=3.7
source activate fooling_limeshap
pip install -r requirements.txt
```

You should be able to run the code now!

We provide a short walk through on COMPAS in `COMPAS_Example.ipynb`.  This is a nice place to get started to see how our method works.  The full experiments from the paper can be found in `compas_experiment.py`, `cc_experiment.py`, and `german_experiment.py`.  