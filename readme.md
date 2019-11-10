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

## Intuition

How does this method work? Consider some data distributed along a line in the (x,y) plane like such and the instance to explain in red. 

![one](images/one.jpg | width=100)





## References

Please consider citing our paper if you found this work useful!

```
@article{SlackHilgard2019FoolingLIMESHAP,
	title={How can we fool LIME and SHAP? Adversarial Attacks on Post hoc Explanation Methods},
	author={Dylan Slack and Sophie Hilgard and Emily Jia and Sameer Singh and Himabindu Lakkaraju},
	journal={arXiv},
	year={2019},
}

```

## Contact

This code was developed by Dylan Slack, Sophie Hilgard, and Emily Jia.  Reach out to us with any questions!

Our emails are: [mailto:dslack@uci.edu](mailto:dslack@uci.edu), [ash798@g.harvard.edu](mailto:ash798@g.harvard.edu), and [ejia@college.harvard.edu](mailto:ejia@college.harvard.edu).