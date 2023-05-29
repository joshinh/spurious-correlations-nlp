# spurious-correlations-nlp

This is the code for the paper [Are All Spurious Correlations in Natural Language Alike? An Analysis through a Causal Lens](https://arxiv.org/abs/2210.14011) which was accepted at EMNLP 2022. This code was majorly adapated from three different sources:

1. For the real NLP experiments - https://github.com/technion-cs-nlp/bias-probing
2. For the INLP experiments - https://github.com/shauli-ravfogel/nullspace_projection
3. For the toy synthetic experiments - https://github.com/cjlovering/predicting-inductive-biases

## Running the code

Some of the instructions are common with the instructions from the original codebase.

### 1. Setting up the environment
Create a new conda environment and install libraries:
```bash
pip3 install -r requirements.txt
``` 

### 2. Datasets

Most of the NLP datasets are already prepared and our available at ```nlp/data```. Some which were too large to upload (MNLI training data and MNLI synthetic training data) are uploaded [here](https://drive.google.com/file/d/1RzVsoglufma8gvKxc4hSXTHgHSopm6RR/view?usp=sharing). Overall, these datasets include the MNLI training dataset, MNLI synthetic training datasets, as well as the training and evaluation datasets corresponding to the different groups (e.g. high word overlap).

### 3. Training

To train models either using a simple cross entropy loss on orginal/balanced dataset, or using POE/DFL, navigate to ```nlp/scripts``` and run:

```bash
python3 training.py
``` 

You can modify the training dataset, method, hyperaparameters etc. in ```training.yaml```.

### 3. Probing

To run the probing, you can check out the different configs in ```nlp/configs```. You can then run:

```bash
python3 probing.py --seed=42 --name="baseline" --task_config_file="mnli_lex_class.json" --model_name_or_path="seed:42/baseline" --overwrite_cache
``` 

### 4. INLP

To obtain an invariant representation using the INLP method, you can run:
```bash
python3 debias.py
``` 

The file includes path to the saved model representations which will be debiased -- you will need to modify this according to where you store it.


### 5. Toy Synthetic Experiments

To be updated soon!


## Citation, authors, and contact

### Bibtex

```
@inproceedings{joshi2022spurious,
        author={Nitish Joshi, Xiang Pan and He He},
        title={Are All Spurious Features in Natural Language Alike? An Analysis
through a Causal Lens},
        booktitle={EMNLP},
        year={2022}
}
```

### Authors
[Nitish Joshi](https://joshinh.github.io), [Xiang Pan](https://xiangpan.netlify.app) and [He He](https://hhexiy.github.io)





