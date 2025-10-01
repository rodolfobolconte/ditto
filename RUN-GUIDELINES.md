# Guidelines to Run

## 1 - Creating Environment with Conda:

Can run these commands first:

```
conda create -n ditto python=3.7.7
conda activate ditto
conda install -c conda-forge nvidia-apex
```

## 2 - Install Microsoft C++ Build Tools

It is necessary to install Microsoft C++ Build Tools before to install requirements, cause exists some packages depends C extensions, so download the tool in these link [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/), after that install the Microsoft C++ Build Tools.

## 3 - Errors in `requirements.txt`:

```gensim==3.8.1
numpy==1.19.2
regex==2019.12.20
scipy==1.3.2
sentencepiece==0.1.85
sklearn==0.0
spacy==3.1
# torch==1.9.0+cu111 # Don't install with this line, try the command bellow out of requirements.txt:
# pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
transformers==4.9.2
tqdm==4.41.0
jsonlines==1.2.0
nltk==3.5
tensorboardX # new package in this list
```

- Don't try to install torch with the requirements, because it not exists in torch site apparently, instead this, try to install using pip and torch_stable link;
- The `tensorboardX` is necessary to run `train_ditto.py`, so just add it in `requirements.txt`.

After that, install torch out of requirements and after that, install the requirements:

```
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## 4 - Error in Code

It is need `stopwords` from `nltk` package to run `train_ditto.py`, so you can download it in the file adding in the first lines of the code:

```
import nltk
nltk.download('stopwords')
```

## 5 - Error in Package

The error `ImportError: cannot import name 'container_abcs' from 'torch._six'` can appears in `envs\ditto\lib\site-packages\apex\amp\_amp_state.py`, from the specific lines of the file:

```
import os
import torch

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 0: # change 0 by 1
    import collections.abc as container_abcs
else:
    from torch._six import container_abcs
```

This error occurs because the version of cuda in the if statement, so you can change 0 by 1 in the if.

## 6 - Command to run Ditto

You can run the train step with this command:

`python train_ditto.py --task Structured/DBLP-ACM --batch_size 32 --max_len 128 --lr 3e-5 --n_epochs 20 --finetuning --lm roberta --fp16 --da drop_col`

`python train_ditto.py --task Textual/Abt-Buy --batch_size 32 --max_len 128 --lr 3e-5 --n_epochs 20 --finetuning --lm roberta --fp16 --da drop_col`