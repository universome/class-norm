# About

This repo contains the code for the [Class Normalization for Continual Zero-Shot Learning paper](https://arxiv.org/abs/2006.11328) from ICLR 2021:
- the code to reproduce ZSL and CZSL results
- the proposed CZSL metrics (located in `src/utils/metrics.py`)
- fast python implementation of the AUSUC metric

<p style="text-align: center;">
<!-- [[Paper]](https://arxiv.org/abs/2006.11328) [[Google Colab]](https://colab.research.google.com/drive/125-hNotS79DH-6lb3CNcN3PaDZfPxasV?usp=sharing) [Website (TBD)] -->
<a href="https://arxiv.org/abs/2006.11328" target="_blank">[arXiv Paper]</a>
<a href="https://colab.research.google.com/drive/125-hNotS79DH-6lb3CNcN3PaDZfPxasV?usp=sharing" target="_blank">[Google Colab]</a>
<a href="https://openreview.net/forum?id=7pgFL2Dkyyy" target="_blank">[OpenReview Paper]</a>
</p>
<!-- [[Website (TBD)]](https://universome.github.io/class-norm-for-czsl) -->

In this project, we explored different normalization strategies used in ZSL and proposed a new one (class normalization) that is suited for deep attribute embedders.
This allowed us to outperform the existing ZSL model with a simple 3-layer MLP trained just in 30 seconds.
Also, we extended ZSL ideas into a more generalized setting: Continual Zero-Shot Learning, proposed a set of metrics for it and tested several baselines.

<div style="text-align:center">
<img src="images/class-norm-illustration.jpg" alt="Class Normalization illustration" width="500"/>
</div>

# Installation & training
### Data preparation
#### For ZSL
For ZSL, we tested our method on the standard GBU datasets which you can download from [the original website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly).
It is the easiest to follow our [Google Colab](class-norm-for-czsl.ipynb) to reproduce the results.

#### For CZSL
For CZSL, we tested our method on SUN and CUB datasets.
In contrast to ZSL, in CZSL we used raw images as inputs instead of an ImageNet-pretrained model's features.
For CUB, please follow the instructions in the [A-GEM repo](https://github.com/facebookresearch/agem). Note, that CUB images dataset are now to be downloaded manually from [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), but we used the same splits as A-GEM.
Put the A-GEM splits into the CUB data folder.

For SUN, download the data from the [official website](https://cs.brown.edu/~gmpatter/sunattributes.html), put it under `data/SUN` and then follow the instructions in [scripts/sun_data_preprocessing.py](scripts/sun_data_preprocessing.py)

### Installing the `firelab` dependency
You will need to install [firelab library](https://github.com/universome/firelab) to run the training:
```
pip install firelab
```

### How to run ZSL training
Please, refer to this [Google Colab notebook](class-norm-for-czsl.ipynb): it contains the code to reproduce our results.

## How to run CZSL training
To run CZSL training you will need to run the command:
```
python src/run.py -c basic|agem|mas|joint -d cub|sun
```
Please note, that by default we load all the data into memory (to speed up things).
This behaviour is controled by the `in_memory` flag in the config.

# Results
## Zero-shot learning results
<div style="text-align:center">
<img src="images/zsl-results-table.jpg" alt="ZSL results" style="max-width: 500px"/>
</div>

## Continual Zero-Shot Learning results
<div style="text-align:center">
<img src="images/czsl-results-table.jpg" alt="CZSL results" style="max-width: 500px"/>
</div>


## Training speed results for ZSL
<div style="text-align:center">
<img src="images/training-speed-results.jpg" alt="Training speed results" style="max-width: 500px"/>
</div>
