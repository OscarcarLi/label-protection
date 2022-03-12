# label-protection
Code Repo for paper Label Leakage and Protection in Two-party Split Learning (ICLR 2022).

## Requirements

* Python 3
* Tensorflow >2.0
* scikit-learn

## Dataset

* For demonstration, we have provided a small portion (0.5%) of Criteo dataset in ```dataset/criteo```. 
* Interested readers can use above dataset to test our code.
* For the actual datasets (Avazu, ISIC), readers should download the datasets
Criteo: https://www.kaggle.com/c/criteo-display-ad-challenge/data
Avazu: https://www.kaggle.com/c/avazu-ctr-prediction/data
ISIC: https://www.kaggle.com/c/siim-isic-melanoma-classification/data
and use
```preprocess_criteo_subset.py```
```preprocess_avazu.py``` and ```preprocess_ISIC.ipynb``` to preprocess the corresponding datasets. 

## Run

We provide a script for each dataset to test our protection methods. 

* ```run_script_criteo.py``` for Criteo
* ```run_script_avazu.py``` for Avazu
* ```run_script_isic.py``` for ISIC

In each script, we have provided configurations to run Marvell, max_norm, iso (isotropic gaussian) and no_noise (no perturbation)
The corresponding command line option for each of the perturbation methods are:
* Marvell: ```--noise_layer_function sumKL```
* max_norm: ```--noise_layer_function expectation```
* iso: ```--noise_layer_function white_gaussian```
* no_noise: use ```--noise_layer_function white_gaussian --ratio 0.0```

## Visualization 

* We have provided  ```diff_methods_tradeoff_viz_save_memory-*.ipynb``` to visualize the tradeoff results in tensorboard logs.  

