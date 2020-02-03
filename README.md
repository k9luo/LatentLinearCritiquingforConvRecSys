LatentLinearCritiquingforConvRecSys
====================================================================

If you are interested in building up your research on this work, please cite:
```
@inproceedings{sanner:www20,
  author = {Kai Luo and Scott Sanner and Ga Wu and Hanze Li and Hojin Yang},
  title = {Latent Linear Critiquing for Conversational Recommender Systems},
  year = {2020},
  booktitle = {Proceedings of the 29th International Conference on the World Wide Web (WWW-20)},
  address = {Taipei, Taiwan},
  url_paper = {https://ssanner.github.io/papers/www20_llc.pdf}
}
```

# Author Affiliate
<p align="center">
<a href="https://www.utoronto.ca//"><img src="https://github.com/k9luo/DeepCritiquingForVAEBasedRecSys/blob/master/logos/U-of-T-logo.svg" height="80"></a> | 
<a href="https://vectorinstitute.ai/"><img src="https://github.com/k9luo/DeepCritiquingForVAEBasedRecSys/blob/master/logos/vectorlogo.svg" height="80"></a> | 
</p>

# Algorithm Implemented
1. LP Option1
1. LP Option2
1. LP Option3

# Dataset
1. Amazon CDs&Vinyl,
2. Beer Advocate,

We don't have rights to release the datasets. Please ask permission from Professor [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/).

Please refer to the [`preprocess` folder](https://github.com/wuga214/DeepCritiquingForRecSys/tree/master/preprocess) for preprocessing raw datasets steps.

# Keyphrase
Keyphrases we used are not necessarily the best. If you are interested in how we extracted those keyphrases, please refer to the [`preprocess` folder](https://github.com/wuga214/DeepCritiquingForRecSys/tree/master/preprocess). If you are interested in what keyphrases we extracted, please refer to the [`data` folder](https://github.com/wuga214/DeepCritiquingForRecSys/tree/master/data).

# Example Commands

### Reproduce Critiquing
```
python reproduce_critiquing.py --data_dir "data/beer/" --dataset_name beer/ --save_path beer/critiquing_results.csv --num_users_sampled 10 --critiquing_model_name LP1Simplified
```
