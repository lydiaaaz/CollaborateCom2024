# Multi-level Representation Learning with Neural Hawkes Process for Information Diffusion Prediction

This is the implementation of Multi-level Representation Learning with Neural Hawkes Process for Information Diffusion Prediction (CollaborateCom 2024)

## Requirements

- python == 3.7.0
- pytorch == 1.9.1
- dgl == 0.8.2
- scikit-learn == 1.0
- numpy == 1.21.5
- pandas == 1.1.5
- tqdm == 4.62.3

## Dataset

- files in the directory `./data` are examples of preprocessed datasets that the top 10 lines and the last 10 lines are kept.

## Running command

```sh
#Weibo
python main.py --dataset weibo
#Twitter
python main.py --dataset twitter
#APS
python main.py --dataset aps
```

After running, the log file, results, and trained model are saved under the directories of `log`, `saved_results,` and `saved_models`  respectively.

