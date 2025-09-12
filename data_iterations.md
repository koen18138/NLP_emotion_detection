# Sentiment analysis
## Model
The first sentiment analysis model tried was a bert transformer model namely bert-base-uncased using similarly named tokenizer. This model is saved in Model_c021d711

| | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **negative** | 0.00 | 0.00 | 0.00 | 10 |
| **neutral** | 0.78 | 0.99 | 0.87 | 122 |
| **positive** | 0.00 | 0.00 | 0.00 | 25 |
| **accuracy** | | | 0.77 | 157 |
| **macro avg** | 0.26 | 0.33 | 0.29 | 157 |
| **weighted avg**| 0.61 | 0.77 | 0.68 | 157 |

{
  "eval_loss": 0.613419234752655,
  "eval_accuracy": 0.8384,
  "eval_f1": 0.8382026846842687,
  "eval_runtime": 30.4923,
  "eval_samples_per_second": 20.497,
  "eval_steps_per_second": 0.656,
  "epoch": 3.0
}


## Data
Initially i tested the model using http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz. but i found that the training data found in this dataset is not representitative of our intended content to analysis. preprocessing done on this dataset is text cleaning. namely removing every character that is not a-z, A-Z, 0-9, ',' oe "'". in addition to shuffling the training data so there isnt a class inbalance in the small subset used for testing. another problem with the data is that it only contains a label for postive and negative and not neutral


new dataset should remove links