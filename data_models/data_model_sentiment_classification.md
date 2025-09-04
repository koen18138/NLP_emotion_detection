# Sentiment Classification Data Model

## Input Data

| Field Name | Type   | Description                   |
|------------|--------|-------------------------------|
| sentence_token       | string | The tokenized input text to be analyzed |

**Example:**
```csv
sentence
value1
value2
```


## Input Data

| Field Name | Type   | Description                   |
|------------|--------|-------------------------------|
| label | string | Predicted sentiment label |
| score | float | Confidence score for the prediction| 

**Example:**
```csv
label,score
positive,0.4
negative,0.7