
## Model Overview
#### Architecture
Our model employs a RobertaForSequenceClassification architecture, leveraging the multilingual capabilities of RoBERTa-base. Key technical specifications include:

- Base Model: RoBERTa-base (125M parameters) pretrained on GoEmotions
- Classification Head: Custom linear layer mapping 768-dimensional hidden states to 7 emotion classes
- Input Processing: WordPiece tokenization with 50,265 vocabulary size
- Maximum Sequence Length: 256 tokens
- Attention Mechanism: 12-layer, 12-head self-attention with 768 hidden dimensions

#### Design Choices

We preserved the pretrained encoder weights to maintain multilingual understanding while reinitializing the classification head for our 7-class problem. We did this because it would be less effecient to keep the amount of heads the same. We selected RoBERTa over BERT for superior performance on sentiment tasks and chose 256 token length to balance computational efficiency with capturing emotional context

#### Purpose
This model was developed to enable automated emotion detection in Dutch media content. We did this for our client CIA (content intelligence agency). Our client did not want to rely on expensive and time consuming API,s

We wanted to provide the CIA with automated content analysis capabilities
Supporting research on emotional expression in Dutch digital media
Enabling emotion monitoring for possibly audience engagement metrics or emotional profiling 

#### Development Context
Key Assumptions:

- Emotions translate conceptually between English and Dutch
- Currently the model is only capable to be used on text data not video (the pipeline will solve this)
- The model only outputs 7 emotions we did this in collaboration with the client(anger, disgust, neutral, fear, surprise, happiness, sadness) This might not be sufficient for high level emotion detection

#### Constraints:

- Limited availability of native Dutch emotion-labeled data
- Computational resources restricted to single GPU training
- Time constraint of 3 training epochs due to academic project timeline
- Class imbalance in available Dutch emotion datasets
- Short sentences generally perform worse due to lack of context and resulting ambiguity. 


## intended use cases

### primary use cases
- Media Transcript Analysis like detecting emotional arcs in Dutch television programs and podcasts
- Social Media Monitoring like analyzing audience emotional responses to Dutch media content
- Content Recommendation like emotion-based content filtering for streaming platforms
- Subtitle Analysis like emotional tone assessment for international content localization

#### Specific Example
A Dutch broadcasting company could use this model to analyze viewer comments on their streaming platform, identifying which scenes give strong emotional responses and adjust content production accordingly.

### out of scope use cases
- Clinical mental health assessment or diagnosis
- Legal evidence or forensic analysis
- High-stakes decision-making without human oversight
- Real-time safety-critical applications
- Individual psychological profiling

## Dataset Details
For training the model we mainly used the go emotions dataset available [here](https://www.kaggle.com/datasets/debarshichanda/goemotions). This dataset includes over 58k samples and has 27 labels that we divided over 7 different labels. We however did fill this data with other data gotten from our peers to have more data on underrepresented classes (fear and disgust). The test data was given by our client CIA and consisted over 1050 lines with mainly only happiness and neutral just like our training data distribution looks like this.
- happiness    33.54 %
- neutral      30.95 %
- anger        10.91 %
- sadness       8.43 %
- surprise      6.72 %
- disgust       5.00 %
- fear          4.45 %
#### cultural considerations
- Emotion expression varies between Dutch and English cultures
- Direct translations may miss cultural nuances (for example, Dutch directness vs English politeness)
- Idioms and metaphors may not preserve emotional content
#### Multilingual Challenges
- Code-Switching. Model struggles with Dutch-English mixed text
- Dialectical Variations. Performance varies across Dutch dialects
- Translation Artifacts. Some emotional nuance lost in translation process (e.g, idioms, colloquial speech)
- Translation Artifacts. Specific issues like mistranslations (e.g., "one"/"een") can change sentence meaning and introduce ambiguity.
- Label Mapping. The non-standard mapping from 28 fine-grained GoEmotions labels to 7 classes can lead to systematic errors if a different mapping is used in the future.

## Performance Metrics and Evaluation
The model was evaluated on a test set of 1042 samples provided by the client (CIA).

### Summary Metrics
| Metric | Value | Comment |
|:---|:---|:---|
| **Overall Accuracy** | **0.7975** | High overall accuracy is driven by the majority classes (neutral and happiness). |
| **Macro-Averaged F1** | **0.5353** | Indicates poor performance on several minority classes, revealing the weakness of the model when class imbalance is not factored in. |
| **Weighted-Averaged F1** | **0.7726** | Higher weighted F1 suggests reasonable performance across the dataset when accounting for class size. |

### Per-Class Performance (F1-score and Support)
Performance varies significantly due to class imbalance.

| Label | F1-score | Precision | Recall | Support | Performance Observation |
|:---|:---|:---|:---|:---|:---|
| neutral | **0.8779** | 0.8350 | 0.9255 | 711 | Performs well, driven by large support. |
| happiness | **0.7179** | 0.6707 | 0.7724 | 145 | Strong performance, second-most frequent class. |
| anger | **0.6667** | 0.6316 | 0.7059 | 34 | Moderate performance. |
| sadness | 0.4946 | 0.8519 | 0.3485 | 66 | Lower recall indicates missed predictions. |
| disgust | 0.5714 | 0.6667 | 0.5000 | 4 | Unreliable estimate due to very small support. |
| surprise | 0.2703 | 0.5882 | 0.1754 | 57 | Poor performance; often confused with neutral/happiness. |
| fear | **0.1481** | 1.0000 | **0.0800** | 25 | Very low recall, rarely predicts this class. |

### Error Analysis and Limitations

- The model shows signs of overfitting starting after epoch 1, with validation loss increasing ($0.744 \rightarrow 0.896 \rightarrow 0.94$) while training loss steadily decreases ($0.614 \rightarrow 0.371$).
- The model overpredicts neutral and happiness. Neutral is the most common error target (130 false predictions as neutral).
- Neutral, surprise, and sadness are the largest contributors to errors (53, 47, and 43 errors respectively on the test set).
- Common causes of errors include ambiguous wording, non-specific/general language, overlapping emotions (happiness vs neutral), and the loss of context in short texts or due to truncation.
- Class imbalance is the biggest factor of the performance differences. Small classes (disgust, fear) are unreliable and predictions are strongly biased toward neutral.

![Error Analysis](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-4-1/blob/main/nlp_cia/task_9/error_analysis_fine_tuned_pretrained_transformer_model.md)

## Explainability and Transparency
work in progress

## Recommendations for Use

#### technical (minimal) requirements 
- transformers >= 4.30.0
- torch >= 1.9.0
- numpy >= 1.20.0

#### Operational Considerations

- Might want to build a web application around the pipeline for easy excess for clients.
- Might want to retrain model on larger dataset and better servers for better accuracy.
- Implement class-weighted loss, oversampling minority classes and/or improve representation to improve minority class performance (e.g., fear, disgust, surprise).
- Check and clean/repair labels, specifically addressing problematic translation mappings (like "one"/"een" issues) to reduce systematic errors.
- Lower the learning rate, use stronger regularization/dropout, and/or use the model checkpoint from the best validation F1 (likely epoch 1).
- Consider tuning decision thresholds per class to increase recall for underrepresented classes like fear.

#### Media Industry Applications
- Act like a content moderator for flagging emotionally charged content for review.
- Track user engagement during different emotional peaks for example different tv shows.
- Use it for ad placement to match the emotional tone of the content.

## Sustainability Considerations

The environmental impact of this model is negligible. during multiple iterations on a single GPU (RTX 4070) over 15 hours my laptop used around 4.5KWH based on some simple calculations. However if the client were to retrain our model on more data it could be a lot more.

The client could opt for investing in different environmentally friendly solutions like solar infrastructure the combat the high energie demands of AI training. Other less invasive solutions could be using a pre trained model instead of training from scratch.