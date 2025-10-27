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
- omputational resources restricted to single GPU training
- Time constraint of 3 training epochs due to academic project timeline
- Class imbalance in available Dutch emotion datasets


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
for training the model we mainly used the go emotions dataset available [here](https://www.kaggle.com/datasets/debarshichanda/goemotions). This dataset includes over 58k samples and has 27 labels that we divided over 7 different labels. We however did fill this data with other data gotten from our peers to have more data on underrepresented classes (fear and disgust). The test data was given by our client CIA and consisted over 1050 lines with mainly only happiness and neutral just like our training data distribution looks like this.
- happiness    33.539827 %
- neutral      30.949460 %
- anger        10.909388 %
- sadness       8.429424 %
- surprise      6.724321 %
- disgust       4.998773 %
- fear          4.448806 %
#### cultural considerations
- Emotion expression varies between Dutch and English cultures
- Direct translations may miss cultural nuances (for example, Dutch directness vs English politeness)
- Idioms and metaphors may not preserve emotional content
#### Multilingual Challenges
- Code-Switching. Model struggles with Dutch-English mixed text
- Dialectical Variations. Performance varies across Dutch dialects
- Translation Artifacts. Some emotional nuance lost in translation process

## Performance Metrics and Evaluation
work in progress

## Explainability and Transparency
work in progress

## Recommendations for Use

#### technical (minimal) requirements 
- transformers >= 4.30.0
- torch >= 1.9.0
- numpy >= 1.20.0

#### Operational Considerations

- might want to build a web application around the pipeline for easy excess for clients
- might want to retrain model on larger dataset and better servers for better accuracy

#### Media Industry Applications
- act like a content moderator for flagging emotionally charged content for review
- track user engagement during different emotional peaks for example different tv shows
- use it for ad placement to match the emotional tone of the content

## Sustainability Considerations

The environmental impact of this model is negligible. during multiple iterations on a single GPU (RTX 4070) over 15 hours my laptop used around 4.5KWH based on some simple calculations. However if the client were to retrain our model on more data it could be a lot more.

the client could opt for investing in different environmentally friendly solutions like solar infrastructure the combat the high energie demands of AI training. Other less invasive solutions could be using a pre trained model instead of training from scratch. 
