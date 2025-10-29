# Error analysis

## Dataset & labels
- The label mapping is not perfect since we are mapping the 28 fine-grained emotions from GoEmotions labels to 7 classes. This mapping is not a standard, so it can lead to systematic errors if a different mapping is used later on. 
- During translation some words get mistranslated from dutch to english (e.g., mistranslations like "one"/"een"). This has the chance to change the meaning of a sentence.
- Neutral, surprise, sadness are the largest contributors to errors (notably neutral ≈ 53, surprise ≈ 47, sadness ≈ 43 on the test set). See Appendix Tables A2 and A3 for errors per true/predicted label.
- Errors correlate with class imbalance (neutral dominates with 711 samples). See Appendix Table A1, 
- Short sentences generally perform worse since there is not a lot of context which creates ambiguity.
- There is a strong imbalance biases predictions toward neutral; small classes (disgust, fear) are unreliable.

### Per-class performance
- Neutral: the model performs well on neutral (F1 = 0.8779) driven by its large support (711 samples).
- Happiness: strong performance (F1 = 0.7179), likely helped by being the second-most frequent class and clearer semantic boundary.
- Fear: very low recall (0.0800) despite precision = 1.0, indicating the model rarely predicts this class; low support (25) is a factor.
- Surprise: poor performance (F1 = 0.2703) even with moderate support; likely confused with neutral/happiness.
- Disgust: unreliable estimate due to very small support (4).
- Summary: class imbalance is a principal driver of per-class performance differences; minority classes need targeted intervention.

## Model performance breakdown
- The model overpredicts neutral and happiness. See Confusion Matrix 1 in Appendix and the classification report in Appendix Table A1. See Appendix Tables A2–A3 for errors per true and predicted label.
### General observations
- Overall accuracy is 0.7975, but the macro-averaged F1 is only 0.535, indicating poor performance on several classes despite high overall accuracy driven by the dominant neutral class.
- The class imbalance (neutral heavily over-represented) skews metrics: weighted averages look reasonable while macro averages reveal weak minority-class performance.
- Misclassification patterns: neutral is the most common error target (130 false predictions as neutral). Happiness is the largest confounder for neutral. A word-cloud analysis suggests many misclassified examples contain general, non-specific language; mistranslations (e.g., "one" vs "een") introduce additional ambiguity.
- The training loss steadily decreases (0.614 -> 0.459 -> 0.371) while validation loss increases (0.744 -> 0.896 -> 0.94) as seen in Fine-Tuning results 1. Validation F1 drops from 0.7726 to 0.7486 and only partially recovers to 0.7567, with accuracy and recall following similar declines. This pattern indicates overfitting beginning after epoch 1. The notebook uses load_best_model_at_end and per‑epoch eval; confirm it restores the epoch with the highest validation F1 (epoch 1). Recommended mitigations: lower learning rate, stronger regularization/dropout, data augmentation or more training data, and class-aware reweighting.

## Error types and qualitative patterns
- Common causes of errors are ambiguous wording, mistranslation, non-specific/general language, overlapping emotions (happiness vs neutral).
- Missing context and short texts cause many errors.

## Tokenization and preprocessing
- Truncating sentences can remove important context.

## Attribution and interpretability
- Run token-level attributions (Integrated Gradients, LIME, SHAP) and attention inspection to find spurious correlations (common words driving neutral predictions).
- Use these to create targeted rules or features.

## Human annotation & disagreement
- Some misclassifications are likely annotation errors or ambiguous examples. 
- Where disagreement about the label is high or the sentence is ambiguous, consider multi-label or "ambiguous" labels.

## Experiments  to run
- Highest-impact, low-effort first:
1. Rebalance classes: class-weighted loss or oversampling minority classes.
2. Clean/repair labels: audit mislabeled examples and problematic translation mappings (correct "one"/"een" issues).


### Additional recommended experiments (WIP)
3. Use additional features: incorporate simple lexical or syntactic features (sentiment score, POS tags, word embeddings) to help distinguish neutral vs. happiness and other subtle distinctions.
4. Error-specific analysis: extract frequent tokens/spans in misclassified examples (See Wordcloud misclassified) and create targeted preprocessing or rules to reduce noise from non-specific language.
5. Threshold tuning: tune decision thresholds per class (or apply class-specific temperature scaling) to increase recall for underrepresented classes without large precision loss.
6. Data augmentation: back-translation or paraphrasing for minority classes; targeted annotation to increase support for fear/surprise/sadness.
7. Ensembling and post-processing: small ensembles or simple rule-based overrides for high-impact confusions (e.g., if model predicts neutral but strong positive sentiment signals exist, re-evaluate).

## Prioritization
- Priority 1 (high gain, low effort): class rebalancing, label cleanup, threshold tuning for recall on fear/surprise.
- Priority 2 (medium effort): data augmentation for minority classes, targeted annotation to increase support.
- Priority 3 (higher effort): architecture changes, ensembling, or expensive re-annotation.

# Conclusion
The model achieves solid accuracy on majority classes but underperforms on minority and subtle emotion classes (fear, surprise, sadness). Focused efforts on class balance, label quality, simple feature augmentation and calibration/threshold tuning should yield the largest improvements with moderate effort.

# Appendix

### Table A1 — Classification report (test set)
| Label      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| anger      | 0.6316    | 0.7059 | 0.6667   | 34      |
| disgust    | 0.6667    | 0.5000 | 0.5714   | 4       |
| fear       | 1.0000    | 0.0800 | 0.1481   | 25      |
| happiness  | 0.6707    | 0.7724 | 0.7179   | 145     |
| neutral    | 0.8350    | 0.9255 | 0.8779   | 711     |
| sadness    | 0.8519    | 0.3485 | 0.4946   | 66      |
| surprise   | 0.5882    | 0.1754 | 0.2703   | 57      |
| **accuracy**   |           |        | **0.7975** | 1042    |
| **macro avg**  | 0.7491    | 0.5011 | 0.5353   | 1042    |
| **weighted avg** | 0.7964    | 0.7975 | 0.7726   | 1042    |

### Table A2 — Errors per True Label
| True Label | Errors |
|------------|--------|
| neutral    | 53     |
| surprise   | 47     |
| sadness    | 43     |
| happiness  | 33     |
| fear       | 23     |
| anger      | 10     |
| disgust    | 2      |

### Table A3 — Errors per Predicted Label
| Predicted Label | Errors |
|-----------------|--------|
| neutral         | 130    |
| happiness       | 55     |
| anger           | 14     |
| surprise        | 7      |
| sadness         | 4      |
| disgust         | 1      |

### Confusion Matrix 1 Confusion Matrix fine-tuned pretrained roberta-base-go_emotions model
![Confusion Matrix fine-tuned pretrained roberta-base-go_emotions model](/nlp_cia/task_9/images/Fine-tuned-confmatrix.png)

### Fine-Tuning results 1
| Epoch | Training Loss | Validation Loss | Accuracy |    F1    | Precision | Recall |
|:-----:|:-------------:|:---------------:|:--------:|:-------:|:---------:|:------:|
|   1   |    0.613600   |     0.743808    | 0.797505 | 0.772562 | 0.796392  | 0.797505 |
|   2   |    0.459200   |     0.896016    | 0.753359 | 0.748641 | 0.771063  | 0.753359 |
|   3   |    0.371500   |     0.939642    | 0.764875 | 0.756729 | 0.773386  | 0.764875 |

### Wordcloud misclassified
![Confusion Matrix fine-tuned pretrained roberta-base-go_emotions model](images/fine-tuned%20preds%20stopword%20misclassified.png)
### Wordcloud correctly classified
![Confusion Matrix fine-tuned pretrained roberta-base-go_emotions model](images/fine-tuned%20preds%20stopword.png)
### F1 Score and support
![Confusion Matrix fine-tuned pretrained roberta-base-go_emotions model](images/fine-tuned%20f1%20per%20class%20with%20support.png)
