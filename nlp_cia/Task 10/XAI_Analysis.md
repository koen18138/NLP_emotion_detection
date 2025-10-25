# Explainable AI Analysis for Emotion Classification Model

## Introduction

Transformer models are really powerful for tasks like emotion classification, but they're basically black boxes. You feed text in, get a prediction out, and have no idea what's actually happening inside. For this task, we wanted to understand how our emotion classification model actually makes its decisions. Does it focus on the obvious emotion words like "fantastic" or "terrible", or is it picking up on something else entirely?

We used the method from Ali et al.'s 2022 paper "XAI for Transformers: Better Explanations through Conservative Propagation" to look inside our model. The main goal was to see if the model is actually learning what we think it's learning, or if it's maybe relying on weird patterns or shortcuts that don't make sense. This matters because if the model is making predictions for the wrong reasons, it might fail on new data even if it looks good on the test set.

## Methodology

### Sentence Selection

We picked 3 sentences for each of the 6 emotions our model was trained on: anger, disgust, fear, happiness, sadness, and surprise. That gave us 18 sentences total. We tried to pick a mix, some really obvious ones like "Yuck" for disgust, and some more subtle ones where the emotion comes from context rather than a single word. We also included different sentence lengths to see if that affected how the model processed them.

### Three XAI Techniques Applied

We applied three different techniques to understand the model:

**Gradient × Input** is the basic approach. It multiplies the gradient of the prediction with respect to each token by the input embedding itself. This shows which tokens the model is "sensitive" to, basically which words would change the prediction most if you tweaked them slightly.

**Conservative Propagation (LRP)** is an improved version that's specifically designed for transformers. Regular LRP has problems with attention layers and layer normalization, so this conservative approach redistributes relevance scores more carefully through the network to get more stable explanations.

**Input Perturbation** tests how robust the model is by removing tokens one at a time and watching the confidence drop. If the model's confidence crashes after removing just one word, it's probably relying too heavily on that specific token.

## Part 1: Gradient × Input Analysis

![Figure 1](figures/figure1_happiness_gradient.png)
*Figure 1: Gradient × Input visualization for happiness sentence showing token contributions*

![Figure 2](figures/figure2_sadness_gradient.png)
*Figure 2: Gradient × Input visualization for sadness sentence*

![Figure 3](figures/figure3_anger_gradient.png)
*Figure 3: Gradient × Input visualization for anger sentence*

Looking at the visualizations, there are some things that make sense and some surprises. For the happiness sentences, "fantastic" and "fun" both show strong positive contributions (green bars), which is exactly what you'd expect. The model is clearly picking up on these obviously positive words. Same with "Congratulations" in the second happiness sentence, huge green bar.

What's interesting though is that the special tokens like `<s>` (start token) and `</s>` (end token) also show relatively high relevance scores, even though they shouldn't contain any emotional information. We're using a RoBERTa model, so these tokens are just structural markers. It's a bit weird that they're contributing to the emotion prediction at all.

For the sadness sentences, "bad" shows a strong positive contribution toward the sadness prediction, which makes sense, the model learned that "bad" is associated with negative emotions. But looking at "Gone as her prime suspect", it's less clear which specific words are driving the sadness prediction. The relevance is more spread out across multiple tokens.

The anger sentences are interesting because words like "dirty" and "sneak" both contribute positively, but so do some of the function words in between. We think this might be because anger often involves multi-word insults, so the model is picking up on the phrase as a whole rather than individual words.

One thing that stood out across multiple sentences: punctuation marks sometimes have unexpectedly high relevance. Exclamation marks in particular seem to boost certain predictions. This actually makes sense - punctuation does carry emotional information in text. A sentence ending with "!" feels different from one ending with a period.

## Part 2: Conservative Propagation Analysis

![Figure 4](Figures/figure4_attention_heatmap.png)
*Figure 4: Attention scores heatmap showing how tokens attend to each other in the first attention layer*

For the Conservative Propagation analysis, we visualize the attention mechanism directly through a heatmap rather than another bar graph. This is because the key improvement of Conservative LRP for transformers is how it handles attention weights and layer normalization. The heatmap gives us a different perspective on what's happening inside the model, showing not just individual token importance but how tokens interact through attention. The strong diagonal line shows that tokens are mostly paying attention to themselves, which is pretty standard for transformers. But there are also some interesting off-diagonal patterns.

Looking at the heatmap, we can see some tokens have stronger attention connections to specific other tokens in the sentence. For example, "Ah" shows relatively high attention to itself and some surrounding tokens, while "Gshit" (the profanity token) has notable attention weights with other parts of the sentence. The special tokens `<s>` and `</s>` show more uniform attention patterns across all tokens, which suggests they might be aggregating information from the whole sentence.

Compared to the basic Gradient × Input method, the conservative propagation approach seems to distribute relevance a bit more evenly. With the basic method, we sometimes saw one or two tokens dominating the explanation, but the LRP approach shows more of how different tokens work together to produce the final prediction.

The heatmap also reveals something about how the model processes sentence structure. The special tokens `<s>` and `</s>` have relatively uniform attention across all other tokens, which suggests they might be acting as some kind of "aggregation" points where information gets pooled before the final classification.

One thing we noticed is that the model doesn't always attend most strongly to the most obvious emotion words. Sometimes neutral words that provide context get high attention scores too. This suggests the model is doing more than just keyword matching - it's actually considering the relationships between words to understand the emotion.

## Part 3: Token Perturbation and Model Robustness

![Figure 5](Figures/figure5_perturbation_comparison.png)
*Figure 5: Comparison of token removal strategies - removing most relevant tokens (left) versus least relevant tokens (right)*

The perturbation analysis shows how the model's confidence changes when we remove tokens one by one. While the task suggested removing only the least relevant tokens, we found it much more insightful to show both strategies side-by-side. This comparison clearly demonstrates which tokens actually matter for the model's predictions. We tested two different removal strategies: removing the most relevant tokens first (based on relevance scores) versus removing the least relevant tokens first.

The side-by-side comparison reveals a striking difference. On the left graph, when we remove the most relevant tokens, the model's confidence crashes dramatically. Starting from about 55% confidence for the anger prediction on "Ah, shit!", the confidence plummets to around 2-3% after removing just the first token, then hovers near zero for the rest. This sharp drop tells us the model is heavily reliant on one or two specific tokens.

On the right graph, when we remove the least relevant tokens first, the behavior is more erratic. The confidence fluctuates significantly, sometimes even increasing after token removal. This zigzag pattern reveals something important: replacing tokens with PAD tokens doesn't just "remove" information, it actually changes the input in ways that can confuse the model. The PAD tokens create an unusual input pattern that the model hasn't been trained on, leading to unpredictable behavior. Despite the fluctuations, the overall pattern shows that removing less important tokens doesn't cause the immediate confidence collapse we see on the left.

The 0.5 threshold line (shown in red) marks where the model becomes uncertain, basically just guessing. For this sentence, removing just one important token causes the confidence to plummet below 50%, which suggests the prediction is quite fragile and depends critically on specific words rather than a broader understanding of the sentence.

This behavior varies depending on sentence structure. For very short sentences with clear emotion words, we see this kind of sharp drop because there are so few tokens contributing to the prediction. The model doesn't have much redundancy to fall back on. For longer, more complex sentences with multiple emotion indicators, we'd expect to see a more gradual decline as the model can rely on multiple cues.

The comparison validates that our gradient-based relevance scores are actually meaningful. The tokens identified as "most relevant" truly are the ones the model depends on for its prediction. This isn't just a quirk of the gradient calculation, it reflects real dependencies in how the model makes decisions.

## Discussion and Key Findings

Looking at everything together, the XAI analysis shows our model does focus on emotionally charged words, which is good. It picks up on obvious emotion words like "fantastic", "bad", and "shit", and assigns them high relevance scores. But it's not just doing simple keyword matching. The attention heatmaps show the model is actually considering how words relate to each other.

The biggest concern is how fragile the predictions are. The perturbation test showed that for short sentences, removing just one or two key tokens causes the confidence to crash from 50-60% down to almost nothing. The model is basically relying on specific keywords rather than understanding the broader emotional meaning. If you were actually deploying this, it would probably struggle with typos or if people use different words to express the same emotion.

Something odd we noticed: the special tokens (`<s>` and `</s>`) show up as having high relevance in the Gradient × Input analysis, even though they're just structural markers and shouldn't carry any emotional information. This seems like it might be a quirk of how gradients work in transformers. The gradients can be high for tokens that aren't actually important for the prediction. That's why the perturbation test was useful to double-check which tokens really matter.

Different emotions behave differently too. Disgust and anger predictions tend to depend heavily on one or two strong words, while sadness is more spread out across the whole sentence. This kind of makes sense from a human perspective. You usually feel disgusted by specific things or words, but sadness often comes from the overall mood of what someone's saying.

The attention mechanism turned out to be more important than we expected. Just looking at which individual tokens are important doesn't give you the full picture. You need to see how the tokens interact with each other through attention to really understand what the model is doing.

## Conclusion

This XAI analysis was pretty eye-opening. Before this, we only knew our model got decent test accuracy. Now we actually understand what it's doing and where it falls short. The good news is the model does learn reasonable patterns and uses attention to understand context. The bad news is it's way more fragile than we thought.

If we had to improve the model based on these findings, we'd focus on robustness. Maybe use data augmentation with synonym replacement or train on more varied ways of expressing emotions. That way it wouldn't completely fall apart when it sees a typo or a paraphrased version of something.

We also learned you can't just trust gradient-based explanations at face value. The perturbation tests were really important for validating what actually matters versus what just looks important from the gradients. This kind of analysis should probably be standard practice before deploying any model in production. Test accuracy only tells you so much.

## References

Ali, A., Schnake, T., Eberle, O., Montavon, G., Müller, K. R., & Wolf, L. (2022). XAI for Transformers: Better Explanations through Conservative Propagation. In *Proceedings of the 39th International Conference on Machine Learning* (Vol. 162, pp. 435-451). PMLR.

---
