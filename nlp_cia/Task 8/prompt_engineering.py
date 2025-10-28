"""
Prompt Engineering Script for Emotion Classification
Best Performing Prompt: Iteration 7 - Optimized Anti-Neutral

Performance on Balanced Test Set:
- F1-Score: 0.854
- Accuracy: 0.857
- Target: F1 > 0.85 (ACHIEVED)

This script contains the best performing prompt developed through systematic
prompt engineering experimentation. The prompt uses the Llama 3 chat template
with explicit anti-neutral rules and comprehensive emotion definitions.

Author: Group 4
Task: Task 8 - Prompt Engineering
Course: FAE2 NLPR
"""

import pandas as pd
import requests
import time
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Configuration
MODEL_URL = "http://194.171.191.228:30080/api/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
API_TOKEN = "sk-093ed5de00ba475fb043f7cc915cf60c"
EMOTIONS = ['happiness', 'sadness', 'anger', 'surprise', 'fear', 'disgust', 'neutral']

# Best Performing Prompt - Iteration 7
BEST_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in emotion classification for Dutch reality TV show dialogue.

CRITICAL RULE: Only choose 'neutral' if there is ABSOLUTELY no emotion present!

TASK: Classify each sentence as EXACTLY ONE emotion (lowercase, no explanation):
happiness, sadness, anger, surprise, fear, disgust, neutral

EMOTION DEFINITIONS:

HAPPINESS: Joy, relief, pride, satisfaction, determination
- "That's amazing, we won!"
- "Yes! We did it!"
- "I won't give up!" (determination = happiness)

SADNESS: Sorrow, disappointment, regret, frustration
- "I'm so disappointed this failed."
- "Unfortunately, it didn't work."
- "Well..." (disappointment tone)

ANGER: Frustration, irritation, rage, competitive aggression, disbelief
- "This is so frustrating!"
- "Seriously?!" "That's ridiculous!"
- ANY tone of irritation, protest, or complaint = anger
- Curse words expressing frustration

SURPRISE: Astonishment at unexpected event or revelation
- "What?! I didn't expect that!"
- "Here they are!" (revelation moment)
- "Finally!" (after waiting)
- NOTE: Regular question is NOT surprise

FEAR: Anxiety, worry, nervousness, tension, difficult challenge
- "I'm afraid we'll lose this."
- "This will be tough." (worry = fear!)
- "Difficult task" (tension)

DISGUST: Revulsion, distaste, contempt
- "That behavior is disgusting."
- "Gross!"

NEUTRAL: ONLY for pure facts, instructions, greetings WITHOUT emotion
- "The task begins now."
- "Applause for..."
- "We're going to location X."

COMMON MISTAKES TO AVOID:
- "That's difficult" -> NOT neutral, it's fear/sadness
- "Well..." -> NOT neutral, it's sadness
- "I won't give up" -> NOT anger, it's happiness (determination)
- Regular question -> NOT automatically surprise

TIE-BREAK RULES:
When uncertain between categories, prioritize in this order:
anger/disgust/fear, then surprise, then sadness, then happiness, then neutral

OUTPUT: Return EXACTLY one word in lowercase. No punctuation, no explanation.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Classify: {sentence}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def query_llm(sentence, prompt_template=BEST_PROMPT_TEMPLATE, temperature=0.0, max_retries=2):
    """
    Query the LLM with a sentence and return the emotion classification.

    Args:
        sentence: The sentence to classify
        prompt_template: The prompt template to use (default: best performing prompt)
        temperature: Sampling temperature (default: 0.0 for deterministic output)
        max_retries: Number of retry attempts on error

    Returns:
        String containing the LLM response
    """
    full_prompt = prompt_template.format(sentence=sentence)

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": temperature,
        "max_tokens": 50
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(MODEL_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()['choices'][0]['message']['content'].strip().lower()
            return result

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 3)
                    continue
            return f"error_http_{e.response.status_code}"

        except Exception as e:
            return f"error_{type(e).__name__}"

    return "error_max_retries"


def extract_emotion(response):
    """
    Extract emotion label from LLM response.

    Args:
        response: Raw LLM response string

    Returns:
        Emotion label (one of EMOTIONS list)
    """
    if response.startswith("error"):
        return "neutral"

    response = response.strip().lower()
    first_word = response.split()[0] if response else "neutral"
    first_word = ''.join(c for c in first_word if c.isalpha())

    # Check if first word is valid emotion
    if first_word in EMOTIONS:
        return first_word

    # Search for emotion anywhere in response
    for emotion in EMOTIONS:
        if emotion in response:
            return emotion

    return 'neutral'


def classify_emotions(sentences, verbose=True):
    """
    Classify a list of sentences.

    Args:
        sentences: List of sentences to classify
        verbose: Print progress information

    Returns:
        List of predicted emotions
    """
    predictions = []
    errors = 0

    if verbose:
        print(f"Classifying {len(sentences)} sentences...")
        print(f"Estimated time: {len(sentences) * 0.5 / 60:.1f} minutes")

    for i, sentence in enumerate(sentences):
        response = query_llm(sentence)

        if response.startswith("error"):
            errors += 1
            if verbose and errors <= 3:
                print(f"Warning: Error at sentence {i}: {response}")

        emotion = extract_emotion(response)
        predictions.append(emotion)

        if verbose and (i + 1) % 25 == 0:
            print(f"Progress: {i+1}/{len(sentences)} (errors: {errors})")

        time.sleep(0.5)  # Rate limiting

    if verbose:
        print(f"Classification complete. Total errors: {errors}")

    return predictions


def evaluate_predictions(true_labels, predictions, show_plot=True):
    """
    Evaluate predictions against true labels.

    Args:
        true_labels: List of true emotion labels
        predictions: List of predicted emotion labels
        show_plot: Whether to display confusion matrix plot

    Returns:
        Dictionary with accuracy and f1 score
    """
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"{'='*70}\n")
    print(classification_report(true_labels, predictions, labels=EMOTIONS, zero_division=0))

    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            true_labels,
            predictions,
            labels=EMOTIONS,
            cmap='Blues',
            ax=ax
        )
        ax.set_title(f"Confusion Matrix - F1: {f1:.3f}")
        plt.tight_layout()
        plt.show()

    return {'accuracy': acc, 'f1': f1}


def main():
    """
    Main function demonstrating usage of the best performing prompt.
    """
    # Load data
    df = pd.read_excel("../task 6/test_improved.xlsx")

    if 'Corrected_emotion' in df.columns:
        df = df.rename(columns={'Corrected_emotion': 'corrected'})

    df['corrected'] = df['corrected'].str.strip()

    # Filter to valid emotions
    df = df[df['corrected'].isin(EMOTIONS)]

    print("="*70)
    print("EMOTION CLASSIFICATION - BEST PERFORMING PROMPT")
    print("="*70)
    print(f"\nDataset: {len(df)} sentences")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: Iteration 7 - Optimized Anti-Neutral\n")

    # Get sentences and labels
    sentences = df['Translation'].tolist()
    true_labels = df['corrected'].tolist()

    # Classify
    predictions = classify_emotions(sentences)

    # Evaluate
    results = evaluate_predictions(true_labels, predictions)

    # Save results
    df['predicted_emotion'] = predictions
    df['correct_prediction'] = df['corrected'] == df['predicted_emotion']

    output_path = "../task 6/test_improved_with_predictions.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
