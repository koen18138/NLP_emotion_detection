import os

from assembly import transcribe_and_create_excel, API_KEY
from utils import load_model_and_tokenizer, load_inference_data, url_to_mp3
from sentiment_analysis_eval import get_predictions


if __name__ == "__main__":
    print(os.getcwd())
    print("Please enter URL to transcribe and classify:")
    url = input().strip()
    print("Starting video processing...")
    url_to_mp3(url)
    
    transcript, df = transcribe_and_create_excel(
        api_key=API_KEY,
    )
    print(transcript)
    print(df.head())

    print("Starting Sentiment Classification...")
    tokenizer, model = load_model_and_tokenizer(model_dir="models\\Model_14e03c00")
    sentences, df = load_inference_data(tokenizer)
    preds = get_predictions(model, sentences)
    print(f"Predictions: {preds}")
    df['predicted_sentiment'] = preds
    output_path = os.path.join("data", "classification", "sentence_sentiment.csv")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    df.to_csv(output_path, index=False)
