import os

from preprocessing import process_videos_in_directory, transcribe_and_process_audio_directory
from utils import load_model_and_tokenizer, load_inference_data
from sentiment_analysis_eval import get_predictions


if __name__ == "__main__":
    print("Starting video processing...")
    process_videos_in_directory()
    transcribe_and_process_audio_directory()
    


    print("Starting Sentiment Classification...")
    tokenizer, model = load_model_and_tokenizer(model_dir="./models/Model_c021d711")
    sentences, df = load_inference_data(tokenizer)
    preds = get_predictions(model, sentences)
    print(f"Predictions: {preds}")
    df['predicted_sentiment'] = preds
    output_path = os.path.join("data", "classification", "sentence_sentiment.csv")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    df.to_csv(output_path, index=False)
