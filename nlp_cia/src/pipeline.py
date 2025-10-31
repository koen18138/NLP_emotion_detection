
import os

def main():
    # Try and get user input for url
    try:
        url = str(input("Please enter URL to transcribe and classify: ").strip())
    except ValueError:
        print("Invalid input. Please enter a valid URL.")
    
    # Try to download and convert video to mp3
    try:
        from utils import url_to_mp3
        print("Starting video processing...")
        filepath_mp3 = url_to_mp3(url)
    except Exception as e:
        print(f"Error downloading or converting video: {e}")
        return
    
    # Try to transcribe audio to text
    try:
        # Get user input for transcription method
        try:
            transcription_method = str(input("Choose transcription method - 'assembly' (fast, accurate and requires API key) or 'whisper' (slow and accurate) (default 'assembly'): ").strip().lower())
        except ValueError:
            print("Invalid input for transcription method. Defaulting to 'assembly'.")
            transcription_method = 'assembly'

        # Default to 'assembly' if input is invalid
        if transcription_method == 'whisper':
            from whisper import transcribe_to_df, create_asr_pipeline
            print("Starting transcription using Whisper...")
            pipe = create_asr_pipeline()
            df = transcribe_to_df(pipe, filepath_mp3, return_timestamps="sentence")
        else:
            from assembly import transcribe_and_create_excel, API_KEY
            print("Starting transcription using assembly...")
            _, df = transcribe_and_create_excel(
                api_key=API_KEY,
                audio_filepath=filepath_mp3,
                get_speaker=False
            ) 
    except Exception as e:
        print(f"Error during transcription: {e}")
        return   
    
    # Try to translate transcribed text
    try:
        from machine_translation import translate_sentences
        print("Starting translation...")
        text = translate_sentences(df['Sentence'].tolist())
    except Exception as e:
        print(f"Error during translation: {e}")
        return
    
    # Add translations to DataFrame
    try:
        df["Translation"] = text
    except ValueError as e:
        print(f"Error adding translations to DataFrame: {e}")
        return
    
    # Try to load model and tokenizer
    try:
        from transformers_ import load_model_and_tokenizer
        print("Loading emotion classification model and tokenizer...")
        tokenizer, model = load_model_and_tokenizer()
    except Exception as e:
        print(f'Error loading emotion classification model and tokenizer: {e}')

    # Try to encode data for inference
    try:
        from transformers_ import load_inference_data
        print("Encoding data for inference...")
        encoded_sentences, _ = load_inference_data(tokenizer, df)
    except Exception as e: 
        print(f'Error encoding data for inference: {e}')
        return
    
    # Try to perform inference and get predictions
    try:
        from transformers_ import get_predictions
        print("Starting emotion classification...")
        predictions = get_predictions(model, encoded_sentences)
        df["Emotion"] = predictions
    except Exception as e:
        print(f'Error performing inference: {e}')
        return
    
    # Try to save output DataFrame to CSV
    try:
        output_path = os.path.join("data", "pipeline_output.csv")

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
            
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error saving output CSV: {e}")
        return

if __name__ == "__main__":
    main()
