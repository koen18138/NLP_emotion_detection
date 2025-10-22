
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
    
    try:
        from assembly import transcribe_and_create_excel, API_KEY
        print("Starting transcription...")
        _, df = transcribe_and_create_excel(
            api_key=API_KEY,
            audio_filepath=filepath_mp3,
            get_speaker=True
        ) 
    except Exception as e:
        print(f"Error during transcription: {e}")
        return   
    
    try:
        from machine_translation import translate_sentences
        print("Starting translation...")
        text = translate_sentences(df['Sentence'].tolist())
    except Exception as e:
        print(f"Error during translation: {e}")
        return
    
    try:
        df["Translation"] = text
    except ValueError as e:
        print(f"Error adding translations to DataFrame: {e}")
        return
    
    try:
        from transformers_ import load_model_and_tokenizer
        print("Loading emotion classification model and tokenizer...")
        tokenizer, model = load_model_and_tokenizer()
    except Exception as e:
        print(f'Error loading emotion classification model and tokenizer: {e}')
        
    try:
        from transformers_ import load_inference_data
        print("Encoding data for inference...")
        encoded_sentences, _ = load_inference_data(tokenizer, df)
    except Exception as e: 
        print(f'Error encoding data for inference: {e}')
        return
    
    try:
        from transformers_ import get_predictions
        print("Starting emotion classification...")
        predictions = get_predictions(model, encoded_sentences)
        df["Emotion"] = predictions
    except Exception as e:
        print(f'Error performing inference: {e}')
        return
    
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
