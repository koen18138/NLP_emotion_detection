# Readme

This file explains, in very simple terms, what the pieces in this project do and how the main pipeline works.

---

## High-level flow (pipeline.py -> main)
1. Ask the user for a YouTube URL.
2. Download the video's audio and convert it to `mp3` (utils.url_to_mp3).
3. Transcribe the audio with AssemblyAI (speaker optional) or Whisper and split into sentences (assembly.transcribe_and_create_excel).
4. Translate each sentence from Dutch to English (machine_translation.translate_sentences).
5. Load a fine-tuned transformer model + tokenizer (transformers_.load_model_and_tokenizer).
6. Encode the translated sentences for the model (transformers_.load_inference_data / encode_data).
7. Run the model to get emotion predictions (transformers_.get_predictions).
8. Save results to `data/pipeline_output.csv`.

Errors at each step are printed and stop the pipeline.

---

## Files and main responsibilities

- pipeline.py
  - Orchestrates the whole process from download → transcript → translate → classify → save.
  - Calls functions from other modules and writes `data/pipeline_output.csv`.

- utils.py
  - Function: `url_to_mp3(url, output_dir="data\\audio")`
  - Uses pytubefix (YouTube) to download audio from a YouTube URL and rename to `.mp3`.
  - Returns the mp3 file path.
- whisper.py
  - Function: `transcribe_to_df(pipe, audio_path: str, return_timestamps: str = "sentence", **pipeline_kwargs), create_asr_pipeline()`.
  - Uses Whisper "openai/whisper-large-v3" to transcribe audio.
  - Creates a ASR pipeline using and returns a Dataframe with `Start time`, `End time` and `Sentence`.
  - Note: Whisper pipeline is not able to get start and end time.
- assembly.py
  - Function: `transcribe_and_create_excel(api_key, audio_filepath=..., get_speaker=True, ...)`
  - Uses AssemblyAI to transcribe audio.
  - Optionally extracts speaker labels and splits utterances into sentences.
  - Returns the raw `transcript` object and a pandas `DataFrame` with columns like `Sentence`, `Start Time`, `End Time`, and `Speaker`.
  - Also writes CSV or XLSX to disk.
  - Note: API_KEY is defined in this file — replace with your own key if needed.

- machine_translation.py
  - Function: `translate_sentences(sentences, batch_size=16)`
  - Loads the Helsinki-NLP MarianMT Dutch→English model and tokenizer.
  - Translates sentences in batches and returns a list of translations.
  - Uses GPU if available.

- transformers_.py
  - Function: `load_model_and_tokenizer(model_dir="models/model_pretrained")`
    - Loads a saved tokenizer and sequence-classification model from the given directory (local files only).
  - Function: `load_inference_data(tokenizer, inference_data=None, file_path=..., text_column='Translation')`
    - Cleans text, tokenizes using the tokenizer, returns encoded tensors and the DataFrame.
  - Function: `encode_data(texts, tokenizer, labels=None, max_len=128)`
    - Tokenizes texts and returns PyTorch tensors for `input_ids` and `attention_mask`. If `labels` given, encodes them too.
  - Function: `text_cleaning(text)`
    - Very small cleaning: remove HTML, brackets, and non-alphanumeric characters.
  - Function: `get_predictions(model, encodings)`
    - Runs model in eval mode, gets logits, argmax, converts indices back to label strings via a fitted LabelEncoder.
  - Label mapping is defined in the file (happiness, anger, surprise, sadness, disgust, fear, neutral).

---

## Important paths & outputs
- Audio download: `data\audio\output.mp3` (returned path from utils.url_to_mp3)
- Transcription CSV: default from assembly is `data\transcription\csv\transcription.csv` (but pipeline uses returned DataFrame)
- Final pipeline output: `data/pipeline_output.csv` (written by pipeline.py)

---

## How to run (simple)
## **Change diretory to project root**
```
cd nlp_cia
```
## **install requirements using poetry**
```
poetry install
```
## **Download model from:**
Onedrive, [pre trained transformer model](https://teams.microsoft.com/l/message/19:9421d7be8f234e209f3981152310a85f@thread.v2/1761136294063?context=%7B%22contextType%22%3A%22chat%22%7D)
## **Place model in:**
`nlp_cia/models/model_pretrained`
## **Ensure AssamblyAI API key is valid**
## **Run pipeline using**
```
poetry run python src/pipeline.py
```
## **When prompted paste link into command line**

# Running notebooks
## **Change directory to project root**
```
cd nlp_cia
```
## **install requirements using poetry**
```
poetry install
```
## **Install ipykernel if not installed**
if ipykernel not installed do
```
poetry add ipykernel
```
## **Create jupyter kernel from poetry enviroment**
```
poetry run python -m ipykernel install --user --name <your-kernel-name>
```
## **Restart code editor**


---

## Notes
- AssemblyAI needs a valid API key and internet access.
- The translation and transformer models are downloaded from Hugging Face on first run (large files).
- If a file or model is missing the code prints an error and stops.
- Replace the hard-coded API key in `assembly.py` with your own key if you want to run it.
