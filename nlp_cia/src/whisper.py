import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pandas as pd
from typing import Optional

def create_asr_pipeline():
    """
    Create and return an automatic speech recognition (ASR) pipeline using a pre-trained Whisper model.
    The model is loaded with optimizations for either GPU or CPU based on availability.
    
    Returns:
        pipeline: An ASR pipeline ready for transcription tasks.
    """
    # Set device to GPU if available, otherwise use CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Use float16 for GPU, float32 for CPU for optimal performance
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # The model ID
    model_id = "openai/whisper-large-v3"

    # Load the pre-trained Whisper model with specified dtype and memory optimizations
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    print(f"Model {model_id} loaded successfully.")

    # Move the model to the selected device (GPU or CPU)
    model.to(device)
    print(f"Model moved to device: {device}")

    # Load the processor for feature extraction and tokenization
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Processor for {model_id} loaded successfully.")

    # Create an automatic speech recognition pipeline using the model and processor
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, 
        dtype=torch_dtype,
        device=device,
    )
    print("ASR pipeline created successfully.")
    return pipe


def format_timestamp(seconds: Optional[float]) -> str:
    """
    Format seconds as HH:MM:SS.mmm. Returns empty string for None.
    """
    if seconds is None:
        return ""
    try:
        sec = float(seconds)
    except Exception:
        return ""
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds_int = int(sec % 60)
    milliseconds = int((sec - int(sec)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}.{milliseconds:03d}"


def transcribe_to_df(pipe, audio_path: str, return_timestamps: str = "sentence", **pipeline_kwargs) -> pd.DataFrame:
    """
    Use the global `pipe` to transcribe `audio_path` and return a DataFrame with columns:
      - 'Start Time' (formatted)
      - 'End Time' (formatted)
      - 'Sentence'
    Additional kwargs are forwarded to the pipeline call (e.g., chunk_length_s, stride_length_s).
    """
    # Call pipeline; pipeline accepts either a file path or array-like audio input
    result = pipe(audio_path, return_timestamps=return_timestamps, **pipeline_kwargs)

    # Extract segments/chunks depending on pipeline output shape
    segments = result.get("segments") or result.get("chunks") or result.get("words") or result.get("timestamps") or []

    data = []
    if not segments and "text" in result:
        # Fallback: single text result without timestamps
        data.append({"Start Time": 0.0, "End Time": None, "Sentence": result["text"].strip()})
    else:
        for seg in segments:
            # support different key names that may appear in various pipeline versions
            start = seg.get("start") or seg.get("start_time") or seg.get("begin") or seg.get("start_sec") or 0.0
            end = seg.get("end") or seg.get("end_time") or seg.get("finish") or seg.get("end_sec") or None
            text = seg.get("text") or seg.get("sentence") or seg.get("transcript") or ""
            data.append({"Start Time": start, "End Time": end, "Sentence": text.strip()})

    df = pd.DataFrame({
        "Start Time": [format_timestamp(s["Start Time"]) for s in data],
        "End Time":   [format_timestamp(s["End Time"])   for s in data],
        "Sentence":   [s["Sentence"] for s in data],
    })
    return df


if __name__ == "__main__":
    pass