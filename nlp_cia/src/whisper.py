import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


# Set device to GPU if available, otherwise use CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Use float16 for GPU, float32 for CPU for optimal performance
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# The model ID
# model_id = "openai/whisper-large-v3"
model_id = "openai/whisper-base"
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


if __name__ == "__main__":
    pass