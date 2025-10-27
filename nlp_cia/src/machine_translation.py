from transformers import MarianMTModel, MarianTokenizer
import torch

def load_model(model_name="Helsinki-NLP/opus-mt-nl-en"):
    """Load pretrained MarianMT model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    return model, tokenizer, device

def translate_sentences(sentences, batch_size=16):
    """Translate a list of sentences using the pretrained model."""

    model, tokenizer, device = load_model()
    translations = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        # Tokenize and move input tensors to device
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            translated = model.generate(**inputs)
        
        # Decode the translations
        decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        translations.extend(decoded)
    
    return translations

if __name__ == "__main__":
    pass