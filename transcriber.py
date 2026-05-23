from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import librosa


# Load the model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Moving model to GPU...")
model.to(device)


print(f"Loading on device: {device}")

def audio_to_text(audio_path):
    print(f"Loading audio file: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)

    print("Processing audio...")
    print("Converting audio to input features...")
    processed_inputs = processor(audio, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=16_000)
    
    # Move input features to the same device as the model
    input_features = processed_inputs.input_features.to(device)

    print("Generating transcription...")
    generated_ids = model.generate(input_features, return_timestamps=True)

    print("Decoding transcription...")
    print("Decoding tokens to text...")
    decoded_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print("Transcription complete.")
    return decoded_text
