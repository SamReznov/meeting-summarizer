from transformers import pipeline
from transcriber import audio_to_text
from summarizer import summarize_text


if __name__ == "__main__":
    audio_path = "denver_extract.mp3"  # Replace with your audio file path
    # Transcribe the audio file
    transcription = audio_to_text(audio_path)
    summarized_text = summarize_text(transcription)

    print("\nOriginal Text:")
    print(transcription)
    print("\nSummarized Text:") 
    print(summarized_text)