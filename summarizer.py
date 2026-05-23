from transformers import AutoTokenizer, BartForConditionalGeneration
import torch


# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
model.to("cuda" if torch.cuda.is_available() else "cpu")


def chunk_text(
    text,
    chunk_size=2000
):
    """
    Split long text into smaller text chunks.
    """

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[
            i:i + chunk_size
        ]

        chunks.append(chunk)

    return chunks


def summarize_text(text):
    """
    Summarize the input text using BART.
    """

    text_chunks = chunk_text(text)

    print(
        f"\nTotal chunks: "
        f"{len(text_chunks)}"
    )

    all_summaries = []

    for index, chunk in enumerate(text_chunks):
            f"\nSummarizing chunk "
            f"{len(text_chunks)}"
            print(chunk)


            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="max_length",max_length=1024).to(model.device)
            
            #We are moving the input tensors to the same device as the model
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            summary_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=150, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            all_summaries.append(summary)

    final_summary = " ".join(all_summaries)
    return final_summary
