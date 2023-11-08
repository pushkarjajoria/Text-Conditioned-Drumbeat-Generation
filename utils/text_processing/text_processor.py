from transformers import BertTokenizer, BertModel
import torch


def get_bert_mini_embedding(texts):
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model tokenizer
    tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-512_A-8')

    # Encode batch of text
    # Updated to handle batch of texts
    inputs = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    # Load pre-trained model
    model = BertModel.from_pretrained('google/bert_uncased_L-4_H-512_A-8').to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Forward pass, get hidden states
    with torch.no_grad():
        outputs = model(**inputs)

    # Only take the output embeddings from the last layer
    last_hidden_states = outputs.last_hidden_state

    # We will use the mean of the last layer hidden states as the sentence vector
    sentence_embeddings = last_hidden_states.mean(dim=1)

    return sentence_embeddings


if __name__ == "__main__":
    # Test the function
    embedding = get_bert_mini_embedding("This is a test sentence.")
    print(embedding)