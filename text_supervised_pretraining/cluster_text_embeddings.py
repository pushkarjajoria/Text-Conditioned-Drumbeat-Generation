import os
import pickle
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Subset
from DDPM.main import load_config, load_or_process_dataset
from text_supervised_pretraining.model import CLAMP
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from tqdm import tqdm


# Function to save data
def save_data(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


# Function to load data
def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def perform_clustering(X, n_clusters=10):
    """Perform KMeans clustering and return labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return kmeans.labels_


def extract_keywords(texts, labels):
    """Extract keywords for each cluster."""
    cluster_texts = defaultdict(list)
    for text, label in zip(texts, labels):
        cluster_texts[label].append(text)

    cluster_keywords = {}
    for cluster_id, texts in cluster_texts.items():
        counter = Counter(" ".join(texts).split())
        most_common_words = [word for word, _ in counter.most_common(100)]
        cluster_keywords[cluster_id] = most_common_words
    return cluster_keywords


def generate_word_clouds(cluster_keywords, prefix=""):
    """Generate and save word clouds for each cluster."""
    for cluster_id, keywords in cluster_keywords.items():
        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=set(STOPWORDS).add("Music"),
                              min_font_size=10).generate(" ".join(keywords))
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.title(f"{prefix}Cluster {cluster_id}")
        plt.savefig("text_supervised_pretraining/plots/" + f"{prefix.lower()}cluster_{cluster_id}_wordcloud.png", format='png')
        plt.show()


def prepare_embeddings(embeddings):
    """Prepare embeddings for clustering."""
    return np.stack([embedding.cpu().detach().numpy().flatten() for embedding in embeddings])



# Load configuration and model
config_path = 'DDPM/config.yaml'
config = load_config(config_path)
device = 'cuda' if torch.cuda.is_available() else "cpu"
clamp_model = CLAMP().to(device)
clamp_model.load_state_dict(torch.load(config['clamp_model_path'], map_location=device))
clamp_model.eval()

# Get filenames and tags
train_dataset = load_or_process_dataset(dataset_dir=config['dataset_dir'])
# train_dataset = Subset(train_dataset, indices=range(100))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

pickle_file_path = os.path.join('text_supervised_pretraining', 'text_embeddings_mapping.pkl')

if os.path.exists(pickle_file_path):
    print("Pickle file for text embedding found.")
    # Load the mappings
    mapping = load_data(pickle_file_path)
    all_texts = mapping['all_texts']
    text_embeddings = mapping['text_embeddings']
    bert_embeddings = mapping['bert_embeddings']
    music_text_embeddings = mapping['music_text_embeddings']
    music_bert_embeddings = mapping['music_bert_embeddings']
else:
    print("No pickle file available.")
    all_texts = []  # Initialize an empty list to store all text data
    text_embeddings = []
    bert_embeddings = []
    music_text_embeddings = []
    music_bert_embeddings = []

    for drum_beats, text_data in tqdm(train_loader, desc=f"Generating text embeddings"):
        all_texts.extend(text_data)  # Store original text data

        # Get embeddings for original texts
        original_text_embeddings, original_bert_embeddings = clamp_model.get_text_and_bert_embeddings(text_data)
        text_embeddings.extend(original_text_embeddings)
        bert_embeddings.extend(original_bert_embeddings)

        # Modify texts by appending "Music"
        modified_texts = ["Music " + text for text in text_data]

        # Get embeddings for modified texts
        modified_text_embeddings, modified_bert_embeddings = clamp_model.get_text_and_bert_embeddings(modified_texts)
        music_text_embeddings.extend(modified_text_embeddings)
        music_bert_embeddings.extend(modified_bert_embeddings)

    # Save the mapping with all sets of embeddings
    save_data(pickle_file_path, {
        'all_texts': all_texts,
        'text_embeddings': text_embeddings,
        'bert_embeddings': bert_embeddings,
        'music_text_embeddings': music_text_embeddings,
        'music_bert_embeddings': music_bert_embeddings
    })
X = prepare_embeddings(text_embeddings)
X_bert = prepare_embeddings(bert_embeddings)
X_music_text = prepare_embeddings(music_text_embeddings)
X_music_bert = prepare_embeddings(music_bert_embeddings)

# Perform clustering
n_clusters = 10
text_labels = perform_clustering(X, n_clusters=n_clusters)
bert_labels = perform_clustering(X_bert, n_clusters=n_clusters)
music_text_labels = perform_clustering(X_music_text, n_clusters=n_clusters)
music_bert_labels = perform_clustering(X_music_bert, n_clusters=n_clusters)

# Extract keywords and generate word clouds for each set
text_keywords = extract_keywords(all_texts, text_labels)
bert_keywords = extract_keywords(all_texts, bert_labels)
music_text_keywords = extract_keywords(all_texts, music_text_labels)
music_bert_keywords = extract_keywords(all_texts, music_bert_labels)

# Generate word clouds
generate_word_clouds(text_keywords, prefix="Text ")
generate_word_clouds(bert_keywords, prefix="BERT ")
generate_word_clouds(music_text_keywords, prefix="Music Text ")
generate_word_clouds(music_bert_keywords, prefix="Music BERT ")