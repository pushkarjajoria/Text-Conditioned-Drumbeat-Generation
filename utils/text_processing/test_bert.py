import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from unsupervised_pretraining.create_unsupervised_dataset import get_filenames_and_tags
from utils.text_processing.text_processor import get_bert_mini_embedding


def pca(high_dim_data, plot_name='PCA Reduction'):
    pca = PCA(n_components=2)
    low_dim_data = pca.fit_transform(high_dim_data)

    plt.scatter(low_dim_data[:, 0], low_dim_data[:, 1])
    for i in range(len(low_dim_data)):
        plt.annotate(i, (low_dim_data[i, 0], low_dim_data[i, 1]))
    plt.title(plot_name)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


def analyze_tensors(tensors, plot_name='Histogram of Distances Between Points'):
    distances = []
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            distance = torch.norm(tensors[i] - tensors[j]).item()
            distances.append(distance)

    print("Statistics:")
    print("Min distance:", np.min(distances))
    print("Max distance:", np.max(distances))
    print("Mean distance:", np.mean(distances))
    print("Standard Deviation:", np.std(distances))

    # Plotting histogram of distances
    plt.hist(distances, bins=20, edgecolor='k')
    plt.title(plot_name)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == "__main__":
    tags_map = get_filenames_and_tags()
    all_tags = list(tags_map.values())
    random_10_idx = random.sample(range(len(all_tags)), 10)
    random_10_tags = list(map(lambda x: " ".join(all_tags[x][3:]), random_10_idx))
    for i, tag in enumerate(random_10_tags):
        print(f"{i}.) {tag}")
    bert_embedding = list(map(lambda x: get_bert_mini_embedding(x), random_10_tags))

    randomly_generated_sentences = [
        "The sun rises in the east and sets in the west.",
        "Dogs are known for their loyalty and affection.",
        "Reading books can significantly improve your knowledge.",
        "She quickly solved the puzzle and won a prize.",
        "Technology has rapidly evolved over the past decade.",
        "Fresh fruits and vegetables are essential for good health.",
        "Music has the power to elevate our mood and emotions.",
        "Mountains offer a breathtaking view of natural beauty.",
        "Education plays a crucial role in personal development.",
        "Teamwork often leads to greater innovation and productivity."
    ]
    bert_embedding_generated_sentences = list(map(lambda x: get_bert_mini_embedding(x), randomly_generated_sentences))

    print("Bert embedding statistics for music tags from folders")
    analyze_tensors(bert_embedding, "Bert embedding for folder tags")
    print("Bert embedding statistics for 10 random english sentences")
    analyze_tensors(bert_embedding_generated_sentences, "Bert embeddings for random eng sentences")

    # PCA
    np_bert_embedding = np.array(list(map(lambda x: x.numpy(), bert_embedding))).squeeze()
    np_bert_embedding_generated_sentences = np.array(list(map(lambda x: x.numpy(), bert_embedding_generated_sentences))).squeeze()

    pca(np_bert_embedding, "Bert embedding for folder tags")
    pca(np_bert_embedding_generated_sentences, "Bert embeddings for random eng sentences")
