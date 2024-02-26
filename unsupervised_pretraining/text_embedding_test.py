import random
from collections import defaultdict
import torch
from matplotlib import pyplot as plt
from numpy.core.defchararray import isnumeric
from tqdm import tqdm

from DDPM.main import load_config
from unsupervised_pretraining.create_unsupervised_dataset import get_filenames_and_tags
from unsupervised_pretraining.model import CLAMP

# Load configuration and model
config_path = 'DDPM/config.yaml'
config = load_config(config_path)
device = 'cuda' if torch.cuda.is_available() else "cpu"
clamp_model = CLAMP().to(device)
clamp_model.load_state_dict(torch.load(config['clamp_model_path'], map_location=device))
clamp_model.eval()

# Get filenames and tags
file_name_and_tags = get_filenames_and_tags(dataset_dir=config['dataset_dir'], filter_common_tags=True)

# Count variants for each file name
same_file_name = defaultdict(int)
for file_tag in file_name_and_tags.values():
    numeric_part = ''.join(filter(str.isdigit, file_tag))
    if numeric_part:
        same_file_name[file_tag.rstrip(numeric_part)] += 1

# Filter and sort the names by number of variants
repeated_names = [(key, value) for key, value in same_file_name.items() if value > 5]
repeated_names = sorted(repeated_names, key=lambda x: x[1], reverse=True)

# Print and visualize the data
print(f"Number of file names with more than 5 variants: {len(repeated_names)}")
top_repeated_names = repeated_names[:100]
print(top_repeated_names)

# Visualization
names, counts = zip(*top_repeated_names)
# Truncate long file names for visualization


def truncate_name(name, max_length=30):
    return (name[:max_length] + '...') if len(name) > max_length else name


truncated_names = [truncate_name(name) for name in names]
# Visualization with truncated names
plt.figure(figsize=(10, 8))
plt.barh(truncated_names[:20], counts[:20])
plt.xlabel('Number of Variants')
plt.ylabel('File Names')
plt.title('Top 20 File Names by Variant Count')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


from scipy.spatial.distance import cosine
import numpy as np

# Example structure: clusters = {'base_name1': ['full_name1', 'full_name2'], 'base_name2': ['full_name3', 'full_name4']}
clusters = {key: [key + str(i) for i in range(value)] for key, value in same_file_name.items() if value > 1}


# Encode text prompts and calculate distances
def encode_prompts(prompts):
    embeddings = clamp_model.get_text_embeddings(prompts)
    return embeddings


def calculate_distances_within_cluster(cluster):
    embeddings = encode_prompts(cluster)
    distances = [cosine(embeddings[i], embeddings[j]) for i in range(len(cluster)) for j in range(i+1, len(cluster))]
    return distances


def calculate_distances_between_clusters(cluster_a, cluster_b):
    embeddings_a = encode_prompts(cluster_a)
    embeddings_b = encode_prompts(cluster_b)
    distances = [cosine(embeddings_a[i], embeddings_b[j]) for i in range(len(cluster_a)) for j in range(len(cluster_b))]
    return distances

# Randomly sample clusters for intra-cluster calculations
num_samples_intra_cluster = 5  # Adjust based on your preference
intra_cluster_sample = random.sample(list(clusters.values()), min(num_samples_intra_cluster, len(clusters)))

# Randomly sample cluster pairs for inter-cluster calculations
num_samples_inter_cluster = 10  # Adjust based on your preference
all_keys = list(clusters.keys())
inter_cluster_sample = random.sample([(key1, key2) for i, key1 in enumerate(all_keys) for key2 in all_keys[i+1:]], min(num_samples_inter_cluster, len(all_keys)*(len(all_keys)-1)//2))

# Encode prompts and calculate distances (functions as previously defined)
# Ensure encode_prompts can handle your data format and returns embeddings correctly

def sample_and_calculate_distances(clusters_sample, inter=False):
    distances = []
    if inter:
        for (key1, key2) in tqdm(clusters_sample, desc="Calculating inter-cluster distances"):
            cluster_a, cluster_b = clusters[key1], clusters[key2]
            distances += calculate_distances_between_clusters(cluster_a, cluster_b)
    else:
        for cluster in tqdm(clusters_sample, desc="Calculating intra-cluster distances"):
            distances += calculate_distances_within_cluster(cluster)
    return np.mean(distances), np.min(distances), np.max(distances)

mean_intra_cluster_distance, min_intra_cluster_distance, max_intra_cluster_distance = sample_and_calculate_distances(intra_cluster_sample)
mean_inter_cluster_distance, min_inter_cluster_distance, max_inter_cluster_distance = sample_and_calculate_distances(inter_cluster_sample, inter=True)

print(f"Mean Intra-Cluster Distance (sampled): {mean_intra_cluster_distance}")
print(f"Min Intra-Cluster Distance (sampled): {min_intra_cluster_distance}")
print(f"Max Intra-Cluster Distance (sampled): {max_intra_cluster_distance}")
print(f"Mean Inter-Cluster Distance (sampled): {mean_inter_cluster_distance}")
print(f"Min Inter-Cluster Distance (sampled): {min_inter_cluster_distance}")
print(f"Max Inter-Cluster Distance (sampled): {max_inter_cluster_distance}")