import time
from collections import defaultdict

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from wordcloud import WordCloud

from DDPM.main_latent_space import load_or_process_dataset, load_config


def get_keywords_map(config):
    train_dataset = load_or_process_dataset(dataset_dir=config['dataset_dir'])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    map_of_keywords = defaultdict(int)
    for _, text_data in tqdm(train_loader):
        curr_map = set()
        for keyw in text_data[0].split(" "):
            keyw = keyw.lower()
            if keyw not in curr_map:
                curr_map.add(keyw)
                map_of_keywords[keyw] += 1

    sorted_keywords = sorted(map_of_keywords.items(), key=lambda x: x[1], reverse=True)
    total_occurrences = sum(freq for _, freq in sorted_keywords)
    cumulative = 0
    threshold = total_occurrences * 0.95
    top_keywords = []

    for keyword, freq in sorted_keywords:
        cumulative += freq
        top_keywords.append(keyword)
        if cumulative >= threshold:
            break
    return top_keywords


config_path = 'DDPM/config.yaml'
config = load_config(config_path)

top_kw = get_keywords_map(config)
# Join the keywords into a single string with spaces, simulating their frequency based on their occurrence in the list.

text = ' '.join(top_kw)



# Create the word cloud

wordcloud = WordCloud(width = 800, height = 800,

                background_color ='white',

                stopwords = set(),

                min_font_size = 10).generate(text)



# Plot the WordCloud image

plt.figure(figsize = (8, 8), facecolor = None)

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout(pad = 0)

plt.savefig("./Dataset wordcloud 95 perc.jpg")

plt.show()