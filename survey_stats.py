import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Define a function to map text responses to Likert scale
def map_to_likert(question, response):
    mappings = {
        'Quality': {
            'Poor - Unfit for musical composition': 1,
            'Fair - Needs significant improvement to fit in a musical composition': 2,
            'Average - Acceptable but requires some modifications for a musical composition': 3,
            'Good - Well-suited for a musical composition with minor adjustments': 4,
            'Excellent - Perfectly suited for a musical composition as is': 5,
        },
        'Aptness': {
            'Not at all - Completely irrelevant to the text': 1,
            'Slightly - Barely relevant to the text': 2,
            'Moderately - Somewhat relevant but lacks clear connection to the text': 3,
            'Very - Mostly relevant with minor discrepancies from the text': 4,
            'Perfectly - Fully relevant and accurately reflects the text': 5,
        },
        'Novelty': {
            'Not novel at all - Very common and I have heard many similar beats': 1,
            'Slightly novel - Somewhat common with some familiar elements': 2,
            'Moderately novel - A mix of familiar and new elements': 3,
            'Very novel - Rarely encountered, mostly new elements': 4,
            'Extremely novel - Unique and I have never heard anything similar before': 5,
        }
    }
    return mappings[question].get(response, None)  # Returns None if response is not found


def save_likert_csv(survey_results):
    # Define the mapping function for the Likert scale
    def _map_to_likert(response):
        likert_mapping = {
            # Mapping for Quality
            'Poor - Unfit for musical composition': 1,
            'Fair - Needs significant improvement to fit in a musical composition': 2,
            'Average - Acceptable but requires some modifications for a musical composition': 3,
            'Good - Well-suited for a musical composition with minor adjustments': 4,
            'Excellent - Perfectly suited for a musical composition as is': 5,
            # Mapping for Aptness to text-prompt
            'Not at all - Completely irrelevant to the text': 1,
            'Slightly - Barely relevant to the text': 2,
            'Moderately - Somewhat relevant but lacks clear connection to the text': 3,
            'Very - Mostly relevant with minor discrepancies from the text': 4,
            'Perfectly - Fully relevant and accurately reflects the text': 5,
            # Mapping for Novelty
            'Not novel at all - Very common and I have heard many similar beats': 1,
            'Slightly novel - Somewhat common with some familiar elements': 2,
            'Moderately novel - A mix of familiar and new elements': 3,
            'Very novel - Rarely encountered, mostly new elements': 4,
            'Extremely novel - Unique and I have never heard anything similar before': 5
        }
        return likert_mapping.get(response, response)  # Default to the original response if not found
    # Process the survey results DataFrame
    for i in range(9, survey_results.shape[1] - 1, 4):  # Assuming every fourth column starting at 9 is responses
        survey_results.iloc[:, i+1:i+4] = survey_results.iloc[:, i+1:i+4].applymap(_map_to_likert)

    # Save the modified DataFrame to a new CSV file
    survey_results.to_csv('AIMC results/Survey Results Likert.csv', index=False)


def plot_survey_results(category_responses, ymin=2.5, ymax=4):
    # Prepare data for plotting
    questions = ['Quality', 'Aptness', 'Novelty']
    categories = ['dataset', 'multihot', 'bert', 'negative']
    data = {q: [np.mean(category_responses[cat][q]) if category_responses[cat][q] else None for cat in categories] for q
            in questions}

    # Number of questions
    n_groups = len(questions)

    # Create a bar plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.20
    opacity = 0.8

    for i, category in enumerate(categories):
        means = [data[q][i] for q in questions]

        ax.bar(index + i * bar_width, means, bar_width, alpha=opacity, label=category)

    ax.set_xlabel('Questions')
    ax.set_ylabel('Average Likert Scale Response')
    ax.set_title('Survey Results by Category and Question')
    ax.set_xticks(index + bar_width * (len(categories) - 1) / 2)
    ax.set_xticklabels(questions)
    ax.legend()

    # Set custom y-axis limits if provided
    if ymin is not None and ymax is not None:
        ax.set_ylim([ymin, ymax])
    elif ymin is not None:
        ax.set_ylim(bottom=ymin)
    elif ymax is not None:
        ax.set_ylim(top=ymax)

    plt.tight_layout()
    plt.savefig("AIMC results/survey_results.png")
    plt.show()


def top_quality_drumbeats(file_mappings, survey_results):
    def _map_to_likert(response):
        likert_mapping = {
            # Mapping for Quality
            'Poor - Unfit for musical composition': 1,
            'Fair - Needs significant improvement to fit in a musical composition': 2,
            'Average - Acceptable but requires some modifications for a musical composition': 3,
            'Good - Well-suited for a musical composition with minor adjustments': 4,
            'Excellent - Perfectly suited for a musical composition as is': 5,
            # Mapping for Aptness to text-prompt
            'Not at all - Completely irrelevant to the text': 1,
            'Slightly - Barely relevant to the text': 2,
            'Moderately - Somewhat relevant but lacks clear connection to the text': 3,
            'Very - Mostly relevant with minor discrepancies from the text': 4,
            'Perfectly - Fully relevant and accurately reflects the text': 5,
            # Mapping for Novelty
            'Not novel at all - Very common and I have heard many similar beats': 1,
            'Slightly novel - Somewhat common with some familiar elements': 2,
            'Moderately novel - A mix of familiar and new elements': 3,
            'Very novel - Rarely encountered, mostly new elements': 4,
            'Extremely novel - Unique and I have never heard anything similar before': 5
        }
        return likert_mapping.get(response, response)  # Default to the original response if not found

    # Extract relevant columns from survey_results
    quality_cols = [survey_results.columns[i + 1] for i in range(9, survey_results.shape[1] - 1, 4)]

    # Map drumbeat filenames to their responses
    drumbeat_scores = {}
    for col in quality_cols:
        drumbeat_name = col[0].split(' · ')[1] if ' · ' in col[0] else col[0]  # Clean the drumbeat name
        scores = survey_results[col].dropna().apply(_map_to_likert).astype(float)  # Assuming map_to_likert is available
        if drumbeat_name in drumbeat_scores:
            drumbeat_scores[drumbeat_name].extend(scores.tolist())
        else:
            drumbeat_scores[drumbeat_name] = scores.tolist()

    # Calculate average scores for each drumbeat
    average_scores = {drumbeat: sum(scores) / len(scores) if scores else 0 for drumbeat, scores in
                      drumbeat_scores.items()}

    # Find top 5 drumbeats with highest average quality scores
    top_drumbeats = sorted(average_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    # Print the names of the top 5 drumbeats
    print("Top 5 Drumbeats with Highest Aggregated Quality:")
    for i, (name, score) in enumerate(top_drumbeats, start=1):
        print(f"{i}. {name} - Average Quality Score: {score:.2f}")


# Example usage
# top_quality_drumbeats(file_mappings, survey_results)


def plot_likert_distribution(category_responses):
    # Categories and subcategories
    categories = ['dataset', 'multihot', 'bert', 'negative']
    subcategories = ['Quality', 'Aptness', 'Novelty']

    # Prepare the data
    response_counts = {cat: {sub: {i: 0 for i in range(1, 6)} for sub in subcategories} for cat in categories}

    # Count responses for each category and subcategory
    for cat in categories:
        for sub in subcategories:
            if category_responses[cat][sub]:
                for response in category_responses[cat][sub]:
                    if response in [1, 2, 3, 4, 5]:  # Check if response is a valid Likert scale value
                        response_counts[cat][sub][response] += 1

    # Plotting
    fig, axes = plt.subplots(nrows=len(subcategories), ncols=len(categories), figsize=(14, 8), sharey=True)

    for i, sub in enumerate(subcategories):
        for j, cat in enumerate(categories):
            ax = axes[i][j] if len(subcategories) > 1 else axes[j]
            counts = [response_counts[cat][sub][resp] for resp in range(1, 6)]
            ax.bar(range(1, 6), counts, color='b')
            ax.set_title(f'{cat} - {sub}')
            ax.set_xlabel('Likert Scale Response')
            ax.set_xticks(range(1, 6))
            if j == 0:
                ax.set_ylabel('Response Count')

    plt.suptitle('Distribution of Likert Scale Responses Across All Categories and Subcategories')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    plt.savefig("AIMC results/likert_distribution.png")
    plt.show()


# Step 1: Read the CSV files
file_mappings = pd.read_csv('AIMC results/file_mappings.csv')
survey_results = pd.read_csv('AIMC results/Survey Results.csv', header=[0, 1])  # Assuming two header rows
# save_likert_csv(survey_results)
top_quality_drumbeats(file_mappings, survey_results)
exit("Done")

# Step 2: Create structured dictionary for responses
category_responses = {category: {'Quality': [], 'Aptness': [], 'Novelty': []} for category in
                      ['dataset', 'multihot', 'bert', 'negative']}

# Step 3: Extract and map responses
filenames_with_prefix = [survey_results.columns[i][0] for i in range(9, survey_results.shape[1] - 1, 4)]
cleaned_filenames = [fname.split(' · ')[1] if ' · ' in fname else fname for fname in filenames_with_prefix]
adjusted_filenames = [fname + '.mid' for fname in cleaned_filenames]

for index, filename in enumerate(adjusted_filenames):
    actual_index = 9 + index * 4  # Calculating the index of the first response column
    responses = survey_results.iloc[:, actual_index + 1:actual_index + 4]  # Questions corresponding to the filename

    category_match = file_mappings[file_mappings['file_name'] == filename]
    if not category_match.empty:
        category = category_match['category'].values[0]
        subcategories = ['Quality', 'Aptness', 'Novelty']

        for sub_idx, subcategory in enumerate(subcategories):
            mapped_responses = responses.iloc[:, sub_idx].apply(lambda x: map_to_likert(subcategory, x))
            category_responses[category][subcategory].extend(mapped_responses.tolist())

plot_survey_results(category_responses)
plot_likert_distribution(category_responses)