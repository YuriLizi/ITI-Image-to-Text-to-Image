import clip
import torch
import pandas as pd

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load('ViT-B/32', device)

# Define the input text
input_text = "a lot of framed pictures or artwork on the wall and a black chair"  # Update with your input text

# Load the first CSV file
csv_path1 = "/home/linuxu/yuri/data/SceneNet_RGB-D/smaller_combined_train_0_data/2k_short_description_mistral-7b.csv"  # Update with your first CSV file path
df1 = pd.read_csv(csv_path1)
image_names1 = df1['Image Name'].tolist()
text_descriptions1 = df1['Image Description'].tolist()

# Load the second CSV file
csv_path2 = "/home/linuxu/yuri/data/SceneNet_RGB-D/smaller_combined_train_0_data/2k_complete_description_vicuna_7b.csv"  # Update with your second CSV file path
df2 = pd.read_csv(csv_path2)
image_names2 = df2['Image Name'].tolist()
text_descriptions2 = df2['Image Description'].tolist()

# Tokenize and encode the input text and the text descriptions for both CSV files
text_inputs1 = torch.cat([clip.tokenize(desc, truncate=True) for desc in text_descriptions1]).to(device)
text_inputs2 = torch.cat([clip.tokenize(desc, truncate=True) for desc in text_descriptions2]).to(device)
input_text_encoded = clip.tokenize(input_text, truncate=True).to(device)

# Calculate features
with torch.no_grad():
    input_text_features = model.encode_text(input_text_encoded)
    text_features1 = model.encode_text(text_inputs1)
    text_features2 = model.encode_text(text_inputs2)

# Normalize features
input_text_features /= input_text_features.norm(dim=-1, keepdim=True)
text_features1 /= text_features1.norm(dim=-1, keepdim=True)
text_features2 /= text_features2.norm(dim=-1, keepdim=True)

# Calculate similarity
similarity1 = (100.0 * input_text_features @ text_features1.T).softmax(dim=-1)
similarity2 = (100.0 * input_text_features @ text_features2.T).softmax(dim=-1)

# Get top 5 results for both CSV files
values1, indices1 = similarity1[0].topk(5)
values2, indices2 = similarity2[0].topk(5)

# Print results for the first CSV file
print("\nTop predictions from the first CSV file:\n")
for value, index in zip(values1, indices1):
    print(f"Image: {image_names1[index.item()]:>16s} | Description: {text_descriptions1[index.item()]:>16s} | Similarity: {100 * value.item():.2f}%")

# Print results for the second CSV file
print("\nTop predictions from the second CSV file:\n")
for value, index in zip(values2, indices2):
    print(f"Image: {image_names2[index.item()]:>16s} | Description: {text_descriptions2[index.item()]:>16s} | Similarity: {100 * value.item():.2f}%")

# Find common top results
top_images1 = set(image_names1[i.item()] for i in indices1)
top_images2 = set(image_names2[i.item()] for i in indices2)
common_top_images = top_images1.intersection(top_images2)

# Print common top results
print("\nImages that were top results in both CSV files:\n")
for image in common_top_images:
    print(f"Image: {image}")
