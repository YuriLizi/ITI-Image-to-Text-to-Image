import os
import clip
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Function to process an image and return its features
def process_image(image_path):
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features.cpu().numpy().flatten()

# Function to process text descriptions and return their features
def process_texts(texts):
    text_inputs = torch.cat([clip.tokenize(f"{desc}") for desc in texts]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    return text_features.cpu().numpy()

# Paths for images folder and CSV file
images_folder = "/home/linuxu/yuri/data/SceneNet_RGB-D/10_image_testset/images"  # Update with your folder path
csv_path = "/home/linuxu/yuri/data/SceneNet_RGB-D/10_image_testset/10_short_description_mistral-7b.csv"  # Update with your CSV file path
output_csv_path = "/home/linuxu/yuri/data/SceneNet_RGB-D/10_image_testset/clip_features_output.csv"  # Update with desired output path

# Load the CSV file containing text descriptions
df = pd.read_csv(csv_path)

# Initialize lists to store features
image_features_list = []
text_features_list = []

# Process each image in the folder
image_files = sorted(os.listdir(images_folder))
for image_file in tqdm(image_files):
    image_path = os.path.join(images_folder, image_file)
    image_features = process_image(image_path)
    image_features_list.append(image_features)

# Process text descriptions
text_features = process_texts(df['Image Description'])
text_features_list = [feat for feat in text_features]

# Create a new DataFrame with the image and text features
output_df = pd.DataFrame({
    "Image File": image_files,
    "Image Features": image_features_list,
    "Text Features": text_features_list
})

# Save the DataFrame to a new CSV file
output_df.to_csv(output_csv_path, index=False)
