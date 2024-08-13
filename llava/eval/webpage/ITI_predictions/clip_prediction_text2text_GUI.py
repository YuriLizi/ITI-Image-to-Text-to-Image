import os
import tempfile
import clip
import torch
import pandas as pd
from PIL import Image as PILImage
from io import BytesIO
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Define paths (predefined)
CSV_PATH1 = '/home/linuxu/yuri/data/SceneNet_RGB-D/smaller_combined_train_0_data/2k_short_description_mistral-7b.csv'
CSV_PATH2 = ''
IMAGES_FOLDER = '/home/linuxu/yuri/data/SceneNet_RGB-D/smaller_combined_train_0_data/2k_images'


class CLIPSimilarityApp(App):
    def build(self):
        self.title = "CLIP Similarity Search"
        self.current_index = 0  # Track the current result index

        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Input text
        self.input_text = TextInput(hint_text='Input Text', size_hint_y=None, height=30)
        layout.add_widget(self.input_text)

        # Search button
        self.search_button = Button(text='Find Most Similar', size_hint_y=None, height=40)
        self.search_button.bind(on_press=self.find_similarity)
        layout.add_widget(self.search_button)

        # Navigation buttons
        nav_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint_y=None, height=50)
        self.prev_button = Button(text='Previous', size_hint_x=None, width=100, height=40)
        self.prev_button.bind(on_press=self.show_previous_result)
        self.prev_button.opacity = 0  # Initially hidden
        self.next_button = Button(text='Next', size_hint_x=None, width=100, height=40)
        self.next_button.bind(on_press=self.show_next_result)
        self.next_button.opacity = 0  # Initially hidden
        nav_layout.add_widget(self.prev_button)
        nav_layout.add_widget(self.next_button)
        layout.add_widget(nav_layout)

        # Result label
        self.result_label = Label(size_hint_y=None, height=40)
        layout.add_widget(self.result_label)

        # Image display
        self.image = KivyImage(size_hint=(1, 1))
        layout.add_widget(self.image)

        return layout

    def find_similarity(self, instance):
        input_text = self.input_text.text
        if not input_text:
            self.result_label.text = "Please enter input text."
            return

        self.results = self.calculate_similarity(input_text, CSV_PATH1, CSV_PATH2, IMAGES_FOLDER)
        if self.results:
            self.current_index = 0
            self.show_current_result()
            # Show navigation buttons
            self.prev_button.opacity = 1
            self.next_button.opacity = 1
        else:
            self.result_label.text = "No results found."
            self.image.source = ''
            self.prev_button.opacity = 0
            self.next_button.opacity = 0

    def calculate_similarity(self, input_text, csv_path1, csv_path2, images_folder):
        results = []

        def clear_gpu_memory():
            torch.cuda.empty_cache()

        # Load the first CSV
        df1 = pd.read_csv(csv_path1)
        image_names1 = df1['Image Name'].tolist()
        text_descriptions1 = df1['Image Description'].tolist()

        # Tokenize and encode the descriptions from the first CSV
        text_inputs1 = torch.cat([clip.tokenize(desc, truncate=True) for desc in text_descriptions1]).to(device)
        input_text_encoded = clip.tokenize(input_text, truncate=True).to(device)

        with torch.no_grad():
            input_text_features = model.encode_text(input_text_encoded)
            text_features1 = model.encode_text(text_inputs1)

        input_text_features /= input_text_features.norm(dim=-1, keepdim=True)
        text_features1 /= text_features1.norm(dim=-1, keepdim=True)

        similarity1 = (100.0 * input_text_features @ text_features1.T).softmax(dim=-1)

        clear_gpu_memory()

        if csv_path2:
            # Load the second CSV if provided
            df2 = pd.read_csv(csv_path2)
            image_names2 = df2['Image Name'].tolist()
            text_descriptions2 = df2['Image Description'].tolist()

            # Tokenize and encode the descriptions from the second CSV
            text_inputs2 = torch.cat([clip.tokenize(desc, truncate=True) for desc in text_descriptions2]).to(device)

            with torch.no_grad():
                text_features2 = model.encode_text(text_inputs2)

            text_features2 /= text_features2.norm(dim=-1, keepdim=True)

            similarity2 = (100.0 * input_text_features @ text_features2.T).softmax(dim=-1)

            clear_gpu_memory()
        else:
            similarity2 = None

        # Calculate top-5 similarities
        values1, indices1 = similarity1[0].topk(5)

        for i in range(5):
            top_image1 = image_names1[indices1[i].item()]
            top_description1 = text_descriptions1[indices1[i].item()]
            top_image_path1 = os.path.join(images_folder, top_image1)

            if csv_path2 and similarity2 is not None:
                values2, indices2 = similarity2[0].topk(5)
                top_image2 = image_names2[indices2[i].item()]
                top_description2 = text_descriptions2[indices2[i].item()]
                top_image_path2 = os.path.join(images_folder, top_image2)

                if os.path.isfile(top_image_path1) and os.path.isfile(top_image_path2):
                    if values1[i].item() > values2[i].item():
                        top_image_path = top_image_path1
                        description = top_description1
                        score = values1[i].item()
                    else:
                        top_image_path = top_image_path2
                        description = top_description2
                        score = values2[i].item()
                else:
                    top_image_path = top_image_path1
                    description = top_description1
                    score = values1[i].item()
            else:
                top_image_path = top_image_path1
                description = top_description1
                score = values1[i].item()

            if os.path.isfile(top_image_path):
                results.append((top_image_path, description, score))
            else:
                results.append((None, None, None))

        return results

    def show_current_result(self):
        if not self.results or self.current_index < 0 or self.current_index >= len(self.results):
            return

        image_path, description, score = self.results[self.current_index]
        if image_path and description:
            self.result_label.text = f"Most Similar Description: {description}\nSimilarity Score: {score:.2f}%"
            self.display_image(image_path)
        else:
            self.result_label.text = "Error: Image file not found or similarity calculation failed."
            self.image.source = ''

    def show_previous_result(self, instance):
        if self.results and self.current_index > 0:
            self.current_index -= 1
            self.show_current_result()

    def show_next_result(self, instance):
        if self.results and self.current_index < len(self.results) - 1:
            self.current_index += 1
            self.show_current_result()

    def display_image(self, image_path):
        # Open the image and resize it
        image = PILImage.open(image_path)
        image.thumbnail((250, 250))

        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file_path = temp_file.name
            image.save(temp_file_path, format="PNG")

        # Set the image source to the temporary file path and reload
        self.image.source = temp_file_path
        self.image.reload()


if __name__ == "__main__":
    CLIPSimilarityApp().run()
