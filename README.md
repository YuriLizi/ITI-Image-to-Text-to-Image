#  ITI - Image to text to image based on LLaVA and Clip

## Introduction
ITI (Image to Text to Image) is a project designed to bridge the gap between images and text using advanced AI models. The project is divided into two main parts:

### Text Generation with LLaVA:

This part involves using the LLaVA (Language-to-Visual-Awareness) model to generate text descriptions for a dataset of images. The generated descriptions are saved in a CSV file.

### Similarity Search Application with CLIP:

This part utilizes the CLIP (Contrastive Language–Image Pre-training) model to perform similarity searches based on the text descriptions generated in the first part. Users can input a text query to find and display the most similar images from the dataset.

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use) for the dataset and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama community license](https://ai.meta.com/llama/license/) for LLaMA-2 and Vicuna-v1.5). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.


## Contents
- [Install](#install)
- [Dataset](#Dataset-SceneNetRGB-D)
- [Usage](#Usage)
## Install

NOTE: To use both LLaVA and CLIP with an older CUDA version (11.4), you need to create two separate virtual environments with specific versions of PyTorch:

* #### LLaVA requires PyTorch version 2.0.1.
* #### CLIP requires the latest version of PyTorch (2.4.0).



1. Clone this repository and navigate to ITI folder
```bash
git clone https://github.com/YuriLizi/ITI-Image-to-Text-to-Image.git
cd ITI-Image-to-Text-to-Image
```

### Creating Virtual Environment in PyCharm

Launch PyCharm and open your project or create a new one.
Create the First Virtual Environment (for LLaVA):

Navigate to Settings:
Go to File > Settings (or PyCharm > Preferences on macOS).
Add a New Interpreter:
In the left sidebar, select Project: <Your Project Name> > Python Interpreter.
Click on the gear icon next to the current interpreter and select Add.
Choose Virtualenv Environment.
Select New environment and specify the location for the virtual environment. Name it something like llava_env.
In the Base interpreter dropdown, select your system Python interpreter.
Click OK to create the environment.
Install LLaVA Dependencies.

Do the same for CLIP Virtual Environment


2. Install Packages - for building new Dataset - LLaVA Virtual Environment


```Shell
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install Packages - for running Similarity Search Application - CLIP Virtual Environment

   First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```Shell
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install pandas
pip install Kivy

```




## Dataset-SceneNetRGB-D
We utilized the SceneNetRGB-D dataset to generate a subset consisting of 2,000 randomly selected images. For each image, we created two types of descriptions:

#### Short Descriptions:
Generated using the LLaVA-v1.6-mistral-7b model.
#### Full Descriptions:
Generated using the LLaVA-v1.6-vicuna-7b model.

These descriptions were saved in separate CSV files, with each file containing the corresponding descriptions for the images.

## Usage
### Dataset and Scripts
In the directory ITI-Image-to-Text-to-Image/llava/eval/ITI_predictions, you'll find the following resources:

#### Dataset:
The subset of 2,000 randomly selected images from SceneNetRGB-D is located in the folder 2k_SceneNetRGB-D_dataset. This folder contains images along with their descriptions csv.

#### Dataset Builder Scripts:

* SceneNetRGB-D_random_images_picker.py:
Script for picking randomly X images from each folder from the SceneNetRGB-D dataset.


* Image_description_csv.py:
Script for generating CSV files with image descriptions using LLaVA.
SceneNetRGB-D_random_images_picker.py:
Script for selecting a random subset of images from the SceneNetRGB-D dataset and saving them.

### Feature Extraction and Similarity Search Scripts:

* clip_features_extraction.py: Extracts CLIP features from images and saves them in a CSV file.(For future use, was not used for this project and is not supported by the Similarity_Search )


* Similarity_Search_TTI.py: Finds similarity between user input text and image descriptions from the dataset using CLIP.


* Similarity_Search_TTI_GUI.py: Provides a graphical user interface (GUI) for performing similarity searches.

