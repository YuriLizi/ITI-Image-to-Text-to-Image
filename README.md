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


2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```




## Dataset-SceneNetRGB-D
We utilized the SceneNetRGB-D dataset to generate a subset consisting of 2,000 randomly selected images. For each image, we created two types of descriptions:

#### Short Descriptions:
Generated using the LLaVA-v1.6-mistral-7b model.
#### Full Descriptions:
Generated using the LLaVA-v1.6-vicuna-7b model.

These descriptions were saved in separate CSV files, with each file containing the corresponding descriptions for the images.
