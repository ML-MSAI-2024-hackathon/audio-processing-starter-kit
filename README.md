# Getting started with audio processing

The following repo containts a primer on audio processing.

You can choose to run it on Google Cloud or on your local computer.

## ☁️ Google Colab
- open [this notebook](https://github.com/ML-MSAI-2024-hackathon/audio-processing-starter-kit/blob/main/welcome_colab.ipynb) on Google Colab.
- follow the instructions in the notebook to complete the tutorial

## 🖥️ Your local computer
- Clone this repository:
    - Click on the blue "Code" button in the right part of this page
    - Select "GitHub Desktop" option
    - If a popup asking you to grant GitHub Desktop to access github.com comes up, click "Allow"
    - A GitHub Desktop window will open, asking you where to clone the repository
    - Select what folder to clone the repo into, remember this choice for later
- Move into the selected folder of previous step
- Create a virtual environment:
    - `python -m venv starter_kit_env` if you want to use Python virtual env package
    - `conda create -n starter_kit_env` if you want to use Conda, Anaconda or Miniconda
- Activate the newly created virtual environment:
    - `starter_kit_env\Scripts\activate` if you want to use Python virtual env package and you are on Windows
    - `source starter_kit_env/bin/activate` if you want to use Python virtual env package and you are on macOS or Linux
    - `conda activate starter_kit_env` if you want to use Conda, Anaconda or Miniconda
- Install the required Python packages:
    - `pip3 install -r requirements.txt` if you will run the Notebook on Google Colab
- Select the correct virtual environment:
    - Open the editor of your choice (Visual Studio Code, PyCharm, etc.)
    - Open the notebook
    - Select "starter_kit_env" as virtual environment
