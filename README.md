# starter-kit

## Installation

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
    - `pip3 install -r requirements_colab.txt` if you will run the Notebook on Google Colab
    - `pip3 install -r requirements_computer.txt` if you will run the Notebook on your computer
- If you run the Notebook on your computer:
    - Open the editor of your choice (Visual Studio Code, PyCharm, etc.)
    - Open the notebook
    - Select "starter_kit_env" as virtual environment
