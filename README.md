# Clinical NLP Project

This project focuses on the application of Natural Language Processing (NLP) techniques in the clinical domain. It includes functionalities for data cleaning, synonym replacement, preprocessing, vectorization, model training, and inference.

## Project Structure

```
Clinical_NLP
├── data
│   ├── train.csv          # Training dataset for model training
│   └── test_f.csv        # Test dataset for model evaluation
├── notebook
│   └── orginal_notebook.ipynb  # Jupyter notebook with original code and documentation
├── src
│   ├── __init__.py       # Marks the src directory as a Python package
│   ├── data_cleaning.py  # Functions for data cleaning
│   ├── synonym_replacement.py  # Functions for replacing synonyms
│   ├── preprocessing.py   # Functions for data preprocessing
│   ├── vectorization.py   # Functions for vectorizing text data
│   ├── model_training.py   # Code for training the model
│   ├── inference.py       # Functions for making predictions
│   └── utils.py           # Utility functions used across modules
├── requirements.txt       # Lists project dependencies
└── README.md              # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Clinical_NLP
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

- The `notebook/orginal_notebook.ipynb` contains the original code and documentation for the project. It serves as a guide for understanding the data cleaning, synonym replacement, preprocessing, vectorization, model training, and inference processes.

- The `src` directory contains Python modules that can be imported and used in your own scripts or notebooks. Each module has specific functionalities as described in the project structure.

## Modules Description

- **data_cleaning.py**: Contains functions for cleaning the dataset, including expanding abbreviations and replacing synonyms.

- **synonym_replacement.py**: Includes functions specifically for replacing synonyms in the dataset.

- **preprocessing.py**: Contains functions for preprocessing the data, such as tokenization and lemmatization.

- **vectorization.py**: Includes functions for vectorizing the text data using models like ClinicalBERT.

- **model_training.py**: Contains the code for training the model using the processed and vectorized data.

- **inference.py**: Includes functions for making predictions on new data using the trained model.

- **utils.py**: Contains utility functions that are used across different modules in the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.