# Applied-NLP-Project-IN4325
##### Authors: Daniël van Gelder (d.vangelder-1@student.tudelft.nl) and Thomas Bos (t.c.bos@student.tudelft.nl)

## Getting Started

### 1. Dataset and Baseline Model
The dataset for the FNC-1 challenge can be retrieved from the [dataset repository](https://github.com/FakeNewsChallenge/fnc-1). Place all dataset files in the `/data/fnc-1/` directory. Download the baseline from the [baseline repository](https://github.com/FakeNewsChallenge/fnc-1-baseline) and run the baseline model on the dataset. The code needs to be adapted so that it can output the results as a stance file. Change and add the following lines of code to the `fnc_kfold.py` file at line 97:

```python
OUT_DIR = "DIRECTORY_TO_REPOSITORY_DATASET" # Change this to the directory where you want the predictions to be stored
df = pd.read_csv("fnc-1/competition_test_stances.csv", names=['Headline', 'Body ID', 'Stance'], header=0)
df['Stance'] = predicted
df.to_csv(OUT_DIR + "/baseline_output.csv")
```

### Requirements:
- Python
- simpletransformers
- transformers
- numpy
- pandas
- jupyter notebook
- scikit-learn 
- nltk
- gensim
- tqdm

### 2. Running ALBERT
The notebook `albert_fnc1.ipynb` containing further instructions can be opened in Google Colab, which
was used to generate all our results regarding the use of ALBERT on the FNC-1 data set.

### Contact:
If information is missing from this repository, please reach out to either of us so that we can clarify.