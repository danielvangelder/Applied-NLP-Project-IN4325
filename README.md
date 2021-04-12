# Applied-NLP-Project-IN4325
##### authors: Daniël van Gelder (d.vangelder-1@student.tudelft.nl) and Thomas Bos (t.c.bos@student.tudelft.nl)

## Getting Started

### Dataset and Baseline Model
The dataset for the FNC-1 challenge can be retrieved from the [dataset repository](https://github.com/FakeNewsChallenge/fnc-1). Place all dataset files in the `/data/fnc-1/` directory. Download the baseline from the [baseline repository](https://github.com/FakeNewsChallenge/fnc-1-baseline) and run the baseline model on the dataset. The code needs to be adapted so that it can output the results as a stance file. Add the following lines of code to the `fnc_kfold.py` file at line XX:

```python
pass
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

### Contact:
If information is missing from this repository, please reach out to either of us so that we can clarify.