### m5_forecasting
==============================

#### Project Caveats
- This is my first ever competition and github repo ... so be critical. I know a lot of mistakes were made so
  constructive criticism/feedback is much appreciated.
- There are a lot of areas for improvement. The MAIN objective of this was for me to learn. 
- Thanks in advance!



#### Project Overview
- Use hierarchical sales data from Walmart to forecast daily sales for next 28 days.
- More details https://www.kaggle.com/c/m5-forecasting-accuracy/overview/description
- Hierarchical modeling using lgbm at pre-defined strata



#### Run order:
```
python ./src/group_level.py 
python ./src/item_level.py
python ./src/final_scale.py
```



#### Project Organization
------------

    ├── LICENSE
    ├── README.md                   <- The top-level README for developers using this project.
    ├── data
    │   ├── processed               <- The final, canonical data sets for modeling.
    │   └── raw                     <- The original, immutable data dump.
    │
    ├── models                      <- Trained and serialized models, model predictions, or model summaries
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
    │                                  generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                    <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                         <- Source code for use in this project.
    │   ├── __init__.py             <- Makes src a Python module
    │   │
    │   ├── data                    <- Scripts to download or generate data
    │   │   └── etl.py              <- Tranforms raw data, creates lag variables, applies SMOTE for imbalanced data
    │   │
    │   ├── models                  <- Scripts to train models and then use trained models to make
    │   │   │                          predictions
    │   │   ├── predict_model.py    <- Make predictions using defined model parameters
    │   │   ├── train_model.py      <- Hyperparameter runing using RandomizedSearchCV
    |   |   └── lstm_class.py       <- Work in progress. Attempt to incorporate/learn LSTM.
    |   |
    |   ├── paths.py                <- Generates relative file paths
    |   ├── group_level.py          <- Creates forecasts for group/strata (@ state, store, category, and department levels)
    |   ├── item_level.py           <- Creates forecasts for all items
    |   ├── final_scale.py          <- Hierarchical scaling (state --> state/store --> state/store/category --> 
    |   |                              state/store/category/dept --> item
    |   ├── compare_models.py       <- Work in progress. Wanted to learn LSTM and see if it produced better results. Will need 
                                       gpu to run this module. Even then, it'll take a significant amount of time.
--------

<p><small>Project structure based on trimmed version of <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>