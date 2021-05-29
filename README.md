# JobChangePrediction
## Requirement 
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
## Run pretrained models
Run test_evaluate.ipynb
## Preprocess data, resplit data and retrain model
Run notebooks in this order:
- explore.ipynb (optional)
- stratSplit.ipynb (split data into train, cv, test set)
- pipeline.ipynb (preprocess data, this notebook also call custom_transformer.py to get all custome transformer used to preprocess)
- LRmodel.ipynb (train Logistic Regression model and evaluate)
- SVMModel.ipynb (train SVM model and evaluate)
