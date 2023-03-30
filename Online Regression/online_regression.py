
from river import datasets
from river import neighbors
from river import metrics
from river import evaluate
from pprint import pprint

dataset = datasets.TrumpApproval()
dataset

# Look at the first data sample
for x,y in dataset:
    pass

x,y = next(iter(dataset))
pprint(x)
pprint(y)

# Define the model and predict with the first sample
model = neighbors.KNNRegressor()
metric = metrics.MAE()
model.predict_one(x)

# Learn from the first data sample and predict
model = model.learn_one(x,y)
model.predict_one(x)

# Learn from all data
for x, y in dataset:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    metric.update(y, y_pred)

metric

# Evaluate the model
evaluate.progressive_val_score(dataset, model, metric)

