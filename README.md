# ML ZoomCamp 2025cohort

## hw1: [intro](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2025/01-intro/homework.md)

[my solution](./hw1)

```bash
cd hw1
make help
uv sync
make q1
make q2
...
```

## hw2: [regression](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2025/02-regression/homework.md)

DNF

## hw3: [classification](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2025/03-classification/homework.md)

```bash
cd hw3
make help
uv sync
make q1
make q2
...
```

Assuming the phrase "Make sure that the target value `converted` is not in your dataframe." is about modelling step, not `.csv` files

:warning: IMPORTANT course assumption! I made my split helper fill in missing values for datapoints ("NA" for categorical, "0.0" for numerical) because it's okay to do it here, but in real "production" systems it's not cool to do so: you'd probably fill in with statistical value and risk data leakage; also the whole thing becomes a bit more brittle. Here it makes sense since we have same question over and over and I would otherwise have to repeat the process multiple times

Q6 yields same values for linear solver and suggested values of C. Weird? Weird. This means that for given dataset the model has already converged to a good solution at the lowest available C value. Going LOWER (C < 0.01, like 0.001) applies stronger regularization, which hurts performance, which means that too much regularization leads to underfitting and lower accuracy, whlie C between 0.01-100 for this dataset gives good performance. Less regularization (large C) doesn't hurt here, but also doesn't help :shrug:
