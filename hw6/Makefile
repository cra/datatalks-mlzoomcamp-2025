.PHONY: help q1 q2 q3 q4 q5 q6 py-interactive get-data

PY=uv run python
CSV=car_fuel_efficiency.csv
SRC=https://raw.githubusercontent.com/alexeygrigorev/datasets/master/$(CSV)
TRAIN_CSV=$(CSV:.csv=_train.csv)
TEST_CSV=$(CSV:.csv=_test.csv)
VAL_CSV=$(CSV:.csv=_val.csv)
SEED=1
TARGET=fuel_efficiency_mpg

COMMON_ARGS=--target $(TARGET) --seed $(SEED)
TRAIN_ARG=--train $(TRAIN_CSV)
VAL_ARG=--val $(VAL_CSV)

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

get-data: $(CSV) ## download csv dataset if not present

$(CSV):
	wget $(SRC)

py-interactive: $(CSV)  ## run python repl with csv as df
	uv run ipython -i -c 'import pandas as pd; df = pd.read_csv("$(CSV)"); print(df.columns)'

$(TRAIN_CSV) $(VAL_CSV) $(TEST_CSV): $(CSV)
	${PY} split_data.py $(CSV) $(COMMON_ARGS)

q1: $(TRAIN_CSV) ## feature for splitting the data
	${PY} q1.py $(TRAIN_ARGS) $(COMMON_ARGS)

q2: $(TRAIN_CSV) $(VAL_CSV) ## rmse of regressor
	${PY} q2.py $(TRAIN_ARGS) $(COMMON_ARGS) $(VAL_ARGS)

q3: $(TRAIN_CSV) $(VAL_CSV) ## number of estimators
	${PY} q3.py $(TRAIN_ARGS) $(COMMON_ARGS) $(VAL_ARGS)

q4: $(TRAIN_CSV) $(VAL_CSV) ## best max_depth
	${PY} q4.py $(TRAIN_ARGS) $(COMMON_ARGS) $(VAL_ARGS)

q5: $(TRAIN_CSV) ## feature importance
	${PY} q5.py $(TRAIN_ARGS) $(COMMON_ARGS)

q6: $(TRAIN_CSV) $(VAL_CSV) ## xgboost
	${PY} q6.py $(TRAIN_ARGS) $(COMMON_ARGS) $(VAL_ARGS)
