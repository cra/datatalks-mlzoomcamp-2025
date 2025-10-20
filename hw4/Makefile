.PHONY: help q1 q2 q3 q4 q5 q6 py-interactive get-data

PY=uv run python
CSV=course_lead_scoring.csv
SRC=https://raw.githubusercontent.com/alexeygrigorev/datasets/master/$(CSV)
TRAIN_CSV=$(CSV:.csv=_train.csv)
TEST_CSV=$(CSV:.csv=_test.csv)
VAL_CSV=$(CSV:.csv=_val.csv)
SEED=1
TARGET=converted

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

get-data: $(CSV) ## download csv dataset if not present

$(CSV):
	wget $(SRC)

py-interactive: $(CSV)  ## run python repl with csv as df
	uv run ipython -i -c 'import pandas as pd; df = pd.read_csv("$(CSV)"); print(df.columns)'

$(TRAIN_CSV) $(VAL_CSV) $(TEST_CSV): $(CSV)
	${PY} split_data.py $(CSV) --seed $(SEED) --target $(TARGET)

q1: $(TRAIN_CSV) ## ROC AUC feature importance
	${PY} q1.py --train $(TRAIN_CSV)

q2: $(TRAIN_CSV) $(VAL_CSV) ## AUC score of basic logistic regression
	${PY} q2q3q4.py --q2 --train $(TRAIN_CSV) --val $(VAL_CSV)

q3: $(TRAIN_CSV) $(VAL_CSV) ## precision and recall
	${PY} q2q3q4.py --q3 --train $(TRAIN_CSV) --val $(VAL_CSV)

q4: $(TRAIN_CSV) $(VAL_CSV) ## F1 score
	${PY} q2q3q4.py --q4 --train $(TRAIN_CSV) --val $(VAL_CSV)
