.PHONY: help q1 q2 q3 q4 q5 q6 py-interactive get-data

PY=uv run python
CSV=course_lead_scoring.csv
SRC=https://raw.githubusercontent.com/alexeygrigorev/datasets/master/$(CSV)
TRAIN_CSV=$(CSV:.csv=_train.csv)
TEST_CSV=$(CSV:.csv=_test.csv)
VAL_CSV=$(CSV:.csv=_val.csv)
SEED=42
TARGET=converted

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

get-data: $(CSV) ## download csv dataset if not present

$(CSV):
	wget $(SRC)

py-interactive: $(CSV)  ## run python repl with csv as df
	uv run ipython -i -c 'import pandas as pd; df = pd.read_csv("$(CSV)"); print(df.columns)'

q1: $(CSV) ## most freq observation industry
	${PY} -c 'import pandas as pd; df = pd.read_csv("$(CSV)"); print(df["industry"].mode())'

q2: $(CSV) ## biggest correlation
	${PY} q2.py $(CSV)

$(TRAIN_CSV) $(VAL_CSV) $(TEST_CSV): $(CSV)
	${PY} split_data.py $(CSV) --seed $(SEED) --target $(TARGET)

clean:
	rm $(TRAIN_CSV) $(VAL_CSV) $(TEST_CSV) $(CSV)

q3: $(TRAIN_CSV) ## compute mutual information
	${PY} q3.py $(TRAIN_CSV) --target $(TARGET)

