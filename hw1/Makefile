.PHONY: help q1

PY=uv run python -c

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

q1: ## pandas version
	${PY} 'import pandas as pd; print(pd.__version__)'

get-data:  ## download laptopts price dataset
	wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv

py-interactive:
	uv run ipython -i -c 'import pandas as pd; df = pd.read_csv("laptops.csv");'

q2: ## records count
	@echo "Can use wc, but don't forget the header:"
	wc -l laptops.csv
	@echo "Pandas way:"
	${PY} 'import pandas as pd; df = pd.read_csv("laptops.csv"); print(len(df))'

q3: ## laptop brands
	${PY} 'import pandas as pd; df = pd.read_csv("laptops.csv"); print(len(df["Brand"].value_counts().index))'

q4: ## missing values
	${PY} 'import pandas as pd; df = pd.read_csv("laptops.csv"); print(sum(df.isnull().any()))'

q5: ## maximum final price
	${PY} 'import pandas as pd; df = pd.read_csv("laptops.csv"); print(df[df["Brand"] == "Dell"]["Final Price"].max())'


q6: ## median value of screen
	@echo "Median before fillna"
	${PY} 'import pandas as pd; df = pd.read_csv("laptops.csv"); print(df["Screen"].median(skipna=True))'
	@echo "Most frequent before fillna"
	${PY} 'import pandas as pd; df = pd.read_csv("laptops.csv"); print(df["Screen"].mode())'
	@echo "Most frequent after fillna with most freq"
	${PY} 'import pandas as pd; df = pd.read_csv("laptops.csv"); print(df["Screen"].fillna(df["Screen"].mode()).median())'

q7: ## sum of weights
	uv run python q7.py
