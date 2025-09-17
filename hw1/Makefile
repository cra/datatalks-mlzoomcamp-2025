.PHONY: help q1 q2 q3 q4 q5 q6 q7 py-interactive get-data

PY=uv run python -c
CSV=car_fuel_efficiency.csv
SRC=https://raw.githubusercontent.com/alexeygrigorev/datasets/master/$(CSV)

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

q1: ## pandas version
	${PY} 'import pandas as pd; print(pd.__version__)'

get-data: $(CSV) ## download csv dataset if not present

$(CSV):
	wget $(SRC)

py-interactive: $(CSV)  ## run python repl with csv as df
	uv run ipython -i -c 'import pandas as pd; df = pd.read_csv("$(CSV)"); print(df.columns)'

q2: $(CSV) ## records count
	@echo "Can use wc, but don't forget the header:"
	wc -l $(CSV)
	@echo "Pandas way:"
	${PY} 'import pandas as pd; df = pd.read_csv("$(CSV)"); print(len(df))'

q3: $(CSV) ## fuel types
	${PY} 'import pandas as pd; df = pd.read_csv("$(CSV)"); print(df["fuel_type"].value_counts().shape)'

q4: $(CSV) ## missing values
	${PY} 'import pandas as pd; df = pd.read_csv("$(CSV)"); print(sum(df.isnull().any()))'

q5: $(CSV) ## maximum fuel efficiency of asian cars
	${PY} 'import pandas as pd; df = pd.read_csv("$(CSV)"); print(df[df["origin"] == "Asia"]["fuel_efficiency_mpg"].max())'


q6: $(CSV) ## median value of horsepower
	@echo "== Median before fillna"
	${PY} 'import pandas as pd; df = pd.read_csv("$(CSV)"); print(df["horsepower"].median())'
	@echo "== Most frequent before fillna"
	${PY} 'import pandas as pd; df = pd.read_csv("$(CSV)"); print(df["horsepower"].mode())'
	@echo "== Most frequent after fillna with most freq"
	${PY} 'import pandas as pd; df = pd.read_csv("$(CSV)"); print(df["horsepower"].fillna(df["horsepower"].mode()).median())'

q7: $(CSV) ## sum of weights
	uv run python q7.py $(CSV)
