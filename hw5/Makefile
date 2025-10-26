.PHONY: help q1 q2 q3 q4 q5 q6 py-interactive get-data

PY=uv run python
BIN=pipeline_v1.bin
SRC=https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/cohorts/2025/05-deployment/$(BIN)
MY_DOCKER_TAG=cra-mlzoomcamp-hw5:2025

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

get-model: $(BIN) ## download csv dataset if not present

$(BIN):
	wget $(SRC)
	@echo "=============================================="
	@echo "expected md5:"
	@echo "7d17d2e4dfbaf1e408e1a62e6e880d49  $(BIN) (expected)"
	@echo "got:"
	@md5sum $(BIN)

clean:
	rm $(BIN)

q1: ## tell uv version
	uv --version

q2: ## first hash for skicit-learn
	uv add scikit-learn==1.6.1
	@echo "using grep"
	cat uv.lock | grep '^name = "scikit-learn"' -A20 | grep -E -m1 -o 'hash = ".*",'
	@echo "using python tomllib:"
	${PY} -c 'import tomllib; print([p for p in tomllib.load(open("uv.lock", "rb"))["package"] if p["name"] == "scikit-learn"][0]["sdist"]["hash"])'

q3: $(BIN)  ## use model to get prediction
	${PY} q3.py --model $(BIN)

q4:
	@echo "run make q3-serve in one terminal"
	@echo "then, run make q3-score in another terminal"

q4-serve: $(BIN) ## start fastapi server for q4
	MODEL_PATH=$(BIN) uv run q4_serve

q4-score:  ## ping fastapy server to answer q4
	API_URL=http://localhost:8000/v1/predict uv run api_score

q5: ## base image size
	docker pull agrigorev/zoomcamp-model:2025
	docker images | grep agrigorev/zoomcamp-model

q6-build:  ## build docker container for q6
	docker build -t $(MY_DOCKER_TAG) .

q6-run:  ## start api server for q6
	docker run -p 8000:8000 $(MY_DOCKER_TAG)

q6-score:  ## ping fastapy server to answer q6
	API_URL=http://localhost:8000/v2/predict uv run api_score
