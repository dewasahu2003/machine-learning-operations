install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black .

run:
	python run_pipelines.py