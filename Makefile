install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt &&\
			zenml integration install mlflow -y

format:
	black .

run:
	python run_pipelines.py

getstack:
	zenml stack list

regtracker:
	zenml experiment-tracker register mlflow_tracker --flavor=mlflow

regdeploy:
	zenml model-deployer register mlflow_deployer --flavor=mlflow

regstack:
	zenml stack register mlflow_stack -a default -o default -d mlflow_deployer -e mlflow_tracker

expresult:
	mlflow ui --backend-store-uri "file:/home/codespace/.config/zenml/local_stores/5281eb7e-0313-45fd-b9a0-546af6e63cf8/mlruns"

deployPipeline:
	python run_deployment_pipeline.py --config deploy

predict:
	python run_deployment_pipeline.py --config predict
	
ui:
	streamlit run streamlit_app.py

all: install regtracker regdeploy regstack run ui