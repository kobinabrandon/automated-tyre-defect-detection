download: 
	poetry run python src/feature_pipeline/data_sourcing.py

make train-custom:
	poetry run python src/training_pipeline/training.py --custom

make train-hf:
	poetry run python src/training_pipeline/training.py 

