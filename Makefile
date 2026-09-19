.PHONY: install data index demo eval report app api test docker

install:
	pip install -r requirements.txt

data:            ## regenerate simulated inventory, documents and contracts
	python scripts/generate_data.py

index:           ## chunk + embed the document corpus
	python scripts/build_index.py

demo:            ## run every module once (uses mock LLM without a key)
	python scripts/run_demo.py all

eval:            ## contract extraction accuracy: LLM vs rule-based baseline
	python scripts/run_demo.py evaluate --extractor both

report:          ## train models, forecast absorption, write bilingual report
	python scripts/run_demo.py report

app:             ## Streamlit UI
	streamlit run app/streamlit_app.py

api:             ## FastAPI service
	uvicorn api.main:app --reload --port 8000

test:
	AQAR_LLM_MODE=mock pytest

docker:
	docker build -t aqar-intel . && docker run --rm -p 8501:8501 --env-file .env aqar-intel
