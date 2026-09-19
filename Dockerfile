FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Simulated data ships with the repo; the index is (re)built at start so it matches the configured embedder.
EXPOSE 8501 8000
CMD ["sh", "-c", "python scripts/build_index.py && streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]
