# Phishing URL scorer: Gradio app on port 7860, for a Hugging Face Docker Space.
#
# The build context is the Space bundle made by `python app/build_space.py`
# (app.py, requirements.txt, src/features.py, models/, reports/metrics.json).
# Build and run locally:
#   python app/build_space.py
#   docker build -t phishing-url-scorer build/space
#   docker run --rm -p 7860:7860 phishing-url-scorer
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs the container as user id 1000; run as that user everywhere.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1
WORKDIR $HOME/app

COPY --chown=user requirements.txt ./
RUN pip install --user -r requirements.txt

COPY --chown=user app.py background.js ./
COPY --chown=user src ./src
COPY --chown=user models ./models
COPY --chown=user reports ./reports

EXPOSE 7860
CMD ["python", "app.py"]
