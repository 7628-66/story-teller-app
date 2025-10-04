# Minimal Dockerfile for StoryGen
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

# Install base package; extras optional
RUN pip install --no-cache-dir . && \
    pip cache purge

# Default command prints help
ENTRYPOINT ["storygen"]
CMD ["--help"]
