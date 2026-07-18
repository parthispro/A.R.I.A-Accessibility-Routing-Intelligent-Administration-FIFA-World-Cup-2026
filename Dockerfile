FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Install dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/ app/
COPY data/ data/
COPY static/ static/

# Drop root: run as an unprivileged user (container hardening best practice).
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Command to run the application using uvicorn
CMD exec uvicorn app.main:app --host 0.0.0.0 --port $PORT
