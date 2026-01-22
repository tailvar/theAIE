"""
TO run
    -cd ~/PycharmProjects/theAIE/aiecode/ch02
    -docker build -t eng-ch02-api .      # this is the image i created
    -docker run --rm -p 8000:8000 eng-ch02-api
Then in browser
    -http://127.0.0.1:8000/stats?seed=42&n=5

"""

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /app

# Install only what we need to serve the API
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn

# Copy the whole chapter directory into /app
COPY . /app

# ⬅️ THIS is the important bit:
# Make /app/src importable so `from mymltool.core import ...` works
ENV PYTHONPATH=/app/src

EXPOSE 8000

# fastapi_app.py is at /app/fastapi_app.py
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]



