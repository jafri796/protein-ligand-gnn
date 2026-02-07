# Multi-stage Dockerfile for protein-ligand GNN training

# Stage 1: Build stage with dependencies
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies for RDKit, BioPython
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy project code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Default command
CMD ["python", "experiments/train_painn.py", "--config", "config/painn_config.yaml"]
