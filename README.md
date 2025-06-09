# Pipeline for News Classification service training, deployment, and inference

This repository contains source files and artifacts for the MLOps project using 
- FastAPI and Docker for inference
- MLFlow for metrics tracking
- Databricks as a cloud platform for pipeline deployment

**Requirements:**
- Python 3.10
- Docker
- Docker Compose

1. **Clone the repository**:
   Ensure that you have cloned the repository to your local machine.

2. **Navigate to the project directory**:
   Change directory into the project root, where the `docker-compose.yml` and Dockerfiles are located.

   ```bash
   cd path/to/NewsClassifier
   ```

3. **Start Airflow **:
   Use Docker Compose to start the Airflow and MinIO.

   ```bash
   docker-compose -f airflow/docker-compose.yaml up -d
   ```

4. **Build the model API Docker image**:
   This will create a Docker image for the News Classification API from the Dockerfile located in the `model` directory.

   ```bash
   docker build -t news-api model
   ```

5. **Run the model API**:
   Start the model API container. This will serve the model on port 8000 of your local machine.

   ```bash
   docker run -p 8000:80 news-api 
   ```

## Usage

The News classification API is accessible at `http://localhost:8000/predict`. You can send data to this endpoint to get predictions.

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"article": "The stock market is experiencing significant fluctuations due to global economic changes."}'
```

## Stopping the Project

To stop all running containers related to this project:

```bash
docker-compose -f airflow/docker-compose.yml down
docker stop serving_model
```
