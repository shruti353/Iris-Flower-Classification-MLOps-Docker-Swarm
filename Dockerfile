FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Train model during image build
RUN python train.py
EXPOSE 5000
CMD ["python", "app.py"]