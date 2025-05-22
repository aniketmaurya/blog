---
title: 'Building ML APIs for fun and production'
description: 'LitServe is production-ready open-source ML model serving library from Lightning AI.'
date: '2025-05-22'
keywords: blog, writing
layout: post
toc: false
image: /assets/ai-blog.jpg
---

**Deploying Your AI Models with LitServe: A Step-by-Step Tutorial**


Have you ever trained a fantastic AI model, only to hit a roadblock when it comes to deploying it for real-world use? Moving a model from a Jupyter Notebook to a scalable, production-ready API can be a daunting task. That's where **LitServe** comes in!

LitServe, an open-source serving engine built on FastAPI, simplifies the deployment of your AI models. It's optimized for AI workloads, offering features like batching, streaming, and GPU autoscaling to ensure your model performs efficiently under demand.

In this tutorial, we'll walk you through the process of taking your AI model and deploying it using LitServe, making it accessible via a high-performance API.

---

### **Prerequisites**

Before we begin, make sure you have:

* Python 3.8+ installed
* Familiarity with Python and basic AI concepts

---

### **Step 1: Install LitServe**

First things first, let's get LitServe installed. Open your terminal or command prompt and run:

```bash
pip install litserve
```

This will install LitServe and its necessary dependencies, including FastAPI.

---

### **Step 2: Define Your Model API with `LitAPI`**

The heart of your LitServe deployment is the `LitAPI` class. This class acts as the bridge between your incoming API requests and your AI model. You'll define how requests are handled, how your model processes data, and how responses are formatted.

Let's break down the key methods you'll implement within your `LitAPI` class:

* **`setup(self, device)`**: This method is called *once* when your server starts. It's the perfect place to load your pre-trained model, set up any required resources (like tokenizers), and move your model to the specified device (CPU or GPU). LitServe automatically provides the `device` argument based on your server configuration.
* **`decode_request(self, request)`**: For *each incoming request*, this method transforms the raw request payload (e.g., JSON, image bytes) into a format your model expects as input.
* **`predict(self, x)`**: This is where your AI model does its magic! It takes the decoded input `x` (or a batch of inputs) and runs your model to generate predictions.
* **`encode_response(self, output)`**: Finally, this method takes the output from your `predict` method and formats it into the desired response to send back to the client (e.g., a JSON dictionary, a list).

Let's create a file named `server.py` and put the following code inside it. For this tutorial, we'll use a simple PyTorch model, but you can adapt it to any framework (TensorFlow, JAX, scikit-learn, etc.).

```python
import litserve as ls
import torch
import torch.nn as nn
import os

# 1. Define your AI Model
# Replace this with your actual, pre-trained model.
# For demonstration, we'll create a simple linear model.
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1) # Expects 10 input features, outputs 1 value
        
    def forward(self, x):
        return self.linear(x)

# 2. Define your LitAPI class
class MyLitAPI(ls.LitAPI):
    def setup(self, device):
        # Load your model here.
        # If you saved your model (e.g., with torch.save('model.pth')), load it like this:
        # model_path = "path/to/your/model.pth" 
        # self.model = SimpleModel().to(device)
        # self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        # For this example, we'll just instantiate a dummy model.
        self.model = SimpleModel().to(device)
        self.model.eval() # Set model to evaluation mode
        print(f"✅ Model loaded successfully on device: {device}")

    def decode_request(self, request):
        # We expect a JSON request like: {"input": [1.0, 2.0, ..., 10.0]}
        data = request.json()
        input_data = data.get("input")
        if not isinstance(input_data, list) or len(input_data) != 10:
            raise ValueError("Invalid input format. Expected a list of 10 numbers.")
        
        # Convert to a PyTorch tensor
        return torch.tensor(input_data, dtype=torch.float32).unsqueeze(0) # unsqueeze for batch dim

    def predict(self, x):
        with torch.no_grad(): # Disable gradient calculation for inference
            output = self.model(x)
        return output.squeeze(0).tolist() # Convert tensor output to a list

    def encode_response(self, output):
        # Format the model's output as a JSON response
        return {"prediction": output}

# 3. Run the LitServe server
if __name__ == "__main__":
    api = MyLitAPI()

    # Configure the LitServer
    # accelerator="auto" will automatically use GPU if available, otherwise CPU.
    # You can also specify "cpu", "cuda", or "gpu".
    # max_batch_size and batch_timeout are crucial for performance!
    server = ls.LitServer(
        api,
        accelerator="auto", 
        max_batch_size=16, # Process up to 16 requests at once if they arrive quickly
        batch_timeout=0.1, # Wait up to 0.1 seconds for a full batch
        workers_per_device=1 # Number of API workers per CPU/GPU
    )

    # Start the server on port 8000
    server.run(port=8000)
    print(f"\n🚀 LitServe is running on http://127.0.0.1:8000")

```

**A quick note on Batching:** Notice the `max_batch_size` and `batch_timeout` arguments in `ls.LitServer`. These are powerful features that allow LitServe to group multiple incoming requests and process them as a single batch through your model. This significantly boosts GPU utilization and overall throughput, especially for models that benefit from parallel processing.

---

### **Step 3: Run Your LitServe Application**

Now that you've defined your `LitAPI`, it's time to bring your server online! You have a few deployment options:

#### **Option A: Local Self-Hosting (for Development)**

For quick testing and development, you can run your server directly from your Python script:

```bash
python server.py
```

You should see output indicating that LitServe is starting up, typically on `http://127.0.0.1:8000`.

#### **Option B: Deploy with Lightning AI Cloud (Recommended for Production)**

LitServe is developed by Lightning AI, and they provide a fantastic managed platform for seamless deployment, autoscaling, and production-grade features. This is often the most straightforward and robust way to deploy your models at scale.

1.  **Install the Lightning CLI:**
    ```bash
    pip install lightning
    ```
2.  **Log in to Lightning AI:**
    ```bash
    lightning login
    ```
    This command will open your web browser to authenticate your account.
3.  **Deploy your server:**
    Navigate to the directory containing your `server.py` file and run:
    ```bash
    lightning deploy server.py --cloud
    ```
    The Lightning CLI will automatically package your application (including any `requirements.txt` file in your directory), build a Docker image, upload it, and deploy it to a scalable endpoint. You'll receive a public URL for your deployed API!

#### **Option C: Self-Hosting with Docker (Advanced)**

For complete control over your deployment environment on your own infrastructure (local, cloud VM, etc.), you can containerize your LitServe application using Docker.

1.  **Create a `requirements.txt` file:**
    ```
    litserve
    torch
    # Add any other dependencies your model needs (e.g., transformers, numpy, etc.)
    ```

2.  **Create a `Dockerfile` in the same directory as `server.py` and `requirements.txt`:**

    ```dockerfile
    # Use a suitable base image for your model (e.g., with PyTorch and CUDA)
    FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

    WORKDIR /app

    # Copy your application files
    COPY server.py .
    COPY requirements.txt .
    # If you have a saved model file, copy it too:
    # COPY path/to/your/model.pth .

    # Install dependencies
    RUN pip install -r requirements.txt

    # Expose the port LitServe will run on
    EXPOSE 8000

    # Command to run your LitServe server
    CMD ["python", "server.py"]
    ```

3.  **Build your Docker image:**
    ```bash
    docker build -t my-litserve-model .
    ```

4.  **Run your Docker container:**
    ```bash
    docker run -p 8000:8000 my-litserve-model
    ```
    This command maps port 8000 on your host machine to port 8000 inside the Docker container, making your API accessible.

---

### **Step 4: Test Your Deployment**

Once your LitServe application is running, let's test it out!

#### **Using Python (Recommended)**

Create a new Python file (e.g., `client.py`) and add the following:

```python
import requests
import json

# Replace with your deployed URL if you're using Lightning AI Cloud
url = "http://127.0.0.1:8000/predict" 

# Example input for our SimpleModel (10 numbers)
input_data = {"input": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}

try:
    response = requests.post(url, json=input_data)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

    print("Success!")
    print(json.dumps(response.json(), indent=2))

except requests.exceptions.HTTPError as err:
    print(f"HTTP Error: {err}")
    print(f"Response Content: {err.response.text}")
except requests.exceptions.ConnectionError as err:
    print(f"Connection Error: {err}. Make sure your server is running!")
except Exception as err:
    print(f"An unexpected error occurred: {err}")

```

Run this client script:

```bash
python client.py
```

You should see output similar to:

```
Success!
{
  "prediction": [
    -0.1009121686220169
  ]
}
```
(The exact prediction value will vary based on the random initialization of the `SimpleModel`).

#### **Using `curl` (for quick tests)**

If you prefer using `curl` from your terminal:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"input": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}' http://127.0.0.1:8000/predict
```

---

### **Next Steps and Advanced Considerations**

You've successfully deployed your first AI model with LitServe! Here are some key considerations for moving towards robust production deployments:

* **Error Handling:** Implement more specific error handling within your `decode_request` and `predict` methods to provide meaningful messages to your API users.
* **Authentication:** For production APIs, you'll need security. LitServe supports API key authentication out of the box.
* **Model Versioning:** As your models evolve, plan for how you'll manage different versions of your API.
* **Logging and Monitoring:** Set up comprehensive logging to track requests, responses, and potential issues, and integrate with monitoring tools.
* **Resource Tuning:** Experiment with `max_batch_size`, `batch_timeout`, `workers_per_device`, and the `accelerator` setting in `LitServer` to find the optimal configuration for your specific model and hardware.
* **Complex Pipelines:** LitServe is flexible enough to handle more complex scenarios where multiple models might be chained together in an inference pipeline.

LitServe provides a powerful yet user-friendly way to serve your AI models. By leveraging its optimizations and the streamlined deployment options, you can get your AI solutions into the hands of users faster and more efficiently. Happy deploying!
