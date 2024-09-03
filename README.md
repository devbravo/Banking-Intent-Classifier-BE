# Customer Intent Classification in Banking Sector

This project aims to develop a customer intent classification model for the banking sector using PyTorch, BentoML, Docker, and Streamlit. The model predicts the intent behind customer service inquiries, with a total of 77 classes.

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

Customer service plays a crucial role in the banking industry, and efficiently categorizing customer intents can significantly enhance service quality and response times. This project leverages machine learning techniques to automatically classify customer inquiries into predefined categories, facilitating streamlined communication and support processes.

## Tech Stack

- **Model**: PyTorch is used for developing the machine learning model to classify customer intents.
- **Backend**: BentoML is employed to create a RESTful API endpoint for deploying the model.
- **Containerization**: Docker is utilized to containerize the application for seamless deployment and scalability.
- **Cloud Platform**: Google Cloud Platform (GCP) serves as the cloud infrastructure for hosting and managing the deployed application.
- **Front-end**: Python Streamlit is employed to build an intuitive web-based user interface for interacting with the model.

## Prerequisites

- Python 3.8 or higher
- Docker
- Google Cloud SDK (if deploying to GCP)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/Customer-Intent-Classification.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Customer-Intent-Classification
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Train the PyTorch model** using the provided dataset:

    ```bash
    python src/train.py --config config.yaml
    ```

2. **Export the trained model** and create a BentoML service:

    ```bash
    python src/export_model.py --model_path model.pth --bento_service service.py
    ```

3. **Build the Docker container:**

    ```bash
    docker build -t customer-intent-classification .
    ```

4. **Run the Docker container:**

    ```bash
    docker run -p 8080:8080 customer-intent-classification
    ```

5. **Access the Streamlit frontend** by navigating to `http://localhost:8080` in your web browser.

## Dataset

The dataset used for training the model should be placed in the `data/` directory. You can download the dataset from [link to dataset source]. Ensure that the dataset is in the correct format as expected by the training script.

## Deployment

To deploy the application on Google Cloud Platform:

1. Build the Docker container:

    ```bash
    docker build -t gcr.io/your-project-id/customer-intent-classification .
    ```

2. Push the container to Google Container Registry:

    ```bash
    docker push gcr.io/your-project-id/customer-intent-classification
    ```

3. Deploy the container to Google Cloud Run:

    ```bash
    gcloud run deploy --image gcr.io/your-project-id/customer-intent-classification --platform managed
    ```

4. Access the deployed application via the URL provided by Google Cloud Run.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to contribute to the project.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions, feel free to reach out via [diegofranco711@gmail.com](mailto:diegofranco711@gmail.com).