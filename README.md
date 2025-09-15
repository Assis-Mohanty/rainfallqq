
---

# crossOrgin

A machine learning web application for detecting network security threats using FastAPI. This project allows users to train a model on network data and make real-time predictions through a web interface. It integrates with MongoDB for data storage and supports an end-to-end pipeline including ingestion, validation, transformation, training, and inference.

---

## 📑 Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [API Endpoints](#api-endpoints)
* [Examples](#examples)
* [Project Structure](#project-structure)
* [Dependencies](#dependencies)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [License](#license)

---

## 🚀 Features

* End-to-end machine learning pipeline (data ingestion → validation → transformation → training)
* Predictive inference using trained model
* MongoDB data insertion and retrieval
* Modular codebase using custom exception handling and logging
* Interactive FastAPI UI (`/docs`)
* HTML table visualization of predictions
* Environment variable handling with `.env`

---

## ⚙️ Installation

```bash
# Clone the repository
git clone <repo-url>
cd crossOrgin

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🔧 Configuration

Create a `.env` file in the root directory and set your MongoDB connection string:

```env
MONDO_DB_URL="your_mongodb_connection_string"
```

> **Note:** This file is already added to `.gitignore` for security.

---

## 📦 Usage

### Run the FastAPI Server

```bash
python app.py
```

Visit `http://localhost:8000/docs` for the interactive API documentation.

### Train the Model

```bash
curl -X GET http://localhost:8000/train
```

### Predict from CSV

Upload a CSV file via the `/predict` endpoint to get predictions and see results in HTML table format.

---

## 🧪 API Endpoints

| Method | Endpoint   | Description                     |
| ------ | ---------- | ------------------------------- |
| GET    | `/`        | Redirects to `/docs`            |
| GET    | `/train`   | Triggers model training         |
| POST   | `/predict` | Accepts CSV and returns results |

---

## 🧾 Examples

Example CSV upload:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path_to_file.csv"
```

Data Ingestion Example (in `push_data.py`):

```bash
python push_data.py
```

---

## 🧱 Project Structuree

```
├── app.py                 # FastAPI application
├── main.py                # Training pipeline entry point
├── push_data.py           # Push CSV data to MongoDB
├── test_mongo.py          # MongoDB connection test
├── .env                   # Environment variables (e.g., DB URL)
├── requirements.txt       # Project dependencies
├── networksecurity/       # Core logic, models, utilities
├── templates/             # HTML templates for UI
└── README.md              # Project documentation
```

---

## 📚 Dependencies

See `requirements.txt` for the full list. Key packages include:

* `FastAPI`, `uvicorn`
* `pymongo`, `python-dotenv`, `certifi`
* `pandas`, `numpy`, `scikit-learn`
* `mlflow`, `dill`, `pyaml`

---

## 🛠️ Troubleshooting

* **MongoDB Connection Failed**: Make sure `MONDO_DB_URL` is correctly set in `.env`.
* **CSV Upload Fails**: Ensure the file format is valid and matches expected schema.
* **Model Not Found**: Ensure training is done before prediction (`/train` must be called).

---

## 👥 Contributors

* \[Assis Mohanty] — Developer / Maintainer

---



MONDO_DB_URL="mongodb+srv://assismohanty98:Assis2004@cluster0.cxmzj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"# crossOrgin
# rainfallQ
# rainfallqq
