# 🧾 Receipt OCR and Smart Expense Tracker

**An ML-powered receipt recognition and expense management system using OCR, Transfer Learning, and Machine Learning.**

Convert receipt images into structured expense records automatically. The system recognises printed text, extracts important transaction details, categorises expenses, and provides intelligent spending insights.

**Demo Video- https://drive.google.com/file/d/1efbeLu5S-KQ5CgxWY3X8069-3WZgWQM7/view?usp=sharing
---

## 🚀 Overview

Receipt OCR and Smart Expense Tracker is an end-to-end Machine Learning application designed to automate expense tracking using receipt images.

Instead of manually entering transaction details, users can upload a receipt and let the system extract the merchant name, transaction date, and total amount. It also predicts expense categories and provides spending analytics, monthly forecasts, and unusual spending alerts.

The project combines traditional OCR techniques, deep learning-based text recognition, and machine learning algorithms to build an intelligent expense management solution.

## ✨ Key Features

- **Automated Receipt Scanning:** Upload receipt images and extract information automatically.
- **Multiple OCR Engines:** Compare Tesseract, EasyOCR, and TrOCR.
- **Image Preprocessing:** Grayscale conversion, resizing, denoising, deskewing, and adaptive thresholding.
- **Smart Information Extraction:** Extract merchant name, transaction date, and total amount.
- **Expense Categorisation:** Automatically classify expenses using Multinomial Naive Bayes with keyword-based fallback.
- **Personalised Learning:** Improve category predictions using user-confirmed expense records.
- **Expense Dashboard:** View monthly summaries, category breakdowns, and transaction history.
- **Spending Forecasting:** Predict next month's expenses using Linear Regression.
- **Anomaly Detection:** Identify unusual transactions using statistical thresholds.
- **Data Export:** Download expense records in CSV format.
- **Personal and Business Profiles:** Maintain separate expense records.

## 🧠 Machine Learning Models and Techniques

The project uses multiple Machine Learning and Deep Learning techniques.

| Model / Technique | Purpose |
|---|---|
| Tesseract OCR | Traditional OCR baseline |
| EasyOCR (CRAFT + CRNN) | Text detection and recognition |
| TrOCR (Transformer) | Deep learning-based receipt text recognition |
| Multinomial Naive Bayes | Expense category classification |
| Linear Regression | Monthly spending prediction |
| Statistical Anomaly Detection | Identify unusual transactions |

### 1. Optical Character Recognition (OCR)

Three OCR engines are implemented and evaluated:

- **Tesseract:** Traditional OCR engine used as a baseline.
- **EasyOCR:** Uses CRAFT for text detection and CRNN for text recognition.
- **TrOCR:** Transformer-based OCR model fine-tuned for printed receipt text.

TrOCR uses a Vision Transformer encoder and an autoregressive Transformer decoder to recognise text from receipt images.

### 2. Expense Category Classification

Multinomial Naive Bayes is used to classify expenses into seven categories:

- Food & Drink
- Groceries
- Transport
- Health
- Shopping
- Bills
- Office

The classifier learns from previously saved and user-labelled expenses. When sufficient training data or prediction confidence is unavailable, keyword-based rules act as a fallback.

### 3. Spending Forecasting

Ordinary Least Squares Linear Regression is used to analyse the previous six months of expenses and estimate the next month's spending.

### 4. Anomaly Detection

The system identifies unusually high transactions using category-specific historical spending patterns.

A transaction is flagged when it exceeds both:

- Mean spending + 2.5 standard deviations
- 1.5 times the mean spending

---

## 🏗️ System Architecture

The application follows a modular pipeline:

```text
          Receipt Image
                |
                ▼
       Image Preprocessing
              (OpenCV)
                |
                ▼
         Text Detection
             (CRAFT)
                |
                ▼
        Text Recognition
                |
       ┌────────┼────────┐
       ▼        ▼        ▼
   Tesseract EasyOCR   TrOCR
       └────────┼────────┘
                |
                ▼
           Row Grouping
                |
                ▼
        Information Extraction
         (Company, Date, Total)
                |
                ▼
       Expense Categorisation
       (Naive Bayes + Rules)
                |
                ▼
        Expense Management
                |
                ▼
        Spending Analytics
       ┌────────┼────────┐
       ▼        ▼        ▼
    Monthly   Forecast  Anomaly
    Summary   Prediction Detection
```

## 📊 Dataset

### SROIE 2019 Dataset

The project uses the **ICDAR 2019 SROIE (Scanned Receipts OCR and Information Extraction)** dataset for OCR training and evaluation.

Dataset components:

- `img/` – Receipt images.
- `box/` – Text bounding boxes and corresponding transcriptions.
- `entities/` – Ground-truth information such as company, date, address, and total.

### Dataset Split

| Split | Receipts | Line Crops |
|---|---:|---:|
| Training | 474 | 25,278 |
| Validation | 52 | 2,783 |
| Testing | 100 | 5,564 |

The dataset is split by receipt rather than individual text lines to prevent data leakage.

The expense category classifier is evaluated separately using 1,809 labelled expenses across 28 merchants.

## 📈 Model Evaluation

### OCR Performance

The following results were obtained on 100 held-out SROIE test receipts.

| OCR Engine | CER | WER | Company | Date | Total |
|---|---:|---:|---:|---:|---:|
| Tesseract | 21.9% | 42.7% | 22% | 69% | 46% |
| EasyOCR | 23.7% | 58.3% | 21% | 80% | 35% |
| **TrOCR** | **16.6%** | **27.1%** | **38%** | **92%** | **63%** |

TrOCR achieved the best overall performance across the evaluated OCR metrics.

### Expense Classification Performance

| Method | Random 80/20 Split | Unseen Merchants |
|---|---:|---:|
| Majority Class Baseline | 35.9% | 24.7% |
| Keyword Rules | 41.2% | 43.9% |
| Multinomial Naive Bayes | 100% | 49.5% |
| **Naive Bayes + Rules** | **100%** | **64.8%** |

The combined Naive Bayes and keyword-rule approach achieved the best performance on unseen merchants.

*Note: The random split includes merchants appearing in both training and testing data, so unseen-merchant evaluation provides a more challenging generalisation test.*

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Image Processing | OpenCV, NumPy, Pillow |
| OCR | Tesseract, EasyOCR, TrOCR |
| Deep Learning | PyTorch, Hugging Face Transformers |
| Machine Learning | Scikit-learn |
| Backend | FastAPI, Uvicorn |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Evaluation | JiWER (CER, WER) |
| Deployment | Docker, Hugging Face Spaces / Render |

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

Install Tesseract OCR separately and ensure it is added to your system PATH.

### 5. Run the Application

Start the FastAPI server using:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

Open the application in your browser using the local URL configured by your application.

*Note: Replace `main:app` with the actual FastAPI module and application variable name if they differ in your repository.*

---

## 📂 Project Structure

```text
Receipt-OCR-Expense-Tracker/
│
├── backend/
│   ├── main.py
│   ├── ocr.py
│   ├── preprocessing.py
│   ├── extraction.py
│   └── classifier.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── training/
│   ├── prepare_data.py
│   ├── train_trocr.py
│   └── metrics.json
│
├── tests/
│
├── data/
│   └── README.md
│
├── requirements.txt
├── Dockerfile
└── README.md
```

*The structure above is illustrative. Adjust the folder and file names to match your actual repository.*

## 🔬 Evaluation Metrics

The OCR system is evaluated using:

- **Character Error Rate (CER):** Measures character-level recognition errors.
- **Word Error Rate (WER):** Measures word-level recognition errors.
- **Field Exact-Match Accuracy:** Measures the accuracy of extracted company names, dates, and totals.
- **Processing Time:** Measures the time required to process a receipt.

The expense classifier is evaluated using classification accuracy on random and unseen-merchant test splits.

## 🔮 Future Improvements

- Replace rule-based field extraction with LayoutLM or another layout-aware information extraction model.
- Extend multilingual support for Hindi and Punjabi receipts.
- Improve receipt detection under glare, folds, and perspective distortion.
- Introduce confidence-based field verification.
- Support batch receipt uploads.
- Improve generalisation to Indian receipt formats and currencies.
- Enhance spending predictions with more advanced time-series models.

## 👩‍💻 Author

**Palak Mahajan**

Computer Engineering  
Thapar Institute of Engineering and Technology, Patiala

## 📚 References

1. Tesseract OCR – https://github.com/tesseract-ocr/tesseract
2. EasyOCR – https://github.com/JaidedAI/EasyOCR
3. TrOCR – https://arxiv.org/abs/2109.10282
4. SROIE 2019 Dataset – ICDAR 2019 Receipt OCR and Information Extraction.
5. Hugging Face Transformers – https://huggingface.co/docs/transformers
6. OpenCV – https://opencv.org/

---

**If you find this project interesting, consider giving the repository a ⭐!**
