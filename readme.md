# VetChat: Dog Health Assistant

A Streamlit-powered chatbot for veterinary and dog-care questions, built with LangChain, LangGraph, and FAISS.


## 🔎 Features

- Filters out non-dog queries  
- Diagnoses diseases from symptoms  
- Suggests practical home-care methods  
- Flags urgency levels  
- Recommends nearby animal hospitals by ZIP code  
- Provides general dog health advice  


## 🧪 Experiment Results

We evaluated VetChat’s performance using a series of test queries designed to represent different real-world use cases, including:
- Common symptom-based queries
- General health advice queries
- Requests for hospital recommendations
- Irrelevant or non-dog-related inputs

The chatbot's behavior, responses, and flow through the multi-agent system were recorded during these tests. You can watch a demo of the chatbot in action here:

https://www.youtube.com/watch?v=MkC_taBFrC0


## 📁 Repository Structure

```
vetdogchatbot/
├─ code/
│  ├─ data.py
│  ├─ app.py
│  ├─ pmodel.py
│  ├─ prompt.py
│  ├─ requirements.txt
│  └─ .env             # ignored by Git
├─ data/
│  └─ faiss_index/
│  └─ ......           # datasets
├─ .gitignore
└─ README.md
```

## ⚙️ Prerequisites

- Python 3.8+  
- `git`, `venv` (or `conda`)  

## 🚀 Setup 

On terminal:
```
# Clone the repo
git clone https://github.com/YueMenaYang/GenAI-Vetchatbot.git
cd GenAI-Vetchatbot

# Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r code/requirements.txt
```

## 🔑 Configuration
```
# Create your .env file for API keys
# Replace the placeholders with your own API keys before running:
# OPENAI_API_KEY=sk-... → your OpenAI key
# GOOGLE_PLACES_API_KEY=AIza... → your Google Places key
cat <<EOF > code/.env
OPENAI_API_KEY=sk-...
GOOGLE_PLACES_API_KEY=AIza...
EOF

# Ensure the .env file is ignored by Git (prevents accidental commits of API keys)
grep -qxF "code/.env" .gitignore || echo "code/.env" >> .gitignore
```

## 🎉 Run
```
cd code
python -m streamlit run app.py   
```

## 👥 Authors 

Yue Yang, yy3532 

Mengxi Liu, ml5189

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 📚 References
Freeman, K. P., & Klenner, S. (2015). Veterinary Clinical Pathology: A Case-Based Approach. CRC Press.

Lovejoy, J. (2023). “8 Vet-Approved Home Remedies for Your Dog.” PetMD.

Oliveira, W. (2024). “Animal Condition Classification Dataset.” Kaggle.

Shojai, A. (2016). Dog Facts: The Pet Parent’s A-to-Z Home Care Encyclopedia. Furry Muse.

