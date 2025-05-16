# VetChat: Dog Health Assistant

A Streamlit-powered chatbot for veterinary and dog-care questions, built with LangChain, LangGraph, and FAISS.


## 🔎 Features

- Filters out non-dog queries  
- Diagnoses diseases from symptoms  
- Suggests practical home-care methods  
- Flags urgency levels  
- Recommends nearby animal hospitals by ZIP code  
- Provides general dog health advice  


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
#Clone the repo
git clone https://github.com/YueMenaYang/GenAI-Vetchatbot.git
cd GenAI-Vetchatbot

#Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r code/requirements.txt
```

## 🔑 Configuration
```
# Create your .env file for API keys
cat <<EOF > code/.env
OPENAI_API_KEY=sk-...
GOOGLE_PLACES_API_KEY=AIza...
EOF

# Ensure it's ignored
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

