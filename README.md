# Smart_Librarian_Chatbot
An AI chatbot that acts as a librarian. It recommends books depending on the user's interests and preferences, using OpenAI GPT + RAG (ChromaDB).


## Installing and running the project
1. Create virtual environment venv:
```
py -m venv venv
```

2. Activate virtual environment from cmd:
```
venv\Scripts\activate.bat 
```

3. Install requirements:
```
pip install -r requirements.txt
```


4. Run backend:
```
uvicorn backend.app:app --reload --port 8000
```

5. Run frontend: (streamlit UI):
```
streamlit run frontend/streamlit_app.py
```


## Features

- **Book Database (RAG Source)** : Stores book summaries, titles, and themes in ChromaDB (at startup, the database is seeded from `data/books.json`).  

- **Semantic Search with OpenAI Embeddings**    

- **AI Recommendations (GPT + RAG)**  : chooses the best matches from RAG context, (uses `gpt-4o-mini` -> it can be changed from `config.py`)

- **Language Filter** : Detects inappropriate/offensive words and blocks requests politely.


- **Text-to-Speech (TTS):** Generates audio recommendations.

- **Image Generation:** Creates a book-cover style illustration.


- **Frontend (Streamlit)**  : Simple web UI. 
## Example Queries
- “O carte cu prietenie si magie”  
- “Ce recomanzi pentru cineva care prefera carti fantasy?”  
- “Caut o distopie cu teme puternice”  
- “Ce este 1984?”  
- “Recomandari de citit pentru o zi ploioasa”  
- “Ceva ce aduce aminte de Paris/New York”    
- “Vreau o carte despre libertate și control social.”  
  


---
