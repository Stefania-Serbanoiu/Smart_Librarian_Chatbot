#backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from .models import (
    SearchRequest,
    RecommendationRequest,
    RecommendationResult,
    RecommendationItem,
    TTSRequest,
    ImageRequest,
)
from .db import search
from .rag import run_recommendation_pipeline_multi
from .tools import tts_save, generate_book_image


app = FastAPI(title="Smart Librarian – RAG + Tool")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/rag/search")
def rag_search(query: str, top_k: int = 4):
    hits = search(query, top_k=top_k)
    return {"hits": hits}


@app.post("/recommend", response_model=RecommendationResult)
def recommend(req: RecommendationRequest):
    recs = run_recommendation_pipeline_multi(
        query=req.query,
        top_k=req.top_k,
        num_recs=req.num_recommendations,
        language_filter=req.language_filter,
    )

    # Inadequate language warning
    if recs is None:
        return RecommendationResult(items=[
            RecommendationItem(
                title="",
                rationale="Te rog folosește un limbaj adecvat și reîncearcă.",
                detailed_summary=""
            )
        ])

    if not recs:
        return RecommendationResult(items=[
            RecommendationItem( # No matches found
                title="",
                rationale="Nu am găsit potriviri. Încearcă altă formulare.",
                detailed_summary=""
            )
        ])

    items = []
    for (title, rationale, detailed) in recs:
        image_path = None
        audio_path = None
        if req.generate_image:
            out_img = Path("generated") / f"{title.replace(' ', '_').lower()}_cover.png"
            generate_book_image(title, "", out_img)
            image_path = str(out_img)
        if req.tts:
            out_audio = Path("generated") / f"{title.replace(' ', '_').lower()}_rec.wav"
            tts_text = f"Recomandarea mea: {title}. Pe scurt: {rationale}. Rezumat: {detailed}"
            tts_save(tts_text, out_audio)
            audio_path = str(out_audio)
        items.append(RecommendationItem(
            title=title,
            rationale=rationale,
            detailed_summary=detailed,
            image_path=image_path,
            audio_path=audio_path
        ))

    return RecommendationResult(items=items)


@app.post("/tts")
def tts_ep(req: TTSRequest):
    out = Path("generated") / req.filename
    tts_save(req.text, out)
    return {"audio_path": str(out)}


@app.post("/image")
def image_ep(req: ImageRequest):
    out = Path("generated") / req.filename
    generate_book_image(req.title, req.themes or "", out)
    return {"image_path": str(out)}
