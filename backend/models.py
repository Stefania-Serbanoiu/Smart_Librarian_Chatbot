#backend/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Book(BaseModel):
    """
    Model for representing a book
    """
    id: str
    title: str
    summary: str
    themes: List[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    """
    Model for representing a search request
    """
    query: str
    top_k: int = 4 


class SearchHit(BaseModel):
    """
    Model for representing a search hit
    """
    id: str
    document: str
    metadata: Dict[str, Any]


class RecommendationItem(BaseModel):
    """
    Model for representing a recommendation item from the RAG 
    """
    title: str
    rationale: str
    detailed_summary: str
    image_path: Optional[str] = None # it can have a generated image per item
    audio_path: Optional[str] = None # it can have a generated audio per item


class RecommendationRequest(BaseModel):
    """
    Model for representing a recommendation request
    """
    query: str
    top_k: int = 4                    # in how many results from rag to look into
    num_recommendations: int = 1      # nr of recommendations to generate in the final response
    language_filter: bool = True      # if we use language filter
    generate_image: bool = False     # if we generate an image for each item
    tts: bool = False                # if we use text-to-speech


class RecommendationResult(BaseModel):
    """
    Model for representing a recommendation result 
    """
    items: List[RecommendationItem] # we can have multiple recommendations in the same response 


class TTSRequest(BaseModel):
    """
    Model for representing a text-to-speech request
    """
    text: str
    voice: Optional[str] = "verse"   # we use "verse" as default voice
    filename: Optional[str] = "recommendation.wav"


class ImageRequest(BaseModel):
    """
    Model for representing an image request
    """
    title: str
    themes: Optional[str] = ""
    filename: Optional[str] = "cover.png"
