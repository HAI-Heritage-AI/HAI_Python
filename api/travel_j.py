# travel_j.py
from fastapi import APIRouter, Depends
from services.travel_service_j import TravelServiceJ
from models.travel_model_j import TravelRecommendationJ

router = APIRouter()

@router.get("/recommend", response_model=TravelRecommendationJ)
async def recommend_travel(destination: str, service: TravelServiceJ = Depends(TravelServiceJ)):
    return await service.get_recommendation(destination)