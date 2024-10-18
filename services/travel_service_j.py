import os
from perplexityai import Perplexity
from models.travel_model_j import TravelRecommendationJ

class TravelServiceJ:
    def __init__(self):
        self.perplexity = Perplexity(api_key=os.getenv("PERPLEXITY_API_KEY"))

    async def get_recommendation(self, destination: str) -> TravelRecommendationJ:
        query = f"여행 추천: {destination}에 가면 무엇을 할 수 있을까요? 3가지 추천해주세요."
        response = await self.perplexity.query(query)
        recommendations = response.split("\n")[:3]
        return TravelRecommendationJ(destination=destination, recommendations=recommendations)