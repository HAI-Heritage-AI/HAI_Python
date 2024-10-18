from pydantic import BaseModel
from typing import List

class TravelRecommendationJ(BaseModel):
    destination: str
    recommendations: List[str]