# from pydantic import BaseModel
# from typing import List, Optional

# class RecommendRequest(BaseModel):
#     question: str
#     k: Optional[int] = 4

# class Recommendation(BaseModel):
#     recommendation: str
#     explanation: str
#     retrieved: List[str]

# class RecommendResponse(BaseModel):
#     success: bool
#     result: Recommendation | None = None
#     error: str | None = None

from pydantic import BaseModel
from typing import List, Optional, Any

class RecommendRequest(BaseModel):
    question: str
    k: Optional[int] = 4

class Recommendation(BaseModel):
    recommendation: str
    explanation: str
    retrieved: List[Any] 

class RecommendResponse(BaseModel):
    success: bool
    result: Recommendation | None = None
    error: str | None = None