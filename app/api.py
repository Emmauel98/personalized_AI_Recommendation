from fastapi import APIRouter, HTTPException
from app.models import RecommendRequest, RecommendResponse, Recommendation
from app.services.llm_service import LLMService

router = APIRouter()
llm_service = LLMService()

@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    try:
        result = llm_service.recommend(question=req.question, k=req.k)

        rec = Recommendation(
            # recommendation=result["recommendation"],
            # explanation=result["explanation"],
            # retrieved=result["retrieved"],
            # used_items=result["used_items"]
            recommendation=result.get("recommendation", "N/A"),
            explanation=result.get("explanation", "Error in LLM output structure."),
            # Assuming 'retrieved' in the result dictionary holds the metadata list
            retrieved=result.get("retrieved", [])
        )

        return RecommendResponse(success=True, result=rec)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
