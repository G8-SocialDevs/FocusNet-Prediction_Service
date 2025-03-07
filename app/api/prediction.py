from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import calendar as cal
from app.database import get_db
from app.models.task import Task
from app.services.predict import predict

router = APIRouter()

@router.post("/prediction")
def predict_category(title: str, description: str, db: Session = Depends(get_db)):
    try:
        return predict(title, description, db)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))