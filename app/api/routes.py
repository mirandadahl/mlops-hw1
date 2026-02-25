import os
import json
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.services.email_topic_inference import EmailTopicInferenceService
from app.features.factory import FeatureGeneratorFactory
from app.dataclasses import Email

router = APIRouter()

class EmailRequest(BaseModel):
    subject: str
    body: str

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailClassificationResponse(BaseModel):
    classification_method: str
    predicted_topic: str
    topic_scores: Optional[Dict[str, float]] = None
    similarity: Optional[float] = None
    best_match: Optional[Dict[str, Any]] = None
    top_email_scores: Optional[List[Dict[str, Any]]] = None
    features: Dict[str, Any]
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    message: str
    email_id: int
    
# ADDED Classes
class TopicRequest(BaseModel):
    name: str
    description: str

class TopicAddResponse(BaseModel):
    message: str
    topic: str
    description: str
    total_topics: int

class StoredEmailRequest(BaseModel):
    subject: str
    body: str
    ground_truth: Optional[str] = None

class StoredEmailResponse(BaseModel):
    message: str
    email_id: int
    ground_truth: Optional[str] = None
    
# Help for file I/0
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

def _topics_path() -> str:
    return os.path.join(DATA_DIR, "topic_keywords.json")

def _emails_path() -> str:
    return os.path.join(DATA_DIR, "emails.json")

def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def _save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        

# Classificaiton Endpoint (updates made from the original)
@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(
    request: EmailRequest,
    method: str = Query(
        default="topic",
        description="Classification method: 'topic' (default) or 'email' (similarity to stored emails)",
    ),
):
    """Classify an email using either topic-description similarity or stored-email similarity."""
    try:
        if method not in ("topic", "email"):
            raise HTTPException(
                status_code=400,
                detail="method must be 'topic' or 'email'",
            )
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        result = inference_service.classify_email(email, method=method)

        return EmailClassificationResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Topics Endpoints (updates made from the original)

@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"], "details": info["topics_with_descriptions"]}


@router.post("/topics", response_model=TopicAddResponse)
async def add_topic(request: TopicRequest):
    """Dynamically add a new topic and persist it to the topics file.
    
    Assignment requirement #2: Create an endpoint to dynamically add new topics.
    """
    try:
        topic_name = request.name.strip().lower()
        description = request.description.strip()

        if not topic_name or not description:
            raise HTTPException(status_code=400, detail="Both name and description are required")

        topics_file = _topics_path()
        topic_data = _load_json(topics_file)

        if topic_name in topic_data:
            raise HTTPException(
                status_code=409,
                detail=f"Topic '{topic_name}' already exists. Use a different name.",
            )

        topic_data[topic_name] = {"description": description}
        _save_json(topics_file, topic_data)

        return TopicAddResponse(
            message=f"Topic '{topic_name}' added successfully",
            topic=topic_name,
            description=description,
            total_topics=len(topic_data),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/topics/{topic_name}")
async def delete_topic(topic_name: str):
    """Delete an existing topic"""
    try:
        topics_file = _topics_path()
        topic_data = _load_json(topics_file)

        if topic_name not in topic_data:
            raise HTTPException(status_code=404, detail=f"Topic '{topic_name}' not found")

        del topic_data[topic_name]
        _save_json(topics_file, topic_data)

        return {"message": f"Topic '{topic_name}' deleted", "remaining_topics": len(topic_data)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  Stored Emails Endpoints

@router.get("/emails")
async def get_emails():
    """List all stored emails"""
    emails = _load_json(_emails_path())
    return {"emails": emails, "total": len(emails)}


@router.post("/emails", response_model=StoredEmailResponse)
async def store_email(request: StoredEmailRequest):
    """Store an email with an optional ground_truth label.
    
    Assignment requirement #3: Create an endpoint to store emails with optional ground truth.
    The ground truth label is used by the email-similarity classifier.
    """
    try:
        emails_file = _emails_path()
        emails = _load_json(emails_file)

        new_id = len(emails) + 1
        entry = {
            "id": new_id,
            "subject": request.subject,
            "body": request.body,
            "ground_truth": request.ground_truth,
        }
        emails.append(entry)
        _save_json(emails_file, emails)

        return StoredEmailResponse(
            message="Email stored successfully",
            email_id=new_id,
            ground_truth=request.ground_truth,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#  Features Endpoint (LAB ASSIGNMENT - Part 2 of 2 )

@router.get("/features")
async def get_features():
    """Return information about all available feature generators.
    
    Completes Lab Assignment Part 2 of 2.
    """
    try:
        generators = FeatureGeneratorFactory.get_available_generators()
        return {"available_generators": generators}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  Pipeline info

@router.get("/pipeline/info")
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()
    
# TODO: LAB ASSIGNMENT - Part 2 of 2  
# Create a GET endpoint at "/features" that returns information about all feature generators
# available in the system.
#
# Requirements:
# 1. Create a GET endpoint at "/features"
# 2. Import FeatureGeneratorFactory from app.features.factory
# 3. Use FeatureGeneratorFactory.get_available_generators() to get generator info
# 4. Return a JSON response with the available generators and their feature names
# 5. Handle any exceptions with appropriate HTTP error responses
#
# Expected response format:
# {
#   "available_generators": [
#     {
#       "name": "spam",
#       "features": ["has_spam_words"]
#     },
#     ...
#   ]
# }
#
# Hint: Look at the existing endpoints above for patterns on error handling
# Hint: You may need to instantiate generators to get their feature names