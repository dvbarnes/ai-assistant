
import os

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, ToolCorrectnessMetric
from deepeval.models import AzureOpenAIModel
from deepeval.test_case import LLMTestCase
import dspy

from agents.agent import Router


model= AzureOpenAIModel(
    model=os.getenv("MODEL"),
    deployment_name=os.getenv("MODEL"),
    api_key=os.getenv("OPEN_API_KEY"),
    api_version="2025-01-01-preview",
    base_url=os.getenv("OPEN_API_URL"),
    temperature=0
)


dspy.configure(
    lm=dspy.LM(
        os.getenv("MODEL_NAME"), 
        api_key=os.getenv("OPEN_API_KEY"), 
        api_base=os.getenv("OPEN_API_URL"),
        temperature=0
    )
)

def test_pass_in_request_about_contacts_returns_correct_label():
    # ARRANGE
    router = dspy.ChainOfThought(Router)
    
    # ACT
    result = router(user_request="John asked me to find some time this week to meet, please email him 3 times that I'm free.")
    print(result)
    # ASSERT    
    assert "CONTACT_READER" in result.get('needs')
    assert "CALENDAR_READER" in result.get('needs')
    assert "EMAIL_MANAGER" in result.get('needs')
    

