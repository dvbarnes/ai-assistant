

import os

from deepeval import evaluate
from deepeval.models import AzureOpenAIModel
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import AnswerRelevancyMetric, ToolCorrectnessMetric
import dspy

from agents.CalendarManager.calendar_manager_agent import CalendarAgentResponse, CalendarManagerApp
from agents.models.user_context import UserContext

from dotenv import load_dotenv

load_dotenv()
model= AzureOpenAIModel(
    model=os.getenv("MODEL"),
    deployment_name=os.getenv("MODEL"),
    api_key=os.getenv("OPEN_API_KEY"),
    api_version="2025-01-01-preview",
    base_url=os.getenv("OPEN_API_URL"),
    temperature=0
)

app = CalendarManagerApp()
dspy.configure(
    lm=dspy.LM(
        os.getenv("MODEL_NAME"), 
        api_key=os.getenv("OPEN_API_KEY"), 
        api_base=os.getenv("OPEN_API_URL"),
        temperature=0
    )
)

user_context = UserContext(
            first_name=os.getenv("USER_FIRST_NAME"),
            last_name=os.getenv("USER_LAST_NAME"),
            email=os.getenv("USER_EMAIL")
        )

r1:CalendarAgentResponse = app(
    message="book a meeting with jim",
    context=user_context
)

test_case = LLMTestCase(
    input=r1.response,
    actual_output="I'm missing the following required information: Jim's email address, the meeting duration, and the preferred date or date range. Please provide these before I can continue.",
    # Replace this with the tools that was actually used by your LLM agent
    tools_called=[ToolCall(name= tool) for tool in r1.tools],
    expected_tools=[ToolCall(name= 'finish')],
)

r2:CalendarAgentResponse = app(
    message="john asked me to find some time this week to meet, please email him 3 times that I'm free his email is john@john.com",
    context=user_context
    )

test_case_2 = LLMTestCase(
    input=r2.response,
    actual_output="I have emailed John to check his availability for the proposed time slots Once John responds, I can help finalize the meeting.",
    # Replace this with the tools that was actually used by your LLM agent
    tools_called=[ToolCall(name= tool) for tool in r2.tools],
    expected_tools=[ToolCall(name="get_availability"),ToolCall(name="send_email")],
    
)

metric = ToolCorrectnessMetric(model=model)
answer_relevancy = AnswerRelevancyMetric(model=model, threshold=0.6)
evaluate(test_cases=[test_case,test_case_2], metrics=[metric, answer_relevancy])