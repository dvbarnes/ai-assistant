

import os

from deepeval import evaluate
from deepeval.models import AzureOpenAIModel
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric
import dspy

from agents.CalendarManager.calendar_manager_agent import CalendarManagerApp
from agents.models.user_context import UserContext

from dotenv import load_dotenv

load_dotenv()
model= AzureOpenAIModel(
    model=os.getenv("MODEL_NAME"),
    deployment_name=os.getenv("MODEL_NAME"),
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
        api_base=os.getenv("OPEN_API_URL")
    )
)

test_case = LLMTestCase(
    input=app(message="book a meeting with john",
            context=UserContext(
            first_name=os.getenv("USER_FIRST_NAME"),
            last_name=os.getenv("USER_LAST_NAME"),
            email=os.getenv("USER_EMAIL")
        )),
    actual_output="We offer a 30-day full refund at no extra cost.",
    # Replace this with the tools that was actually used by your LLM agent
    tools_called=[ToolCall(name="get_availability"), ToolCall(name="get_help")],
    expected_tools=[ToolCall(name="get_availability")],
)


test_case_2 = LLMTestCase(
    input=app(message="ohn asked me to find some time this week to meet, please email him 3 times that I'm free his email is john@john.com",
            context=UserContext(
            first_name=os.getenv("USER_FIRST_NAME"),
            last_name=os.getenv("USER_LAST_NAME"),
            email=os.getenv("USER_EMAIL")
        )),
    actual_output="We offer a 30-day full refund at no extra cost.",
    tools_called=[ToolCall(name="get_availability"), ToolCall(name="send_email")],
    expected_tools=[ToolCall(name="get_availability"),ToolCall(name="send_email")],
)

metric = ToolCorrectnessMetric(model=model)

evaluate(test_cases=[test_case,test_case_2], metrics=[metric])