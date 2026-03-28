

from datetime import datetime
from locale import format_string
import os

from deepeval import evaluate
from deepeval.models import AzureOpenAIModel
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import AnswerRelevancyMetric, ToolCorrectnessMetric
import dspy

from agents.CalendarManager.calendar_manager_agent import CalendarAgentResponse, CalendarManagerApp
from agents.CalendarManager.tools.tools import Date, get_availability, send_email
from agents.models.user_context import UserContext

from dotenv import load_dotenv

from tests.mock_tool import MockTool

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

date_format_string = "%Y-%m-%d %H:%M:%S"

def test_missing_required_information_returns_correct_message():
    
    app = CalendarManagerApp()
    r1:CalendarAgentResponse = app(
        message="book a meeting with jim",
        context=user_context
    )

    test_case = LLMTestCase(
        input=r1.response,
        actual_output="I'm missing the following required information: attendee email address, meeting duration, and preferred date or date range. Please provide these before I can continue.",
        # Replace this with the tools that was actually used by your LLM agent
        tools_called=[ToolCall(name= tool) for tool in r1.tools],
        expected_tools=[ToolCall(name= 'finish')],
    )
    return test_case

def test_has_all_correct_info_sends_email():
    
    availability_tool_mock = MockTool(func=get_availability, ret_value=[
    Date(
        start_time=datetime.strptime("2026-03-25 09:30:00", date_format_string),
        end_time=datetime.strptime("2026-03-25 11:30:00", date_format_string),
        free = True
    ),
    Date(
        start_time=datetime.strptime("2026-03-26 09:30:00", date_format_string),
        end_time=datetime.strptime("2026-03-26 11:30:00", date_format_string),
        free = True
        ),
    Date(
        start_time=datetime.strptime("2026-03-27 09:30:00", date_format_string),
        end_time=datetime.strptime("2026-03-27 11:30:00", date_format_string),
        free = True
        )
])
    send_email_mock = MockTool(func= send_email)

    mock_tools = [t for t in CalendarManagerApp.tools if t.__name__ != get_availability.__name__]
    mock_tools = [t for t in CalendarManagerApp.tools if t.__name__ != send_email.__name__ ]
    mock_tools = [MockTool(func=t) for t in CalendarManagerApp.tools ]
    mock_tools.append(availability_tool_mock)
    mock_tools.append(send_email_mock)

    app = CalendarManagerApp(tools = mock_tools)
    r2:CalendarAgentResponse = app(
        message="john asked me to find some time this week to meet, please email him 3 times that I'm free his email is john@john.com",
        context=user_context
        )
    
    test_case = LLMTestCase(
        input=r2.response,
        actual_output="I have emailed John to check his availability for the proposed time slots Once John responds, I can help finalize the meeting.",
        # Replace this with the tools that was actually used by your LLM agent
        tools_called=[ToolCall(name= tool) for tool in r2.tools],
        expected_tools=[ToolCall(name=get_availability.__name__),ToolCall(name=send_email.__name__)],
    )
    return test_case

metric = ToolCorrectnessMetric(model=model)
answer_relevancy = AnswerRelevancyMetric(model=model, threshold=0.6)
evaluate(test_cases=[
    test_missing_required_information_returns_correct_message(),
    test_has_all_correct_info_sends_email()
    ], metrics=[metric, answer_relevancy])