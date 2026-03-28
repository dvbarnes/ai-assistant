

import os
from typing import Optional
from unittest.mock import MagicMock

from deepeval import evaluate
from deepeval.models import AzureOpenAIModel
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import AnswerRelevancyMetric, ToolCorrectnessMetric
import dspy

from dotenv import load_dotenv
from dspy.adapters import Tool

from agents.ContactsAgent.contacts_agent import ContactsAgentResponse, ContactsManagerApp
from agents.ContactsAgent.tools import Contact, lookup_user
from agents.models.user_context import UserContext
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


def test_happy_path_returns_correct_user():
    lookup_tool = MockTool(func=lookup_user, ret_value=[Contact(
            first_name= "jim",
            last_name = "Doe",
            email = "john_doe@your_company.com",
            company = "your_company",
        )])
    app = ContactsManagerApp(tools=[lookup_tool])
    r1:ContactsAgentResponse = app(
        message="book a meeting with jim",
        context=user_context
    )

    test_case = LLMTestCase(
        input=r1.response,
        actual_output="Jim's contact information has been successfully retrieved: \n- Name: Jim Doe\n- Email: john_doe@your_company.com\n- Company: your_company\n\nYou can now use this information to book a meeting with him.",
        # Replace this with the tools that was actually used by your LLM agent
        tools_called=[ToolCall(name= tool) for tool in r1.tools],
        expected_tools=[ToolCall(name= 'lookup_user')],
    )
    return test_case

def test_user_not_found_returns_correct_message():
    lookup_tool = MockTool(func=lookup_user, ret_value=[])
    app = ContactsManagerApp(tools=[lookup_tool])
    r2:ContactsManagerApp = app(
        message="book a meeting with John",
        context=user_context
        )

    test_case = LLMTestCase(
        input=r2.response,
        actual_output="I was unable to find contact information for John. Please provide additional details, such as John's last name, email address, or company, to help refine the search.",
        # Replace this with the tools that was actually used by your LLM agent
        tools_called=[ToolCall(name= tool) for tool in r2.tools],
        expected_tools=[ToolCall(name="finish")],
        
    )
    return test_case

def test_user_in_mulitple_companies_infers_correct_user():
    lookup_tool = MockTool(func=lookup_user, ret_value=[
            Contact(
                first_name="Jack",
                last_name="Doe",
                email="Jack_doe@your_company.com",
                company="your_company",
            ),
        Contact(
            first_name="Jack",
                last_name="Doe",
                email="Jack_doe@your_company.com",
                company="Microsoft"
            )

    ])
    app = ContactsManagerApp(tools=[lookup_tool])
    r3:ContactsManagerApp = app(
        message="book a meeting with Jack over at MSFT",
        context=user_context
        )

    test_case = LLMTestCase(
        input=r3.response,
        actual_output="The contact information for Jack at Microsoft is: email - Jack_doe@your_company.com. You can use this to proceed with booking the meeting.",
        # Replace this with the tools that was actually used by your LLM agent
        tools_called=[ToolCall(name= tool) for tool in r3.tools],
        expected_tools=[ToolCall(name="lookup_user"), ToolCall(name="finish")],
        
    )
    return test_case

def test_user_in_same_company_cannot_continue():
    lookup_tool = MockTool(func=lookup_user, ret_value=[
            Contact(
                first_name="Jack",
                last_name="Doe",
                email="Jack_doe123@your_company.com",
                company="Microsoft",
            ),
        Contact(
            first_name="Jack",
                last_name="Doe",
                email="Jack_doe@your_company.com",
                company="Microsoft"
            )

    ])
    app = ContactsManagerApp(tools=[lookup_tool])
    r3:ContactsManagerApp = app(
        message="book a meeting with Jack over at MSFT",
        context=user_context
        )

    test_case = LLMTestCase(
        input=r3.response,
        actual_output="""I found two possible contacts for "Jack" at Microsoft:

1. Jack Doe - Email: Jack_doe123@your_company.com
2. Jack Doe - Email: Jack_doe@your_company.com

Please confirm which contact is correct, or provide additional details to narrow down the search.""",
        # Replace this with the tools that was actually used by your LLM agent
        tools_called=[ToolCall(name= tool) for tool in r3.tools],
        expected_tools=[ToolCall(name="lookup_user"), ToolCall(name="finish")],
        
    )
    return test_case

metric = ToolCorrectnessMetric(model=model)
answer_relevancy = AnswerRelevancyMetric(model=model, threshold=0.6)
evaluate(test_cases=[
    test_happy_path_returns_correct_user(),
    test_user_not_found_returns_correct_message(), 
    test_user_in_mulitple_companies_infers_correct_user(),
    test_user_in_same_company_cannot_continue()
    ], metrics=[metric, answer_relevancy])