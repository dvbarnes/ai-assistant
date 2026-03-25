

import os

from deepeval import evaluate
from deepeval.models import AzureOpenAIModel
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import AnswerRelevancyMetric, ToolCorrectnessMetric
import dspy

from dotenv import load_dotenv

from agents.ContactsAgent.contacts_agent import ContactsAgentResponse, ContactsManagerApp
from agents.models.user_context import UserContext

load_dotenv()
model= AzureOpenAIModel(
    model=os.getenv("MODEL"),
    deployment_name=os.getenv("MODEL"),
    api_key=os.getenv("OPEN_API_KEY"),
    api_version="2025-01-01-preview",
    base_url=os.getenv("OPEN_API_URL"),
    temperature=0
)

app = ContactsManagerApp()
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

r2:ContactsManagerApp = app(
    message="book a meeting with John",
    context=user_context
    )

test_case_2 = LLMTestCase(
    input=r2.response,
    actual_output="I was unable to find contact information for John. Please provide additional details, such as John's last name, email address, or company, to help refine the search.",
    # Replace this with the tools that was actually used by your LLM agent
    tools_called=[ToolCall(name= tool) for tool in r2.tools],
    expected_tools=[ToolCall(name="finish")],
    
)

r3:ContactsManagerApp = app(
    message="book a meeting with Jack over at MSFT",
    context=user_context
    )

test_case_3 = LLMTestCase(
    input=r3.response,
    actual_output="The contact information for Jack at Microsoft is: email - Jack_doe@your_company.com. You can use this to proceed with booking the meeting.",
    # Replace this with the tools that was actually used by your LLM agent
    tools_called=[ToolCall(name= tool) for tool in r3.tools],
    expected_tools=[ToolCall(name="finish")],
    
)

metric = ToolCorrectnessMetric(model=model)
answer_relevancy = AnswerRelevancyMetric(model=model, threshold=0.6)
evaluate(test_cases=[test_case,test_case_2, test_case_3], metrics=[metric, answer_relevancy])