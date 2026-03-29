
from typing import Literal

from deepeval.tracing import observe
import dspy
from pydantic import BaseModel

from agents.CalendarManager.calendar_manager_agent import CalendarManagerApp
from agents.ContactsAgent.contacts_agent import ContactsManagerApp
from agents.EmailManager.email_manager_agent import EmailManagerApp
from agents.agent_tools import get_user_information
from agents.models.user_context import UserContext


class AgentResponse(BaseModel):
    response: str
    tools: list[str]

class AIAssistantAgent(dspy.Signature):
    """
    You are a personal assisant helping to fullifll users requests.
    MANDATORY BEHAVIOR:

    1. Break the user’s request into required steps.

    2. For each step:
    - Determine if you have the capability to complete it
    - If YES → complete the step
    - If NO → stop at that step and explain what is missing

    3. You MUST complete all possible steps before stopping.

    4. DO NOT skip steps or jump directly to the final goal.

    STRICT RULES:
    - NEVER invent or infer an email address
    - ONLY use tool outputs for contact information
    - NEVER simulate sending an email

    FAILURE RULE:
    - Only declare failure at the exact step where capability is missing

    """

    user_request: str = dspy.InputField()
    user_context: UserContext = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
                "Message that summarizes the process result, and the information users need"
            )
        )

def ask_calendar_agent(message: str, user_context: UserContext):
    agent = CalendarManagerApp()
    result =agent(message = message, context=user_context)
    return result.response


def ask_contacts_agent(agent_request: str, user_context: UserContext):
    agent = ContactsManagerApp()
    result =agent(message = agent_request, context=user_context)
    return result.response

def ask_email_agent(agent_request: str, user_context: UserContext):
    """
    Please include any information that is relevant to sending the email
    """
    agent = EmailManagerApp()
    result =agent(message = agent_request, context=user_context)
    return result.response

class AIAssistantApp(dspy.Module):
    tools = [
        ask_calendar_agent,
        ask_contacts_agent,
        ask_email_agent
    ]
    def __init__(self, tools = tools):
        super().__init__()
        self.lead_agent = dspy.ReAct(AIAssistantAgent, tools=tools)

    def forward(self, message: str):
        result = self.lead_agent(user_request=message, user_context= get_user_information())
        print(result)
        return result.get('process_result')

