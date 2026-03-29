
from typing import Literal

from deepeval.tracing import observe
import dspy
from pydantic import BaseModel

from agents.CalendarManager.calendar_manager_agent import CalendarManagerApp
from agents.ContactsAgent.contacts_agent import ContactsManagerApp
from agents.agent_tools import get_user_information
from agents.models.user_context import UserContext


class AgentResponse(BaseModel):
    response: str
    tools: list[str]

class AIAssistantAgent(dspy.Signature):
    """
    You are a personal assisant helping to fullifll users requests.
    STRICT RULES — follow these exactly, no exceptions:
    1. NEVER invent, assume, or infer an email address under any circumstances — 
   not from a name, a company, or context clues.

    2. ONLY use data explicitly provided by the user or returned by a tool. 
   Do not use your own knowledge.

    3. If a tool returns no results or an error, report that result exactly. 
   Do not substitute your own answer.

   4. If the capability is missing:
   - You MUST stop
   - You MUST explain that you lack the capability
   - You MUST NOT attempt to simulate or complete the task
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


def ask_contacts_agent(message: str, user_context: UserContext):
    agent = ContactsManagerApp()
    result =agent(message = message, context=user_context)
    return result.response

class AIAssistantApp(dspy.Module):
    tools = [
        ask_calendar_agent,
        ask_contacts_agent
    ]
    def __init__(self, tools = tools):
        super().__init__()
        self.lead_agent = dspy.ReAct(AIAssistantAgent, tools=tools)

    def forward(self, message: str):
        result = self.lead_agent(user_request=message, user_context= get_user_information())
        print(result)
        return result.get('process_result')

