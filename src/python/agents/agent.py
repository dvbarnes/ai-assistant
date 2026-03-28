
from typing import Literal

from deepeval.tracing import observe
import dspy
from pydantic import BaseModel

from agents.CalendarManager.calendar_manager_agent import CalendarManagerApp
from agents.ContactsAgent.contacts_agent import ContactsManagerApp
from agents.models.user_context import UserContext


class AgentResponse(BaseModel):
    response: str
    tools: list[str]

class Router(dspy.Signature):
    """
You are a helpful assistant. You are responsile for categorizing user questions.
If the user's question is about a product, please return *CALENDAR.
For example, if the user asks a question about Booking meetings, please
return
*CALENDAR

If the user's question is about a Contact, please return *CONTACT.
For example, if the user asks a question about john at microsoft,
please return
*CONTACT

If you are not sure which category a user's question belongs to, return
*CLARIFY followed by a request for clarification in square
brackets. Your request should try to gain enough information
from the user to decide which of the above 2 categories you should
choose for their question.

For example,

if the user enters:
12345689

Please return:

*CLARIFY [I'm sorry but I don't understand what you are asking. Are 
you looking for a product or an order?]

Remember that you ONLY have access to information in our Calendars and
Contacts databases. If the user asks for information which would
not be in either of those databases, please let them know that you do
not have access to that information.
For example, if the user enters:

What is the address of our headquarters? Please return:

*CLARIFY [I'm sorry but I don't have access to that information. I
only have access to information in our Calendars and Contacts databases.
If the information you are looking for is not in one of those two
databases, then I don’t have access to it.]

If you cannot answer the user's question, please try to guide the user
to a question that you can answer using the sources you have access to.
    """

    user_request: str = dspy.InputField()
    needs: list[Literal["CONTACT_READER", "CALENDAR_READER", "CLARIFY", "EMAIL_MANAGER"]] = dspy.OutputField()
    
class AIAssistantAgent(dspy.Signature):
    """
    You are a personal assisant helping to fullifll users requests.
    
    """

    user_request: str = dspy.InputField()
    user_context: UserContext = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
                "Message that summarizes the process result, and the information users need"
            )
        )

class AIAssistantApp(dspy.Module):
    def __init__(self):
        super().__init__()
        self.router = dspy.ChainOfThought(Router)
        self.calendar = CalendarManagerApp()
        self.contacts = ContactsManagerApp()

        
    @observe()
    def forward(self, message: str):
        router_result = self.router(user_request = message)

        return router_result

