from datetime import datetime

from deepeval.tracing import observe
import dspy
from pydantic import BaseModel

from agents.ContactsAgent.tools import lookup_user
from agents.models.user_context import UserContext


class ContactsAgentResponse(BaseModel):
    response: str
    tools: list[str]

class ContactsManagerAgent(dspy.Signature):
    """
    You are a contacts assistant agent. Your ONLY job is to help lookup contacts
using the provided tools.

STRICT RULES — follow these exactly, no exceptions:


1. NEVER invent, assume, or infer an email address under any circumstances — 
   not from a name, a company, or context clues.

2. ONLY use data explicitly provided by the user or returned by a tool. 
   Do not use your own knowledge.

3. If a tool returns no results or an error, report that result exactly. 
   Do not substitute your own answer."""

    user_request: str = dspy.InputField()
    current_date: datetime= dspy.InputField()
    user_context: UserContext = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
                "Message that summarizes the process result, and the information users need"
            )
        )

class ContactsManagerApp(dspy.Module):
    def __init__(self, tools = {
        "lookup_user": dspy.Tool(lookup_user)
    }):
        super().__init__()
        self.agent = dspy.ReAct(ContactsManagerAgent,
            tools = tools
        )
        
    @observe()
    def forward(self, message: str, context: UserContext):
        result = self.agent(user_request= message, 
                   current_date=datetime.now(), 
                   user_context=context)
        p_result =result.get("process_result")
        trajectory: dict[str,str]= result.get("trajectory")
        tools = [value for (key,value) in trajectory.items() if key.startswith("tool_name") ]
        return ContactsAgentResponse(
            response=p_result,
            tools=tools
        )
