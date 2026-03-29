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
You are a contacts assistant agent. Your ONLY job is to retrieve contact information using the provided tools.

MANDATORY BEHAVIOR:

1. If the user asks for contact information about a person:
   - You MUST call the `search_contacts` tool
   - Extract the person's name from the request
   - Pass it as the `name` argument

2. You MUST NOT answer from memory or generate contact details yourself.

3. If the tool returns results:
   - Return the tool output exactly

4. If the tool returns no results:
   - Respond: "No contact information found for [name]."

5. NEVER skip calling the tool when contact lookup is required.

STRICT RULES:
- NEVER invent or infer an email address
- ONLY use tool outputs
   
   """

    user_request: str = dspy.InputField()
    current_date: datetime= dspy.InputField()
    user_context: UserContext = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
                "Message that summarizes the process result, and the information users need"
            )
        )

class ContactsManagerApp(dspy.Module):
    def __init__(self, tools = [lookup_user]):
        super().__init__()
        self.agent = dspy.ReAct(ContactsManagerAgent,
            tools = tools
        )
        
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
