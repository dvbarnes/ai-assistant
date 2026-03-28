from datetime import datetime

from deepeval.tracing import observe
import dspy
from pydantic import BaseModel

from agents.CalendarManager.tools.tools import book_meeting, get_availability, get_current_date, send_email, send_need_help
from agents.models.user_context import UserContext


class CalendarAgentResponse(BaseModel):
    response: str
    tools: list[str]
class CalendarManagerAgent(dspy.Signature):
    """
    You are a calendar assistant agent. Your ONLY job is to help schedule meetings 
using the provided tools.

STRICT RULES — follow these exactly, no exceptions:

1. REQUIRED FIELDS: Before calling any tool, you must have ALL of the following 
   from the user's message:
   - Attendee email address (exact format: name@domain.com)
   - Meeting duration
   - Preferred date or date range

2. IF ANY REQUIRED FIELD IS MISSING: Stop immediately. Do not call any tool. 
   Do not guess, infer, or assume any value. Respond ONLY with:
   "I'm missing the following required information: [list exactly what is missing]. 
   Please provide these before I can continue."

3. NEVER invent, assume, or infer an email address under any circumstances — 
   not from a name, a company, or context clues.

4. ONLY use data explicitly provided by the user or returned by a tool. 
   Do not use your own knowledge.

5. If a tool returns no results or an error, report that result exactly. 
   Do not substitute your own answer."""

    user_request: str = dspy.InputField()
    current_date: datetime= dspy.InputField()
    user_context: UserContext = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
                "Message that summarizes the process result, and the information users need, e.g., the "
                "confirmation_number if a new flight is booked."
            )
        )

class CalendarManagerApp(dspy.Module):
    def __init__(self, tools=[
                get_availability,
                book_meeting,
                send_email,
                get_current_date,
                send_need_help
            ]):
        
        super().__init__()
        self.agent = dspy.ReAct(CalendarManagerAgent,
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
        print(f"result: {result}")
        return CalendarAgentResponse(
            response=p_result,
            tools=tools
        )
