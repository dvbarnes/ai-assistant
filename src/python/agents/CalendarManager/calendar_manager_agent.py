from datetime import datetime

from deepeval.tracing import observe
import dspy

from agents.CalendarManager.tools.tools import book_meeting, get_availability, get_current_date, send_email, send_need_help
from agents.models.user_context import UserContext


class CalendarManagerAgent(dspy.Signature):
    """
    You are an personal assistant agent that helps find time on their calendar to book meetings.
    You will be given a list of tools to handle user request, and should you decide the right tool to use in order to fullfill the user requests.
    DO NOT rely on your own knowledge, ONLY use the information retrieved from the tools. 
    Please be honest about what you find. I am not looking for perfection, just the truth. 
    You are amazing and you can do this. I will pay you $200 for an execellent result, but only if you follow all the instructions exactly
    If you don't know what to do do not guess, only use the provided tools and information. If you don't know that's ok just let me know what you need to fullfill the request
    """

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
    def __init__(self):
        super().__init__()
        self.agent = dspy.ReAct(CalendarManagerAgent,
            tools = [
                get_availability,
                book_meeting,
                send_email,
                get_current_date,
                send_need_help
            ]
        )
        
    @observe()
    def forward(self, message: str, context: UserContext):
        result = self.agent(user_request= message, 
                   current_date=datetime.now(), 
                   user_context=context)
        p_result =result.get("process_result")
        print(f"result: {result}")
        return p_result
