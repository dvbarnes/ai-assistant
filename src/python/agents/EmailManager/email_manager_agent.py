import dspy
from pydantic import BaseModel

from agents.EmailManager.tools import read_email, send_email
from agents.models.user_context import UserContext


class EmailAgentResponse(BaseModel):
    response: str
    tools: list[str]


class EmailManagerAgent(dspy.Signature):
    """
    You are an email manager agent tasked with managing a users email.
    When sending emails please follow the following rules:

    EMAIL STYLE GUIDELINES:

- Default to a concise, professional, and friendly tone.
- Prefer simple and direct language over formal or overly polished phrasing.
- Avoid overly enthusiastic or sales-like language.

SIGN-OFF STYLE:

- Prefer closing emails with:

  Thanks,
  {my_name}

- Use this as the default sign-off unless the context clearly calls for a different tone (e.g., very formal or apologetic situations).

- Do not vary the sign-off unnecessarily (avoid "Best", "Sincerely", etc. unless context strongly justifies it).
    """

    user_request: str = dspy.InputField()
    user_context: UserContext = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
                "Message that summarizes the process result, and the information users need"
            )
        )
class EmailManagerApp(dspy.Module):

    def __init__(self, tools = [send_email, read_email]):
        super().__init__()
        self.agent = dspy.ReAct(EmailManagerAgent,
            tools = tools
        )
        
    def forward(self, message: str, context: UserContext):
        result = self.agent(user_request= message, 
                   user_context=context)
        p_result =result.get("process_result")
        trajectory: dict[str,str]= result.get("trajectory")
        tools = [value for (key,value) in trajectory.items() if key.startswith("tool_name") ]
        return EmailAgentResponse(
            response=p_result,
            tools=tools
        )