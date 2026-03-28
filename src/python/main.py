
import dspy
import os

from agents.CalendarManager.calendar_manager_agent import CalendarManagerApp
from agents.agent import AIAssistantApp
from agents.agent_tools import get_user_information
from agents.models.user_context import UserContext

dspy.configure(
    lm=dspy.LM(
        os.getenv("MODEL_NAME"), 
        api_key=os.getenv("OPEN_API_KEY"), 
        api_base=os.getenv("OPEN_API_URL")
    )
)

def main():
    agent = AIAssistantApp()
    result = agent(
        message="John asked me to find some time this week to meet, please email him 3 times that I'm free."
        #,context= get_user_information()
          )
    print(result)

if __name__ == "__main__":
    main()