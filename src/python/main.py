
import dspy
import os

from agents.CalendarManager.calendar_manager_agent import CalendarManagerApp
from agents.models.user_context import UserContext

dspy.configure(
    lm=dspy.LM(
        os.getenv("MODEL_NAME"), 
        api_key=os.getenv("OPEN_API_KEY"), 
        api_base=os.getenv("OPEN_API_URL")
    )
)

def main():
    agent = CalendarManagerApp()
    result = agent(
        message="John asked me to find some time this week to meet, please email him 3 times that I'm free his email is john@john.com",
        context=UserContext(
            first_name=os.getenv("USER_FIRST_NAME"),
            last_name=os.getenv("USER_LAST_NAME"),
            email=os.getenv("USER_EMAIL")
        )
          )
    print(result)

if __name__ == "__main__":
    main()