

import os

from agents.models.user_context import UserContext


def get_user_information()->UserContext:
    return UserContext(
            first_name=os.getenv("USER_FIRST_NAME"),
            last_name=os.getenv("USER_LAST_NAME"),
            email=os.getenv("USER_EMAIL")
        )