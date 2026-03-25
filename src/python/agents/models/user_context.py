from pydantic import BaseModel


class UserContext(BaseModel):
    first_name: str
    last_name: str
    email: str