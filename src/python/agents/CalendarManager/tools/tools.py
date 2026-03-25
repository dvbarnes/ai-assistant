

from datetime import date, datetime
from deepeval.tracing import observe
from pydantic import BaseModel

import logging

class Date(BaseModel):
    start_time: datetime
    end_time: datetime
    free: bool



format_string = "%Y-%m-%d %H:%M:%S"

calendar_database = [
    Date(
        start_time=datetime.strptime("2026-03-25 09:30:00", format_string),
        end_time=datetime.strptime("2026-03-25 11:30:00", format_string),
        free = True
    ),
    Date(
        start_time=datetime.strptime("2026-03-26 09:30:00", format_string),
        end_time=datetime.strptime("2026-03-26 11:30:00", format_string),
        free = True
        ),
    Date(
        start_time=datetime.strptime("2026-03-27 09:30:00", format_string),
        end_time=datetime.strptime("2026-03-27 11:30:00", format_string),
        free = True
        )
    
]
@observe()
def get_current_date()->date:
    print(f'get_current_date {datetime.datetime.now()}')
    return datetime.datetime.now()

@observe()
def get_availability(start_date: date, end_date: date)-> list[Date]:
    print(f'get_availability {start_date} {end_date}')
    return calendar_database

@observe()
def book_meeting(start_date: datetime, end_date: datetime, recipients: list[str], meeting_title: str, agenda: str):
    print(f"booking meeting for {start_date} {end_date} to: {'; '.join(recipients)} {meeting_title} {agenda}")

@observe()
def send_email(recipients: list[str], email_title: str, email_body: str):
    print(f"to: {'; '.join(recipients)}")
    print(f"{email_title}")
    print("--------------")
    print(f"{email_body}")

@observe()
def send_need_help():
    print("Help!!")
