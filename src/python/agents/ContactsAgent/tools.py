

from typing import Optional

import dspy
from pydantic import BaseModel


class Contact(BaseModel):
    first_name: str
    last_name: str
    email: str
    company: str

# class LookUpUser(dspy.Tool):
#     def __init__(self):
#         super.__init__(func=self.lookup_user)
#         self._parse_function(func=self.lookup_user, arg_desc=None)

def lookup_user(first_name: Optional[str], last_name: Optional[str], email: Optional[str], company: Optional[str])->list[Contact]:
    if(first_name == "John" or first_name == "john"):
        return [Contact(
            first_name= first_name if first_name!= None else "john",
            last_name = last_name if last_name!= None else "Doe",
            email = email if email!= None else "john_doe@your_company.com",
            company = company if company!= None else "your_company",
        )]
    if( first_name == "Jack" or first_name == "jack"):
        ret = []
        ret.append(
            Contact(
                first_name= first_name if first_name!= None else "Jack",
                last_name = last_name if last_name!= None else "Doe",
                email = email if email!= None else "Jack_doe@your_company.com",
                company = "your_company",
            )
        )
        ret.append(
            Contact(first_name= first_name if first_name!= None else "Jack",
                last_name = last_name if last_name!= None else "Doe",
                email = email if email!= None else "Jack_doe@your_company.com",
                company = "Microsoft"
            )
        )
        return list(ret)

    return []