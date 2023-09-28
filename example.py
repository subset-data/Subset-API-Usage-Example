import requests
from dotenv import load_dotenv

load_dotenv()
from os import environ

from typing import List
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import SystemMessageChunk
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

SUBSET_HOST = environ.get("SUBSET_HOST")
SUBSET_API_KEY = environ.get("SUBSET_API_KEY")
OPENAI_API_KEY = environ.get("OPENAI_API_KEY")


class SubsetApi:
    def __init__(self, host: str, api_key: str, port: str = ""):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.headers = {
            "Accept": "*/*",
            "Authorization": f"Bearer {self.api_key}",
        }

    def get(self, path: str, params: dict = {}) -> requests.Response:
        return requests.get(
            self.host + path,
            headers=self.headers,
            params=params,
            verify=False,
        )

    def post(self, path: str, params: dict = {}) -> requests.Response:
        return requests.post(
            self.host + path,
            headers=self.headers,
            json=params,
            verify=False,
        )


def analytics_tool(
    measures: List[str] = [],
    breakdown_fields: List[str] = [],
    subset_description: str = "",
    sorts: List[list] = [[], []],
    limit: int = 10,
):
    subset_api = SubsetApi(SUBSET_HOST, SUBSET_API_KEY, "3001")
    body = {
        "measures": measures,
        "subset_description": subset_description,
        "breakdown_fields": breakdown_fields,
        "sorts": sorts,
        "limit": limit,
    }

    response = subset_api.post("api/v1/query", params=body)
    if response.status_code != 200:
        return {"error": "please rephrase your description and try again"}
    else:
        final = response.json()
        frontend_url = (
            str(SUBSET_HOST).replace("3001", "3000").replace("http", "https")
            + "data/"
            + final.get("slug", "")
        )
        print(f"query url: {frontend_url}")
        return final


class analyticsToolSchema(BaseModel):
    """Input for the subset_analytics_tool"""

    measures: List[str] = Field(
        ...,
        description=(
            "a list of measures from the prompt to aggregate, e.g. ['count', 'sum', 'average']"
        ),
    )
    breakdown_fields: List[str] = Field(
        ...,
        description=(
            "a list of fields from the prompt to group by, e.g. ['user_details.state', 'users_details.state']"
        ),
    )
    sorts: List[list] = Field(
        ...,
        description=(
            "a list of lists of fields from the prompt to sort by, e.g. [['count', 'desc'], ['user_details.city', 'desc']]"
        ),
    )
    limit: int = Field(
        ...,
        description=("the number of results to return, e.g. 10"),
    )
    subset_description: str = Field(
        description=(
            "a single sentence english statement describing entities in your database."
            " Do not pose a question or issue a command, rephrase it as a single sentence descriptive statement"
            "example:'users who have not logged in in the last 30 days'"
        ),
    )


subset_analytics_tool = StructuredTool.from_function(
    analytics_tool,
    name="subset_analytics_tool",
    description=(
        "allows you to answer analytical questions about entities in your database, "
        "using a combination of parameters and and a natural language statement describing the subject set "
        "for the subset_description, you should only use verbs, adjectives and values from your system prompt"
    ),
    args_schema=analyticsToolSchema,
    handle_tool_error=True,
)


if __name__ == "__main__":
    user_satisfied = False
    while not user_satisfied:
        user_prompt = input("What would you like to know? ")
        api = SubsetApi(SUBSET_HOST, SUBSET_API_KEY, "3001")
        response = api.get("api/v1/query/rag", params={"q": user_prompt})
        rag = response.json().get("text", "")
        prompt = (
            "in tools, you may only use verbs, adjectives and measures in the following list:"
            f"{rag}"
        )
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-4",
            openai_api_key=OPENAI_API_KEY,
        )
        tools = [subset_analytics_tool]
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            agent_kwargs={"system_message": SystemMessageChunk(content=prompt)},
            verbose=True,
            debug=True,
        )
        response = agent.run(user_prompt)
        print(response)
        user_satisfied = input("would you like to ask another question? (y/n) ") == "y"
