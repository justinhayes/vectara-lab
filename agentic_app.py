import sys
from vectara_agentic.tools import VectaraToolFactory, ToolsFactory
from vectara_agentic.agent import Agent
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.types import AgentType, ModelProvider
from pydantic import Field, BaseModel


vec_factory = VectaraToolFactory(
    vectara_api_key='<ENTER_VECTARA_PERSONAL_API_KEY>',
    vectara_corpus_key='shakespeare'
)


valid_plays = ["Taming of the Shrew", "Shrew"]


######## Tool to query against all meeting contents
class QueryShakespeareArgs(BaseModel):
    query: str = Field(..., description="The user query.")


query_shakespeare = vec_factory.create_rag_tool(
    tool_name="query_shakespeare",
    tool_description="Query the content from Shakespeare plays",
    tool_args_schema=QueryShakespeareArgs,
    reranker="multilingual_reranker_v1", rerank_k=100,
    n_sentences_before=2, n_sentences_after=2, lambda_val=0.005,
    summary_num_results=15,
    vectara_summarizer='mockingbird-1.0-2024-07-16',
    include_citations=True,
    verbose=True
)


def validate_play(
        play_name: str = Field(description="The name of a Shakespeare play")
) -> bool:
    for valid_play in valid_plays:
        if valid_play.lower() in play_name.lower():
            return True
    return False


# The list of all tools available to the agent
def create_assistant_tools():
    tools_factory = ToolsFactory()
    return [query_shakespeare]+ [tools_factory.create_tool(validate_play)]


# Function to execute a single query. This gives the agent instructions for how to call the different tools.
def run_query(query: str):
    topic_of_expertise = "Shakespeare plays"

    agent_instructions = f"""
        - You are a helpful assistant in conversation with a user who wants to know about Shakespeare plays. 
        - You can answer questions, provide insights, or summarize any information from Shakespeare's plays.
        - A user may ask about the plots, characters, and themes related to Skahespeare's plays.
        - Use the validate_play tool to ensure that the query is about a valid play, and if not then respond saying that you do not know about that play.
        """

    private_llm_config = AgentConfig(
        agent_type=AgentType.REACT,
        main_llm_provider=ModelProvider.PRIVATE,
        main_llm_model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        private_llm_api_base="https://transact-test.itg.ti.com/v1/llm",
        private_llm_api_key="ENTER_LLAMA_API_KEY"
    )

    agent = Agent(
        #agent_config=private_llm_config,
        tools=create_assistant_tools(),
        topic=topic_of_expertise,
        custom_instructions=agent_instructions,
        verbose=True
    )
    agent.report()

    response = agent.chat(query)
    print(f"\n\n>>{query}")
    print(f"\n>>{str(response)}\n")


# Run the query from the command line
run_query(sys.argv[1])