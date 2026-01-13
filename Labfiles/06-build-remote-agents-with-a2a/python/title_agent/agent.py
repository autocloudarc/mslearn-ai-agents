""" 
Azure AI Foundry Agent that generates a title

This module defines a TitleAgent class that interacts with Azure AI Foundry agents.
The agent is specialized in generating titles based on user input and maintains
state throughout a conversation session using threads.
"""

import os
from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import Agent, ListSortOrder, MessageRole

class TitleAgent:
    """
    A wrapper class for managing Azure AI Foundry agents that generate titles.
    
    This class handles:
    - Agent initialization and configuration
    - Thread management for maintaining conversation state
    - Message sending and receiving with the Azure AI agent
    - Error handling for failed runs
    
    Attributes:
        client (AgentsClient): The Azure AI agents client for API communication
        agent (Agent | None): The created title agent instance (lazy-loaded)
    """

    def __init__(self):
        """
        Initialize the TitleAgent instance.
        
        Loads configuration from environment variables and creates an AgentsClient
        connection to Azure AI Foundry using default credentials.
        """
        
        # Create the agents client
        # DefaultAzureCredential uses Azure CLI authentication or managed identity
        # This connects to Azure AI Foundry endpoint specified in environment
        self.client = AgentsClient(
            endpoint=os.getenv("AGENT_ENDPOINT"),
            credential=DefaultAzureCredential()
        )
        
        # Initialize agent as None - it will be created on first use (lazy loading)
        # This pattern avoids unnecessary API calls if the agent isn't needed
        self.agent: Agent | None = None

    async def create_agent(self) -> Agent:
        """
        Create or retrieve the title agent instance.
        
        This method implements lazy loading - the agent is only created once and
        cached for subsequent calls. This improves performance by avoiding duplicate
        agent creation calls to Azure AI Foundry.
        
        Returns:
            Agent: The created or cached title agent instance
            
        Example:
            agent = await title_agent.create_agent()
        """
        # Check if agent already exists in cache
        # If so, return the cached instance to avoid redundant API calls
        if self.agent:
            return self.agent

        # Create the title agent
        # Configure a new agent with specific instructions for title generation
        # The model deployment name is loaded from environment variables
        self.agent = self.client.create_agent(
            model=os.getenv("MODEL_DEPLOYMENT_NAME"),
            name="title-agent",
            instructions="You are a helpful AI assistant that generates creative and concise titles. "
                        "Based on user input or content, generate an appropriate title that captures "
                        "the essence of the topic. Keep titles brief, engaging, and professional."
        )

        # Return the newly created agent instance
        return self.agent
        
    async def run_conversation(self, user_message: str) -> list[str]:
        """
        Send a user message to the agent and retrieve its response.
        
        This method manages the complete conversation flow:
        1. Ensures the agent is initialized
        2. Creates a new conversation thread for isolation
        3. Sends the user message to the agent
        4. Processes the agent's response
        5. Returns the response or error message
        
        Args:
            user_message (str): The message to send to the title agent for processing
            
        Returns:
            list[str]: A list containing the agent's response text(s). Returns error
                      messages if the run fails or no response is received.
                      
        Example:
            responses = await title_agent.run_conversation("Generate a title for a blog about AI")
        """
        
        # Add a message to the thread, process it, and retrieve the response
        # First, ensure the agent is initialized before attempting to use it
        if not self.agent:
            await self.create_agent()

        # Create a thread for the chat session
        # Threads maintain conversation state and allow multiple turns in a conversation
        # Each thread is independent, allowing parallel conversations
        thread = self.client.threads.create()
        
        # Send user message
        # Add the user's message to the thread as the starting point for the agent
        # The message role indicates this is input from the user
        self.client.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content=user_message
        )

        # Create and run the agent
        # This processes the message through the agent and executes any tools if needed
        # create_and_process waits for completion before returning
        run = self.client.runs.create_and_process(
            thread_id=thread.id,
            agent_id=self.agent.id
        )

        # Check if the agent run completed successfully
        # Failures can occur due to API issues, timeout, or model errors
        if run.status == 'failed':
            print(f'Title Agent: Run failed - {run.last_error}')
            return [f'Error: {run.last_error}']

        # Get response messages
        # Retrieve all messages from the thread, sorted in descending order (newest first)
        # This ensures we get the most recent agent response
        messages = self.client.messages.list(thread_id=thread.id, order=ListSortOrder.DESCENDING)
        
        # Process messages to extract the agent's response
        responses = []
        # Iterate through messages looking for the agent's latest response
        for msg in messages:
            # Only get the latest assistant response
            # Filter to only AGENT role messages that contain text content
            # The agent may produce multiple response types; we only care about text
            if msg.role == MessageRole.AGENT and msg.text_messages:
                # Extract the text value from each message
                # Multiple text messages can exist but typically there's one primary response
                for text_msg in msg.text_messages:
                    responses.append(text_msg.text.value)
                # Break after first agent message to get only the latest response
                # Messages are sorted newest first, so first match is the most recent
                break 

        # Return the collected responses, or a default message if none found
        # This ensures the caller always gets a response (success or default message)
        return responses if responses else ['No response received']


async def create_foundry_title_agent() -> TitleAgent:
    """
    Factory function to create and initialize a fully configured TitleAgent.
    
    This function provides a convenient way to instantiate a TitleAgent that is
    ready to use for generating titles. It handles the initialization sequence
    automatically.
    
    Returns:
        TitleAgent: A fully initialized TitleAgent instance with the agent created
                   and ready to process conversations
                   
    Example:
        title_agent = await create_foundry_title_agent()
        responses = await title_agent.run_conversation("Generate a title for a tech article")
    """
    # Create a new TitleAgent instance
    # This calls __init__ which sets up the Azure client connection
    agent = TitleAgent()
    
    # Initialize the agent within the TitleAgent instance
    # This creates the actual agent in Azure AI Foundry
    await agent.create_agent()
    
    # Return the fully initialized agent ready for use
    return agent
