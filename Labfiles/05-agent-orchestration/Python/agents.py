# Add references
# asyncio: enables asynchronous programming to handle concurrent operations
import asyncio
import os
# cast: type hints utility for casting objects to specific types at runtime
from typing import cast
# ChatMessage: represents individual messages in agent communication
# Role: enum defining message roles (user, assistant, etc.)
# SequentialBuilder: orchestrates agents to run in a specific order
# WorkflowOutputEvent: event fired when workflow produces output data
from agent_framework import ChatMessage, Role, SequentialBuilder, WorkflowOutputEvent
# AzureAIAgentClient: client for interacting with Azure AI Agent Service
from agent_framework.azure import AzureAIAgentClient
# AzureCliCredential: authentication using Azure CLI credentials for Azure authentication
from azure.identity import AzureCliCredential
# Load environment variables from .env file
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def main():
    # Main asynchronous function that orchestrates the multi-agent workflow
    # This function coordinates three specialized agents to process customer feedback
    
    # Define instructions for the summarizer agent
    # The summarizer reduces raw customer feedback into a single, concise sentence
    # This condensed format makes feedback easier to process in downstream systems
    summarizer_instructions="""
    Summarize the customer's feedback in one short sentence. Keep it neutral and concise.
    Example output:
    App crashes during photo upload.
    User praises dark mode feature.
    """

    # Define instructions for the classifier agent
    # The classifier categorizes feedback into one of three predefined categories
    # This categorization enables targeted routing and prioritization of feedback
    classifier_instructions="""
    Classify the feedback as one of the following: Positive, Negative, or Feature request.
    """

    # Define instructions for the action recommendation agent
    # The action agent suggests the next step based on the summary and classification
    # Recommended actions guide what team should handle the feedback and how
    action_instructions="""
    Based on the summary and classification, suggest the next action in one short sentence.
    Example output:
    Escalate as a high-priority bug for the mobile team.
    Log as positive feedback to share with design and marketing.
    Log as enhancement request for product backlog.
    """

    # Create the chat client
    # Instantiate Azure CLI credentials to authenticate with Azure services
    # This allows the app to securely access the Azure AI Agent Service
    credential = AzureCliCredential()
    
    # Get project endpoint and deployment name from environment variables
    project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    deployment_name = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")
    
    if not project_endpoint or not deployment_name:
        print("Error: AZURE_AI_PROJECT_ENDPOINT and AZURE_AI_MODEL_DEPLOYMENT_NAME must be set in .env")
        return
    
    # Use async context manager to ensure proper resource cleanup after execution
    # AzureAIAgentClient automatically loads settings from the .env configuration file
    async with (
        AzureAIAgentClient(
            credential=credential,
            ai_project_endpoint=project_endpoint,
            ai_model_deployment_name=deployment_name
        ) as chat_client,
    ):
        # Create agents
        # Instantiate the summarizer agent with its specific instructions
        # Each agent is a distinct entity with its own behavior and role in the workflow
        summarizer = chat_client.create_agent(
            instructions=summarizer_instructions,
            name="summarizer",
        )

        # Instantiate the classifier agent with its specific instructions
        # The classifier categorizes the summarized feedback for routing
        classifier = chat_client.create_agent(
            instructions=classifier_instructions,
            name="classifier",
        )

        # Instantiate the action recommendation agent with its specific instructions
        # This agent determines the next action to take based on prior analysis
        action = chat_client.create_agent(
            instructions=action_instructions,
            name="action",
        )

        # Initialize the current feedback
        # Sample customer feedback that will be processed through the agent pipeline
        # This feedback describes a positive customer support experience
        feedback="""
        I reached out to your customer support yesterday because I couldn't access my account. 
        The representative responded almost immediately, was polite and professional, and fixed the issue within minutes. 
        Honestly, it was one of the best support experiences I've ever had.
        """

        # Build sequential orchestration
        # Create a workflow where agents process feedback in a predefined order
        # Each agent receives the output from previous agents for further analysis
        # SequentialBuilder ensures agents execute sequentially, not in parallel
        workflow = SequentialBuilder().participants([summarizer, classifier, action]).build()

        # Run and collect outputs
        # Initialize list to store all output messages from the workflow execution
        # Each agent's response will be appended as the workflow progresses
        outputs: list[list[ChatMessage]] = []
        # Run the workflow asynchronously with the customer feedback as input
        # The run_stream() method yields events as the workflow processes data
        # Iterate through all events emitted during workflow execution
        async for event in workflow.run_stream(f"Customer feedback: {feedback}"):
            # Check if the event is a WorkflowOutputEvent (contains agent output data)
            # WorkflowOutputEvent fires when an agent completes processing and produces output
            if isinstance(event, WorkflowOutputEvent):
                # Cast the event data to a list of ChatMessage objects for type safety
                # Append the messages to the outputs list for later retrieval and display
                outputs.append(cast(list[ChatMessage], event.data))

        # Display outputs
        # Check if the workflow produced any output messages to display
        # This ensures we don't attempt to print empty results
        if outputs:
            # Iterate through the messages in the final workflow output
            # outputs[-1] contains the last (final) set of messages from the workflow
            # enumerate() provides both the index and message for formatting
            for i, msg in enumerate(outputs[-1], start=1):
                # Extract the author name from the message, defaulting to agent role
                # If no author_name is set, use "assistant" for AI agents or "user" for users
                name = msg.author_name or ("assistant" if msg.role == Role.ASSISTANT else "user")
                # Format and print the message with a visual separator and sequential numbering
                # The output clearly shows who said what and in what order
                print(f"{'-' * 60}\n{i:02d} [{name}]\n{msg.text}")

if __name__ == "__main__":
    # Entry point check ensures code only runs when script is executed directly, not imported
    # asyncio.run() creates and manages the event loop for async operations
    # This launches the main() coroutine and waits for it to complete execution
    asyncio.run(main())