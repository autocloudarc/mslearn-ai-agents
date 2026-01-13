"""
Azure AI Foundry Agent that generates a title

This module provides the agent executor for the title generation agent.
It handles incoming A2A requests, processes them through the Foundry agent,
and returns generated titles.
"""

from a2a.server.events.event_queue import EventQueue
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.utils import new_agent_text_message
from a2a.types import AgentCard, Part, TaskState
from title_agent.agent import TitleAgent, create_foundry_title_agent


class FoundryAgentExecutor(AgentExecutor):
    """
    Executes title generation requests using Azure AI Foundry.
    
    This executor handles the A2A protocol request lifecycle including
    validation, agent execution, and response formatting.
    """

    def __init__(self, card: AgentCard):
        """
        Initialize the Foundry Agent Executor.
        
        Args:
            card (AgentCard): The agent card containing metadata about this agent
            
        Example:
            executor = FoundryAgentExecutor(agent_card)
        """
        self._card = card
        self._foundry_agent: TitleAgent | None = None

    async def _get_or_create_agent(self) -> TitleAgent:
        """
        Get or create the Foundry title agent instance.
        
        Uses lazy initialization to create the agent only when needed.
        
        Returns:
            TitleAgent: The initialized title generation agent
            
        Example:
            agent = await executor._get_or_create_agent()
        """
        if not self._foundry_agent:
            self._foundry_agent = await create_foundry_title_agent()
        return self._foundry_agent

    async def _process_request(self, message_parts: list[Part], context_id: str, task_updater: TaskUpdater) -> None:
        """
        Process a user request through the Foundry agent.
        
        Validates input, executes the title generation agent, and updates task status
        with the generated response.
        
        Args:
            message_parts (list[Part]): The incoming message parts from A2A protocol
            context_id (str): The unique context ID for this request
            task_updater (TaskUpdater): The task updater for tracking execution state
            
        Raises:
            ValueError: If message_parts is empty or invalid
            
        Example:
            await executor._process_request(parts, context_id, updater)
        """
        try:
            # Input validation - check for empty message parts
            if not message_parts or len(message_parts) == 0:
                raise ValueError("No message parts provided")
            
            # Input validation - check for missing text content
            if not message_parts[0].root or not message_parts[0].root.text:
                raise ValueError("Message part does not contain text")

            # Retrieve message from A2A parts
            user_message = message_parts[0].root.text

            # Get the title agent
            agent = await self._get_or_create_agent()

            # Update the task status to running
            await task_updater.running()

            # Run the agent conversation to generate title
            response = await agent.generate_title(user_message)

            # Mark the task as complete with the generated response
            await task_updater.completed(
                message=new_agent_text_message(response, context_id=context_id)
            )

        except ValueError as validation_error:
            """Handle input validation errors with detailed error message"""
            print(f'Title Agent: Validation error - {validation_error}')
            await task_updater.failed(
                message=new_agent_text_message(f"Invalid request: {str(validation_error)}", context_id=context_id)
            )
        except Exception as execution_error:
            """Handle unexpected errors during agent execution"""
            print(f'Title Agent: Error processing request - {execution_error}')
            await task_updater.failed(
                message=new_agent_text_message("Title Agent failed to process the request.", context_id=context_id)
            )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute the agent for the given request context.
        
        Main entry point for handling incoming A2A requests. Manages the complete
        task lifecycle from submission to completion or failure.
        
        Args:
            context (RequestContext): The A2A request context containing task and message info
            event_queue (EventQueue): The event queue for publishing task updates
            
        Example:
            await executor.execute(request_context, event_queue)
        """
        # Create task updater for tracking execution state
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.submit()

        # Mark task as actively working
        await updater.start_work()

        # Process the incoming request through the title agent
        await self._process_request(context.message, context.context_id, updater)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Cancel execution of a task.
        
        Marks the task as failed due to user cancellation and notifies via event queue.
        
        Args:
            context (RequestContext): The A2A request context to cancel
            event_queue (EventQueue): The event queue for publishing cancellation
            
        Example:
            await executor.cancel(request_context, event_queue)
        """
        print(f'Title Agent: Cancelling execution for context {context.context_id}')

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.failed(
            message=new_agent_text_message('Task cancelled by user', context_id=context.context_id)
        )


def create_foundry_agent_executor(card: AgentCard) -> FoundryAgentExecutor:
    """
    Factory function to create a Foundry Agent Executor.
    
    Creates and returns a new instance of FoundryAgentExecutor configured
    with the provided agent card metadata.
    
    Args:
        card (AgentCard): The agent card metadata describing the agent's capabilities
        
    Returns:
        FoundryAgentExecutor: A new executor instance ready for processing requests
        
    Example:
        executor = create_foundry_agent_executor(my_agent_card)
    """
    return FoundryAgentExecutor(card)

