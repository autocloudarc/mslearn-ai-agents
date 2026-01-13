"""
AI Foundry Title Agent Server

This module creates and runs an A2A (Agent-to-Agent) protocol-compliant server
for the Title Agent. The server enables remote agents to discover and communicate
with the title generation capabilities through the A2A protocol.

The Title Agent specializes in generating catchy, engaging blog titles based on
article topics. It's powered by Azure AI Foundry and can be discovered and called
by other agents in a distributed agent architecture.

Architecture:
- A2A Protocol: Enables standardized communication between remote agents
- Uvicorn: ASGI server that hosts the application
- Starlette: Lightweight web framework for handling HTTP requests
- InMemoryTaskStore: Tracks task execution state (execution, pending, results)
"""

import os
import uvicorn

# A2A (Agent-to-Agent) Protocol Imports
# These imports provide the framework for agent-to-agent communication
from a2a.server.apps import A2AStarletteApplication  # Main A2A app wrapper for Starlette
from a2a.server.request_handlers import DefaultRequestHandler  # Handles incoming A2A requests
from a2a.server.tasks import InMemoryTaskStore  # In-memory storage for task execution tracking

# A2A Type Definitions
# These define the structures for agent metadata and capabilities
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

# Environment Configuration
from dotenv import load_dotenv  # Load environment variables from .env file

# Starlette Web Framework Imports
# Starlette is a lightweight ASGI framework for building web applications
from starlette.applications import Starlette  # Main Starlette application
from starlette.requests import Request  # HTTP request object
from starlette.responses import PlainTextResponse  # Simple text HTTP response
from starlette.routing import Route  # Route definition for URL endpoints

# Agent Executor Import
# This creates the actual agent that processes title generation requests
from title_agent.agent_executor import create_foundry_agent_executor

# Load environment variables from .env file
# This sets up configuration like SERVER_URL and TITLE_AGENT_PORT
load_dotenv()

# ============================================================================
# Configuration Variables
# ============================================================================
# Load server configuration from environment variables
# These variables define where the server will run and what port it will use
host = os.environ["SERVER_URL"]
port = os.environ["TITLE_AGENT_PORT"]

# ============================================================================
# Define Agent Skills
# ============================================================================
# Skills define the capabilities that the agent can perform and discover
# Each skill is an operation that other agents can request this agent to perform
# 
# Attributes:
#   - id: Unique identifier for the skill (used in requests)
#   - name: Human-readable name of the skill
#   - description: What the skill does
#   - tags: Keywords for categorizing the skill
#   - examples: Example requests that can trigger this skill
skills = [
    AgentSkill(
        id='generate_blog_title',
        name='Generate Blog Title',
        description='Generates a blog title based on a topic',
        tags=['title'],
        examples=[
            'Can you give me a title for this article?',
        ],
    ),
]

# ============================================================================
# Create Agent Card
# ============================================================================
# The agent card is the metadata that describes this agent to other agents
# in the network. This is shared via the A2A protocol so other agents can
# discover and communicate with this agent.
#
# Attributes:
#   - name: Display name of the agent
#   - description: What the agent does
#   - url: The endpoint where the agent can be accessed
#   - version: Semantic version of the agent
#   - default_input_modes: How the agent receives input (text, etc.)
#   - default_output_modes: How the agent returns output (text, etc.)
#   - capabilities: Specific capabilities of the agent
#   - skills: List of skills the agent can perform
agent_card = AgentCard(
    name='AI Foundry Title Agent',
    description='An intelligent title generator agent powered by Foundry. '
    'I can help you generate catchy titles for your articles.',
    url=f'http://{host}:{port}/',
    version='1.0.0',
    default_input_modes=['text'],
    default_output_modes=['text'],
    capabilities=AgentCapabilities(),
    skills=skills,
)

# ============================================================================
# Create Agent Executor
# ============================================================================
# The agent executor is responsible for processing incoming requests and
# executing the agent's logic. It's created from the agent_executor module
# which contains the implementation of the title generation AI.
#
# The executor handles:
#   - Processing user requests
#   - Calling the Azure AI Foundry model
#   - Formatting and returning responses
agent_executor = create_foundry_agent_executor(agent_card)

# ============================================================================
# Create Request Handler
# ============================================================================
# The request handler manages incoming HTTP requests according to the A2A
# protocol. It routes requests to the agent executor and tracks task state.
#
# Components:
#   - agent_executor: Processes the actual request
#   - InMemoryTaskStore: Tracks execution status and results in memory
#     (For production, consider using a persistent store)
request_handler = DefaultRequestHandler(
    agent_executor=agent_executor, task_store=InMemoryTaskStore()
)

# ============================================================================
# Create A2A Application
# ============================================================================
# The A2AStarletteApplication wraps the agent card and request handler into
# an A2A-compliant application that can be hosted by a web framework.
#
# This application:
#   - Advertises the agent card for discovery
#   - Handles A2A protocol messages
#   - Routes requests to the request handler
#   - Manages agent-to-agent communication
a2a_app = A2AStarletteApplication(
    agent_card=agent_card, http_handler=request_handler
)

# ============================================================================
# Configure Application Routes
# ============================================================================
# Get the default routes from the A2A application
# These routes handle the A2A protocol communication endpoints
routes = a2a_app.routes()

# Add a health check endpoint for monitoring
# This allows external systems to verify the agent is running
async def health_check(request: Request) -> PlainTextResponse:
    """
    Health check endpoint that returns a simple status message.
    
    Args:
        request: The incoming HTTP request
        
    Returns:
        PlainTextResponse: A simple text response indicating the agent is running
    """
    return PlainTextResponse('AI Foundry Title Agent is running!')

routes.append(Route(path='/health', methods=['GET'], endpoint=health_check))

# ============================================================================
# Create Starlette Web Application
# ============================================================================
# Starlette is a lightweight ASGI framework that hosts our agent server
# It handles HTTP requests and routes them appropriately
#
# The app includes:
#   - A2A protocol routes (for agent communication)
#   - Health check route (for monitoring)
app = Starlette(routes=routes)

# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """
    Main entry point that starts the server.
    
    The server runs on the host and port specified in environment variables:
    - SERVER_URL: The hostname or IP where the server listens
    - TITLE_AGENT_PORT: The port number for the server
    
    Uvicorn is used as the ASGI server which provides:
    - Async request handling
    - HTTP/1.1 and HTTP/2 support
    - WebSocket support
    - Automatic worker management
    """
    # Run the server with uvicorn
    # reload=False in production; set to True for development
    uvicorn.run(app, host=host, port=port)

# ============================================================================
# Application Startup
# ============================================================================
if __name__ == '__main__':
    main()

