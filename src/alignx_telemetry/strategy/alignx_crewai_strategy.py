"""
AlignX CrewAI Framework Strategy

This strategy provides framework-aware instrumentation for CrewAI applications.
It integrates with CrewAI's telemetry system to add AlignX-specific context.
"""

import logging
from typing import Any, Dict, Optional
import threading

from ..providers.context import AlignXFrameworkContext
from .strategies import InstrumentationStrategy

logger = logging.getLogger(__name__)

# Optional CrewAI imports with fallbacks
try:
    from crewai.telemetry.telemetry import Telemetry as CrewAITelemetry
    from crewai.crew import Crew
    from crewai.task import Task
    from crewai.agent import Agent
    import wrapt

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    CrewAITelemetry = None
    Crew = None
    Task = None
    Agent = None
    wrapt = None


class AlignXCrewAIContext:
    """Context manager for CrewAI framework information."""

    def __init__(self):
        self.alignx_context = AlignXFrameworkContext()
        self._active_crews: Dict[str, Dict[str, Any]] = {}
        self._active_tasks: Dict[str, Dict[str, Any]] = {}
        self._active_agents: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def set_crew_context(self, crew_id: str, crew_data: Dict[str, Any]):
        """Set the current crew context."""
        with self._lock:
            self._active_crews[crew_id] = crew_data

            # Set framework context for provider instrumentation
            self.alignx_context.set_framework_context(
                framework="crewai",
                crew_id=crew_id,
                operation="crew_execution",
                **crew_data,
            )

    def set_task_context(self, task_id: str, task_data: Dict[str, Any]):
        """Set the current task context."""
        with self._lock:
            self._active_tasks[task_id] = task_data

            # Update framework context for task execution
            current_context = self.alignx_context.get_framework_context()
            if current_context:
                current_context.update(
                    {"task_id": task_id, "operation": "task_execution", **task_data}
                )
                self.alignx_context.set_framework_context(**current_context)

    def set_agent_context(self, agent_id: str, agent_data: Dict[str, Any]):
        """Set the current agent context."""
        with self._lock:
            self._active_agents[agent_id] = agent_data

            # Update framework context for agent execution
            current_context = self.alignx_context.get_framework_context()
            if current_context:
                current_context.update(
                    {"agent_id": agent_id, "operation": "agent_execution", **agent_data}
                )
                self.alignx_context.set_framework_context(**current_context)

    def clear_crew_context(self, crew_id: str):
        """Clear specific crew context."""
        with self._lock:
            self._active_crews.pop(crew_id, None)
            if not self._active_crews:
                self.alignx_context.clear_framework_context()

    def clear_task_context(self, task_id: str):
        """Clear specific task context."""
        with self._lock:
            self._active_tasks.pop(task_id, None)

    def clear_agent_context(self, agent_id: str):
        """Clear specific agent context."""
        with self._lock:
            self._active_agents.pop(agent_id, None)

    def get_current_context(self) -> Dict[str, Any]:
        """Get the current CrewAI context."""
        with self._lock:
            return {
                "framework": "crewai",
                "active_crews": list(self._active_crews.keys()),
                "active_tasks": list(self._active_tasks.keys()),
                "active_agents": list(self._active_agents.keys()),
                "crew_details": self._active_crews.copy(),
                "task_details": self._active_tasks.copy(),
                "agent_details": self._active_agents.copy(),
            }


class AlignXCrewAIStrategy(InstrumentationStrategy):
    """AlignX strategy for CrewAI framework instrumentation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.context_manager = AlignXCrewAIContext()
        self._instrumented = False
        self._original_methods: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "AlignXCrewAIStrategy"

    @property
    def framework_name(self) -> str:
        return "crewai"

    def is_available(self) -> bool:
        """Check if CrewAI is available for instrumentation."""
        return CREWAI_AVAILABLE

    def instrument(self, **kwargs) -> bool:
        """
        Instrument CrewAI to add AlignX framework context.

        This hooks into CrewAI's execution lifecycle to add framework-specific
        context to provider-level spans.
        """
        if not CREWAI_AVAILABLE:
            logger.warning("CrewAI is not installed. Skipping CrewAI instrumentation.")
            return False

        if self._instrumented:
            logger.debug("CrewAI is already instrumented.")
            return True

        try:
            self._instrument_crew_operations()
            self._instrument_task_operations()
            self._instrument_agent_operations()

            self._instrumented = True
            logger.info(
                "AlignX CrewAI framework instrumentation enabled. "
                "Framework context will be added to provider-level spans."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to instrument CrewAI: {e}")
            return False

    def uninstrument(self) -> bool:
        """Remove CrewAI instrumentation."""
        if not CREWAI_AVAILABLE or not self._instrumented:
            return True

        try:
            # Restore original methods if we wrapped them
            for key, original_method in self._original_methods.items():
                class_name, method_name = key.split(".")
                if class_name == "Crew" and Crew:
                    setattr(Crew, method_name, original_method)
                elif class_name == "Task" and Task:
                    setattr(Task, method_name, original_method)
                elif class_name == "Agent" and Agent:
                    setattr(Agent, method_name, original_method)

            self._original_methods.clear()
            self.context_manager.alignx_context.clear_framework_context()
            self._instrumented = False

            logger.info("AlignX CrewAI framework instrumentation disabled.")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstrument CrewAI: {e}")
            return False

    def _instrument_crew_operations(self):
        """Instrument CrewAI crew operations."""
        if not CREWAI_AVAILABLE or not Crew:
            return

        def wrap_crew_kickoff(wrapped, instance, args, kwargs):
            # Extract crew context
            crew_id = getattr(instance, "id", id(instance))
            crew_data = self._extract_crew_data(instance)

            self.context_manager.set_crew_context(str(crew_id), crew_data)

            try:
                result = wrapped(*args, **kwargs)
                return result
            finally:
                self.context_manager.clear_crew_context(str(crew_id))

        # Wrap crew kickoff methods
        if hasattr(Crew, "kickoff"):
            self._original_methods["Crew.kickoff"] = Crew.kickoff
            wrapt.wrap_function_wrapper(Crew, "kickoff", wrap_crew_kickoff)

        if hasattr(Crew, "kickoff_async"):
            self._original_methods["Crew.kickoff_async"] = Crew.kickoff_async
            wrapt.wrap_function_wrapper(Crew, "kickoff_async", wrap_crew_kickoff)

    def _instrument_task_operations(self):
        """Instrument CrewAI task operations."""
        if not CREWAI_AVAILABLE or not Task:
            return

        def wrap_task_execute(wrapped, instance, args, kwargs):
            # Extract task context
            task_id = getattr(instance, "id", id(instance))
            task_data = self._extract_task_data(instance)

            self.context_manager.set_task_context(str(task_id), task_data)

            try:
                result = wrapped(*args, **kwargs)
                return result
            finally:
                self.context_manager.clear_task_context(str(task_id))

        # Wrap task execution methods
        if hasattr(Task, "execute_sync"):
            self._original_methods["Task.execute_sync"] = Task.execute_sync
            wrapt.wrap_function_wrapper(Task, "execute_sync", wrap_task_execute)

        if hasattr(Task, "execute_async"):
            self._original_methods["Task.execute_async"] = Task.execute_async
            wrapt.wrap_function_wrapper(Task, "execute_async", wrap_task_execute)

    def _instrument_agent_operations(self):
        """Instrument CrewAI agent operations."""
        if not CREWAI_AVAILABLE or not Agent:
            return

        def wrap_agent_execute(wrapped, instance, args, kwargs):
            # Extract agent context
            agent_id = getattr(instance, "id", id(instance))
            agent_data = self._extract_agent_data(instance)

            self.context_manager.set_agent_context(str(agent_id), agent_data)

            try:
                result = wrapped(*args, **kwargs)
                return result
            finally:
                self.context_manager.clear_agent_context(str(agent_id))

        # Wrap agent execution methods
        if hasattr(Agent, "execute_task"):
            self._original_methods["Agent.execute_task"] = Agent.execute_task
            wrapt.wrap_function_wrapper(Agent, "execute_task", wrap_agent_execute)

        if hasattr(Agent, "_execute_core"):
            self._original_methods["Agent._execute_core"] = Agent._execute_core
            wrapt.wrap_function_wrapper(Agent, "_execute_core", wrap_agent_execute)

    def _extract_crew_data(self, crew_instance) -> Dict[str, Any]:
        """Extract relevant data from a CrewAI crew instance."""
        data = {}

        # Extract basic crew information
        if hasattr(crew_instance, "agents"):
            data["agent_count"] = (
                len(crew_instance.agents) if crew_instance.agents else 0
            )

        if hasattr(crew_instance, "tasks"):
            data["task_count"] = len(crew_instance.tasks) if crew_instance.tasks else 0

        if hasattr(crew_instance, "process"):
            data["process"] = str(crew_instance.process)

        if hasattr(crew_instance, "verbose"):
            data["verbose"] = crew_instance.verbose

        if hasattr(crew_instance, "memory"):
            data["has_memory"] = crew_instance.memory is not None

        return data

    def _extract_task_data(self, task_instance) -> Dict[str, Any]:
        """Extract relevant data from a CrewAI task instance."""
        data = {}

        # Extract basic task information
        if hasattr(task_instance, "description"):
            description = getattr(task_instance, "description", "")
            data["description_length"] = len(description) if description else 0

        if hasattr(task_instance, "agent"):
            agent = getattr(task_instance, "agent", None)
            if agent and hasattr(agent, "role"):
                data["agent_role"] = getattr(agent, "role", "unknown")

        if hasattr(task_instance, "tools"):
            tools = getattr(task_instance, "tools", [])
            data["tool_count"] = len(tools) if tools else 0

        if hasattr(task_instance, "expected_output"):
            expected_output = getattr(task_instance, "expected_output", "")
            data["has_expected_output"] = bool(expected_output)

        if hasattr(task_instance, "context"):
            context = getattr(task_instance, "context", [])
            data["context_count"] = len(context) if context else 0

        return data

    def _extract_agent_data(self, agent_instance) -> Dict[str, Any]:
        """Extract relevant data from a CrewAI agent instance."""
        data = {}

        # Extract basic agent information
        if hasattr(agent_instance, "role"):
            data["role"] = getattr(agent_instance, "role", "unknown")

        if hasattr(agent_instance, "goal"):
            goal = getattr(agent_instance, "goal", "")
            data["goal_length"] = len(goal) if goal else 0

        if hasattr(agent_instance, "backstory"):
            backstory = getattr(agent_instance, "backstory", "")
            data["backstory_length"] = len(backstory) if backstory else 0

        if hasattr(agent_instance, "tools"):
            tools = getattr(agent_instance, "tools", [])
            data["tool_count"] = len(tools) if tools else 0

        if hasattr(agent_instance, "llm"):
            data["has_llm"] = getattr(agent_instance, "llm", None) is not None

        if hasattr(agent_instance, "verbose"):
            data["verbose"] = getattr(agent_instance, "verbose", False)

        if hasattr(agent_instance, "allow_delegation"):
            data["allow_delegation"] = getattr(
                agent_instance, "allow_delegation", False
            )

        return data

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the CrewAI instrumentation."""
        metadata = {
            "strategy": self.name,
            "framework": self.framework_name,
            "available": self.is_available(),
            "instrumented": self._instrumented,
        }

        if CREWAI_AVAILABLE:
            try:
                import crewai

                metadata["crewai_version"] = crewai.__version__
            except (ImportError, AttributeError):
                metadata["crewai_version"] = "unknown"

        # Add current context information
        metadata["current_context"] = self.context_manager.get_current_context()

        return metadata
