"""AlignX-specific semantic conventions for unified tracing."""


class AlignXSemanticConventions:
    """AlignX-specific semantic conventions for consistent span attribution and correlation.

    These conventions ensure that spans are properly attributed with workflow context,
    provider information, and correlation metadata for unified tracing across the
    AlignX platform.
    """

    # Provider identification (standard OpenTelemetry)
    GEN_AI_SYSTEM = "gen_ai.system"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    LLM_REQUEST_TYPE = "llm.request.type"
    SERVER_ADDRESS = "server.address"

    # AlignX-specific attributes for unified tracing
    ALIGNX_PROVIDER = "alignx.provider"
    ALIGNX_OPERATION_TYPE = "alignx.operation_type"
    ALIGNX_COST_USD = "alignx.cost_usd"
    ALIGNX_REQUEST_TYPE = "alignx.request_type"

    # Correlation attributes
    ALIGNX_CORRELATION_ENABLED = "alignx.correlation_enabled"
    ALIGNX_CORRELATION_TRACE_ID = "alignx.correlation.trace_id"
    ALIGNX_CORRELATION_ORG_ID = "alignx.correlation.org_id"
    ALIGNX_CORRELATION_SOURCE = "alignx.correlation.source"

    # Workflow attributes
    ALIGNX_WORKFLOW_ID = "alignx.workflow.id"
    ALIGNX_WORKFLOW_NODE_NAME = "alignx.workflow.node.name"
    ALIGNX_WORKFLOW_NODE_SEQUENCE = "alignx.workflow.node.sequence"
    ALIGNX_WORKFLOW_NODE_TYPE = "alignx.workflow.node.type"

    # Dashboard and visualization attributes
    GEN_AI_APPLICATION_NAME = "gen_ai.application_name"
    GEN_AI_ENVIRONMENT = "gen_ai.environment"

    # Request type values
    REQUEST_TYPE_WORKFLOW_NODE = "workflow_node"
    REQUEST_TYPE_EXTERNAL = "external"

    # Correlation source values
    CORRELATION_SOURCE_BACKEND_WORKFLOW = "backend_workflow"
