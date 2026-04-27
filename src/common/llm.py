"""LLM utility functions for creating and configuring language models."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from google.genai.types import GoogleSearch, Tool as GenAITool
from langchain_community.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_bedrock_anthropic_callback
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import AzureChatOpenAI

from common.settings import Settings, LLMConfig

GEMINI_HUMAN_MESSAGE = 'Process the instructions above.'


def is_google_llm(llm: BaseChatModel) -> bool:
    """Check if the LLM is a Google model (Gemini) that requires human message.

    Args:
        llm: The language model to check

    Returns:
        True if the LLM is a Google model, False otherwise
    """
    module = llm.__class__.__module__
    class_name = llm.__class__.__name__
    return 'google' in module.lower() or class_name == 'ChatGoogleGenerativeAI'


def get_llm(settings: Settings, config: LLMConfig | None = None, timeout: int = 60) -> BaseChatModel:
    # Validate timeout
    if timeout <= 0:
        raise ValueError(f'timeout must be positive, got {timeout}')

    if config is None:
        raise ValueError(
            'No default LLM configuration found. '
            'Either provide a config parameter (LLMConfig), set settings.default_llm, or configure settings.azure_openai. '
            'Example: settings.default_llm = LLMConfig(type="anthropic", name="claude-3-5-sonnet-20241022", temperature=0)'
        )

    # Extract common fields
    temperature = config.temperature
    model_kwargs = config.model_kwargs
    provider_type = config.type

    if provider_type == 'openai':
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError('langchain-openai is not installed. Install it with: pip install langchain-openai') from e

        if not settings.openai:
            raise ValueError(
                'OpenAI configuration not found in settings. '
                'Configure settings.openai with required credentials. '
                'Example: settings.openai = OpenAIConfig(api_key="sk-...")'
            )

        params = {
            'temperature': temperature,
            'model_name': config.name,
            'openai_api_key': settings.openai.api_key.get_secret_value(),
            'openai_api_base': settings.openai.api_base,
            'model_kwargs': model_kwargs,
            'timeout': timeout,
        }

        # Add organization if provided in settings
        if settings.openai.organization:
            params['openai_organization'] = settings.openai.organization

        try:
            return ChatOpenAI(**params)
        except Exception as e:
            raise RuntimeError(f'Failed to initialize OpenAI model "{config.name}". Check model name and credentials. Error: {e}') from e

    if provider_type == 'azure':
        if not settings.azure_openai:
            raise ValueError(
                'Azure OpenAI configuration not found in settings. '
                'Configure settings.azure_openai with required credentials. '
                'Example: settings.azure_openai = AzureOpenAIConfig(deployment_name="...", api_key="...", endpoint="...", api_version="...")'
            )

        params = {
            'temperature': temperature,
            'azure_deployment': config.name,
            'openai_api_key': settings.azure_openai.api_key.get_secret_value(),
            'azure_endpoint': settings.azure_openai.endpoint,
            'openai_api_version': settings.azure_openai.api_version,
            'timeout': timeout,
            **model_kwargs,
        }

        try:
            return AzureChatOpenAI(**params)
        except Exception as e:
            raise RuntimeError(f'Failed to initialize Azure OpenAI model "{config.name}". Check deployment name, endpoint, and credentials. Error: {e}') from e

    if provider_type == 'google':
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError('langchain-google-genai is not installed. Install it with: pip install langchain-google-genai') from e

        if not settings.google:
            raise ValueError(
                'Google AI configuration not found in settings. '
                'Configure settings.google with required credentials. '
                'Example: settings.google = GoogleConfig(api_key="...")'
            )

        try:
            return ChatGoogleGenerativeAI(
                temperature=temperature,
                model=config.name,
                google_api_key=settings.google.api_key.get_secret_value(),
                model_kwargs=model_kwargs,
                timeout=timeout,
            )
        except Exception as e:
            raise RuntimeError(f'Failed to initialize Google AI model "{config.name}". Check model name and credentials. Error: {e}') from e

    if provider_type == 'oracle':
        try:
            from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
        except ImportError as e:
            raise ImportError('langchain-community is not installed. Install it with: pip install langchain-community') from e

        if not settings.oracle:
            raise ValueError(
                'Oracle OCI configuration not found in settings. '
                'Configure settings.oracle with required credentials. '
                'Example: settings.oracle = OracleConfig(service_endpoint="...", compartment_id="...")'
            )

        # Ensure max_tokens default for Oracle
        if 'max_tokens' not in model_kwargs:
            model_kwargs['max_tokens'] = 4000

        try:
            return ChatOCIGenAI(
                model_id=config.name,
                service_endpoint=settings.oracle.service_endpoint,
                compartment_id=settings.oracle.compartment_id,
                model_kwargs=model_kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f'Failed to initialize Oracle OCI model "{config.name}". Check model ID, service endpoint, and compartment ID. Error: {e}'
            ) from e

    if provider_type == 'anthropic_vertex':
        try:
            from langchain_google_vertexai.model_garden import ChatAnthropicVertex
        except ImportError as e:
            raise ImportError('langchain-google-vertexai is not installed. Install it with: pip install langchain-google-vertexai') from e

        if not settings.anthropic_vertex:
            raise ValueError(
                'Anthropic Vertex AI configuration not found in settings. '
                'Configure settings.anthropic_vertex with required credentials. '
                'Example: settings.anthropic_vertex = AnthropicVertexConfig(project_id="...", region="...")'
            )

        try:
            return ChatAnthropicVertex(
                temperature=temperature,
                model=config.name,
                project=settings.anthropic_vertex.project_id,
                location=settings.anthropic_vertex.region,
                model_kwargs=model_kwargs,
                timeout=timeout,
            )
        except Exception as e:
            raise RuntimeError(f'Failed to initialize Anthropic Vertex AI model "{config.name}". Check model name, project ID, and region. Error: {e}') from e

    if provider_type == 'anthropic':
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError('langchain-anthropic is not installed. Install it with: pip install langchain-anthropic') from e

        from pydantic import SecretStr

        if not settings.anthropic:
            raise ValueError(
                'Anthropic configuration not found in settings. '
                'Configure settings.anthropic with required credentials. '
                'Example: settings.anthropic = AnthropicConfig(api_key="sk-ant-...")'
            )

        try:
            return ChatAnthropic(
                temperature=temperature,
                model_name=config.name,
                api_key=SecretStr(settings.anthropic.api_key.get_secret_value()),
                model_kwargs=model_kwargs,
                timeout=timeout,
                stop=None,
            )
        except Exception as e:
            raise RuntimeError(f'Failed to initialize Anthropic model "{config.name}". Check model name and credentials. Error: {e}') from e

    raise ValueError(f'Unsupported LLM provider: {provider_type}. Supported providers: openai, azure, google, oracle, anthropic_vertex, anthropic')


def get_prompt_template(args: dict[str, Any], human_message: str | None = None) -> ChatPromptTemplate:
    """Create a ChatPromptTemplate from various input formats.

    Args:
        args: Dictionary containing one of: prompt_hub_name, prompt, from_str, message_list, or path
        human_message: Optional human message to append after system message.
                       Used internally for Gemini compatibility.

    Returns:
        ChatPromptTemplate configured with system message and optional human message
    """
    if 'prompt_hub_name' in args:
        # TODO: Import langchain hub when needed
        raise NotImplementedError('Prompt hub support not yet implemented')
    if 'prompt' in args:
        return args['prompt']
    if 'from_str' in args:
        messages: list[tuple[str, str]] = [('system', args['from_str']['template'])]
        if human_message:
            messages.append(('human', human_message))
        return ChatPromptTemplate.from_messages(messages)
    if 'message_list' in args:
        messages = args['message_list']
        return ChatPromptTemplate.from_messages(messages)
    if 'path' in args:
        with open(args['path']) as file:
            messages = [('system', file.read())]
            if human_message:
                messages.append(('human', human_message))
            return ChatPromptTemplate.from_messages(messages)
    else:
        raise ValueError('Either prompt or prompt_hub_name should be provided')


def set_llm_chain(llm: BaseChatModel, **kwargs: Any) -> Runnable:
    """Initialize a chain with prompt template and optional structured output.

    Args:
        llm: The language model to use
        **kwargs: Additional arguments passed to get_prompt_template (from_str, path, etc.)
                  and optionally 'structure' for structured output.

    Returns:
        Configured Runnable chain
    """
    # Auto-detect Google LLM and inject human message for Gemini compatibility
    human_message = GEMINI_HUMAN_MESSAGE if is_google_llm(llm) else None

    system_prompt_template = get_prompt_template(kwargs, human_message=human_message)
    if 'structure' in kwargs:
        return system_prompt_template | llm.with_structured_output(kwargs['structure'])
    return system_prompt_template | llm


async def invoke_with_google_search(llm: BaseChatModel, prompt: str, executor: ThreadPoolExecutor | None = None) -> Any:
    """
    Helper function to invoke Gemini LLM with Google Search tool.

    Uses run_in_executor to avoid blocking async operations during the search.

    Args:
        llm: The Gemini LLM instance to use
        prompt: The formatted prompt string to send to the LLM
        executor: Optional ThreadPoolExecutor for running the blocking call

    Returns:
        The LLM response object
    """
    loop = asyncio.get_event_loop()

    def blocking_invoke():
        return llm.invoke(
            prompt,
            tools=[GenAITool(google_search=GoogleSearch())],
        )

    if executor:
        return await loop.run_in_executor(executor, blocking_invoke)
    return await loop.run_in_executor(None, blocking_invoke)


class DummyCallback:
    """
    A dummy callback for the LLM.
    This is a trick to handle an empty callback.
    """

    def __enter__(self):
        self.total_cost = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def get_dummy_callback():
    return DummyCallback()


def set_callback(llm_type):
    if llm_type.lower() == 'openai' or llm_type.lower() == 'azure':
        callback = get_openai_callback
    elif llm_type.lower() == 'anthropic_bedrock':
        callback = get_bedrock_anthropic_callback
    else:
        callback = get_dummy_callback
    return callback
