from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    type: Literal['openai', 'azure', 'azure_image', 'google', 'google_image', 'oracle', 'anthropic_vertex', 'anthropic'] = Field(
        ..., description='LLM provider type'
    )
    name: str = Field(..., description='Model name or deployment name')
    temperature: float = Field(default=0, ge=0, le=2, description='Sampling temperature')
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict, description='Additional provider-specific arguments (e.g., reasoning_effort for Azure o1 models)'
    )


class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI configuration - all fields required together."""

    deployment_name: str = Field(..., description='Azure OpenAI deployment name')
    api_key: SecretStr = Field(..., description='Azure OpenAI API key')
    api_version: str = Field(..., description='OpenAI API version')
    endpoint: str = Field(..., description='Azure OpenAI endpoint URL')


class OpenAIConfig(BaseModel):
    """OpenAI configuration."""

    api_key: SecretStr = Field(..., description='OpenAI API key')
    organization: str | None = Field(default=None, description='OpenAI organization ID')
    api_base: str | None = Field(default=None, description='OpenAI API base URL')


class GoogleConfig(BaseModel):
    """Google AI configuration."""

    api_key: SecretStr = Field(..., description='Google AI API key')


class AnthropicConfig(BaseModel):
    """Anthropic configuration."""

    api_key: SecretStr = Field(..., description='Anthropic API key')


class AnthropicVertexConfig(BaseModel):
    """Anthropic Vertex AI configuration - all fields required together."""

    project_id: str = Field(..., description='Google Cloud project ID')
    region: str = Field(..., description='Google Cloud region')


class OracleConfig(BaseModel):
    """Oracle OCI Generative AI configuration - all fields required together."""

    service_endpoint: str = Field(..., description='Oracle OCI service endpoint')
    compartment_id: str = Field(..., description='Oracle OCI compartment ID')


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra='ignore',
        env_nested_delimiter='__',
    )

    # Provider configurations (grouped atomically)
    azure_openai: AzureOpenAIConfig | None = Field(default=None, description='Azure OpenAI configuration')
    openai: OpenAIConfig | None = Field(default=None, description='OpenAI configuration')
    google: GoogleConfig | None = Field(default=None, description='Google AI configuration')
    anthropic: AnthropicConfig | None = Field(default=None, description='Anthropic configuration')
    anthropic_vertex: AnthropicVertexConfig | None = Field(default=None, description='Anthropic Vertex AI configuration')
    oracle: OracleConfig | None = Field(default=None, description='Oracle OCI configuration')

    # Default LLM configuration
    default_llm: LLMConfig | None = Field(
        default=None,
        description='Default LLM configuration to use when no config is explicitly provided to get_llm(). '
                    'If not set, will be automatically populated from azure_openai if available.',
    )

    @model_validator(mode='after')
    def set_default_llm_from_azure(self):
        """Automatically set default_llm from azure_openai if not explicitly configured."""
        if self.default_llm is None and self.azure_openai is not None:
            self.default_llm = LLMConfig(
                type='azure',
                name=self.azure_openai.deployment_name,
                temperature=0,
            )
        return self
