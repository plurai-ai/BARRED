"""Utility functions for prompt template rendering."""

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pathlib import Path
from typing import Any


def render_jinja_prompts(
    template_file: str,
    render_context: dict[str, Any],
) -> dict[str, Any]:
    """Render a Jinja2 YAML template and return the parsed data.

    Args:
        template_file: Path to the Jinja2 template file
        render_context: Dictionary of variables to pass to the template

    Returns:
        Parsed YAML data after rendering
    """
    prompts_path = Path(template_file)

    jinja_env = Environment(
        loader=FileSystemLoader(str(prompts_path.parent)),
        undefined=StrictUndefined
    )

    template = jinja_env.get_template(prompts_path.name)
    rendered_yaml = template.render(**render_context)
    prompts_data = yaml.safe_load(rendered_yaml)

    return prompts_data
