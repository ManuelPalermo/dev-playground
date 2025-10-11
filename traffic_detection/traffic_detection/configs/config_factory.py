# NOTE: configs for other sites can be added here
from traffic_detection.configs.demo_site_config import demo_site_config

KNOW_SITE_CONFIGS = [
    "demo",
]


def get_site_config(site_name: str) -> dict:
    """Get the configuration for a specific site.

    Args:
        site_name: Name of the site.

    Returns:
        dict: Configuration dictionary for the specified site.
    """
    if site_name == "demo":
        return demo_site_config()

    raise ValueError(f"Configuration for site '{site_name}' not found. Known ones are: {KNOW_SITE_CONFIGS}")
