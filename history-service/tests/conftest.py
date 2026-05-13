"""
Pytest configuration for history-service tests.

Sets up the test environment by ensuring required environment variables
are available and configuring pytest plugins.
"""
import os
import pytest

# Set default environment variables for testing
os.environ.setdefault("DISCORD_TOKEN", "fake_token_for_testing")
os.environ.setdefault("DISCORD_GUILD_ID", "123456789012345678")
os.environ.setdefault("PROXY_URL", "http://proxy:11436")


@pytest.fixture(scope="session")
def test_config():
    """Provide shared test configuration."""
    return {
        "discord_token": os.environ["DISCORD_TOKEN"],
        "discord_guild_id": os.environ["DISCORD_GUILD_ID"],
        "proxy_url": os.environ["PROXY_URL"],
    }
