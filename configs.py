from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os


load_dotenv()


class Configs(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_temperature: float = os.getenv("OPENAI_TEMPERATURE", 1.0)
    openai_timeout: float = os.getenv("OPENAI_TIMEOUT", 30.0)

    class Config:
        env_file = ".env"
        extra = "allow"


configs = Configs()
