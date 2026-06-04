from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    openai_api_key: str | None = None
    database_url: str = "postgresql://docschat:docschat@db:5432/docschat"
    generation_model: str = "gpt-4.1-mini"
    planner_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1500
    chunk_overlap: int = 300
    retrieval_top_k: int = 6
    fusion_vector_weight: float = 0.7
    fusion_fts_weight: float = 0.3
    max_upload_mb: int = 20
    low_confidence_threshold: float = 0.20


settings = Settings()
