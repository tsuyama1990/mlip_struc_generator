import hydra
from omegaconf import DictConfig, OmegaConf
from nnp_gen.core.config import AppConfig
from nnp_gen.pipeline.runner import PipelineRunner
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("Starting MLIP Structure Generator...")
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Convert Hydra config to Pydantic AppConfig
    try:
        app_config = AppConfig(**OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        logger.error(f"Configuration Validation Error: {e}")
        print(f"Error: Invalid configuration. {e}")
        return

    runner = PipelineRunner(app_config)
    runner.run()
    print("Done.")

if __name__ == "__main__":
    main()
