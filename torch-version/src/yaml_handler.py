import yaml
import logging

def load_yaml(path: str) -> dict:
    """
    Return the parsed yaml
    """

    logger = logging.getLogger("yaml_handler:load_yaml")

    logger.debug("Opening yaml: %s", path)
    yaml_file = open(path)

    logger.debug("Parsing yaml")
    parsed = yaml.load(yaml_file, Loader=yaml.FullLoader)

    logger.info("Yaml successfully loaded")

    return parsed
