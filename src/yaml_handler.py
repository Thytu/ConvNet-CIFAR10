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

def write_yaml(yaml_dict: dict, path: str) -> None:
    """
    Take a dict and write it to a yaml file
    """

    logger = logging.getLogger("yaml_handler:write_yaml")

    logger.debug("Opening file: %s", path)

    with open(path, 'w') as file:
        logger.debug("Writing yaml")

        documents = yaml.dump(yaml_dict, file)

    logger.info("Yaml successfully writed")