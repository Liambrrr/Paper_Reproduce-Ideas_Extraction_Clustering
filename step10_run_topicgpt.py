from topicgpt_python import *
import yaml

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

generate_topic_lvl1(
    "openai",
    "gpt-4o",
    config["data_sample"],
    config["generation"]["prompt"],
    config["generation"]["seed"],
    config["generation"]["output"],
    config["generation"]["topic_output"],
    verbose=config["verbose"],
)