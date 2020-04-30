import json

from pathlib import Path


# Resolve file path for project directory
pwd = Path(__file__).parents[2]


def create_template():
    with open(pwd / "config.json", "w") as json_file:
        config = {
            "pwd": pwd.as_posix(),
            # "data": "",
        }
        json.dump(config, json_file, indent=4)


def read_config():
    try:
        with open(pwd / "config.json", "r") as json_file:
            config = json.load(json_file)
            validate(config)

        return config
    except FileNotFoundError:
        create_template()

        print("Created default configuration file config.json. Modify as needed. \n")

        return config
    except ValueError:
        Path(pwd / "config.json").unlink()

        print("Could not resolve config.json. Re-run configuration.")


def validate(config):
    try:
        flag = True

        for item in config:
            flag = flag and Path(config[item]).exists()

        if flag is False:
            raise FileNotFoundError

        # Create required sub-folders
        for item in ["notebook", "png"]:
            sub_path = Path(config["pwd"]) / item
            sub_path.mkdir(exist_ok=True)

        # Symlink and check .pq data location
        # Replaces any existing symlink
        pq_path = Path(pwd / "pq")
        if pq_path.exists():
            pq_path.unlink(missing_ok=True)
        pq_path.symlink_to(config["data"])

        if Path(config["data"]).exists():
            print(sorted(pq_path.glob("*")))
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Config dictionary contains unresolvable path(s). \n")


if __name__ == "__main__":
    result = read_config()

    if isinstance(result, dict):
        print(result)
