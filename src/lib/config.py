import json

from pathlib import Path


# Resolve file path for project working directory
pwd = Path(__file__).absolute().parents[2]


def create_template():
    with open(pwd / "config.json", "w") as json_file:
        config = {
            "pwd": pwd.as_posix(),
            "data": "",
            "output": "",
        }
        json.dump(config, json_file, indent=4)


def read_config():
    try:
        with open(pwd / "config.json", "r") as json_file:
            config = json.load(json_file)
            validate(config)
    except FileNotFoundError:
        create_template()

        print("Created default configuration file config.json. Modify as needed. \n")
    except ValueError:
        Path(pwd / "config.json").unlink()

        print("Could not resolve config.json. Re-run configuration.")
    finally:
        return config


def validate(config, output=False):
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
        pq_path = Path(pwd / "data")
        if pq_path.exists():
            pq_path.unlink(missing_ok=True)
        pq_path.symlink_to(config["data"])

        if Path(config["data"]).exists():
            if output is True:
                print("Found the following data entries: ")

                for item in sorted(pq_path.glob("*")):
                    print(f"\t {item.name}")

                print()
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Config dictionary contains unresolvable path(s). \n")

    # TODO: need a flag for symlink validation


if __name__ == "__main__":
    result = read_config()

    if isinstance(result, dict):
        print(json.dumps(result, indent=4))