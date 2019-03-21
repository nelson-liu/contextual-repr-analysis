#! /usr/bin/env python

# Script to launch contexteval Beaker jobs.

import argparse
import os
import json
import random
import tempfile
import subprocess
import sys

# This has to happen before we import spacy (even indirectly), because for some crazy reason spacy
# thought it was a good idea to set the random seed on import...
random_int = random.randint(0, 2**32)
from allennlp.common.params import Params


def main(param_file: str, args: argparse.Namespace):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
    image = f"contexteval:{commit}"
    overrides = ""

    # Reads params and sets environment.
    params = Params.from_file(param_file, overrides)
    flat_params = params.as_flat_dict()
    env = {}
    for k, v in flat_params.items():
        k = str(k).replace('.', '_')
        env[k] = str(v)

    # Check if the experiment name exists. If it does, don't run and exit.
    if args.name:
        try:
            name_check_result = subprocess.check_output(f'beaker experiment inspect {args.name}',
                                                        shell=True, universal_newlines=True).strip()
            # Check if the status is "succeeded". If so, exit.
            status = json.loads(name_check_result)[0]["nodes"][0].get("status", None)
            if status == "succeeded" or status == "running":
                print(f"{args.name} already exists and has status {status}, exiting...")
                sys.exit(0)
            else:
                # Delete this experiment and rerun
                print(f"{args.name} already exists and was not successful, deleting and rerunning...")
                # Temporary hack since there's no beaker experiment delete -- rename to beaker experiment name_broken
                deletion_result = subprocess.check_output(f'beaker experiment rename {args.name} {args.name}_broken',
                                                          shell=True, universal_newlines=True).strip()
        except subprocess.CalledProcessError as called_process_error:
            # Rerun the experiment (proceed with execution), since
            # the experiment doesn't exist
            pass

    # If the git repository is dirty, add a random hash.
    result = subprocess.run('git diff-index --quiet HEAD --', shell=True)
    if result.returncode != 0:
        dirty_hash = "%x" % random_int
        image += "-" + dirty_hash

    if args.blueprint:
        blueprint = args.blueprint
        print(f"Using the specified blueprint: {blueprint}")
    else:
        print(f"Building the Docker image ({image})...")
        subprocess.run(f'docker build -t {image} .', shell=True, check=True)

        print(f"Create a Beaker blueprint...")
        blueprint = subprocess.check_output(f'beaker blueprint create --quiet {image}', shell=True,
                                            universal_newlines=True).strip()
        print(f"  Blueprint created: {blueprint}")

    config_dataset_id = subprocess.check_output(f'beaker dataset create --quiet {param_file}', shell=True,
                                                universal_newlines=True).strip()
    overrides = {}
    if args.layer_num is not None:
        if "dataset_reader" in params:
            overrides["dataset_reader"] = {
                "contextualizer": {
                    "layer_num": args.layer_num
                }
            }
        if "validation_dataset_reader" in params:
            overrides["validation_dataset_reader"] = {
                "contextualizer": {
                    "layer_num": args.layer_num
                }
            }

    if args.max_instances is not None:
        overrides["dataset_reader"]["max_instances"] = args.max_instances
        overrides["validation_dataset_reader"]["max_instances"] = args.max_instances

    allennlp_command = [
        "allennlp",
        "train",
        "/config.json",
        "-s",
        "/output",
        "--file-friendly-logging",
        "--include-package",
        "contexteval",
        "--overrides",
        json.dumps(overrides)
    ]

    dataset_mounts = []
    for source in args.source + [f"{config_dataset_id}:/config.json"]:
        datasetId, containerPath = source.split(":")
        if not os.path.isabs(containerPath):
            absoluteContainerPath = os.path.join(args.workdir, containerPath)
            print("Provided container path {} is relative, converted to "
                  "be relative to workdir ({})".format(containerPath, absoluteContainerPath))
        else:
            absoluteContainerPath = containerPath
        dataset_mounts.append({
            "datasetId": datasetId,
            "containerPath": absoluteContainerPath
        })

    for var in args.env:
        key, value = var.split("=")
        env[key] = value

    requirements = {}
    if args.cpu:
        requirements["cpu"] = float(args.cpu)
    if args.memory:
        requirements["memory"] = args.memory
    if args.gpu_count:
        requirements["gpuCount"] = int(args.gpu_count)
    config_spec = {
        "description": args.desc,
        "blueprint": blueprint,
        "resultPath": "/output",
        "args": allennlp_command,
        "datasetMounts": dataset_mounts,
        "requirements": requirements,
        "env": env
    }
    config_task = {"spec": config_spec, "name": "training"}

    config = {
        "tasks": [config_task]
    }

    output_path = args.spec_output_path if args.spec_output_path else tempfile.mkstemp(
        ".yaml", "beaker-config-")[1]
    with open(output_path, "w") as output:
        output.write(json.dumps(config, indent=4))
    print(f"Beaker spec written to {output_path}.")

    experiment_command = ["beaker", "experiment", "create", "--file", output_path]
    if args.name:
        experiment_command.append("--name")
        experiment_command.append(args.name.replace(" ", "-"))

    if args.dry_run:
        print(f"This is a dry run (--dry-run).  Launch your job with the following command:")
        print(f"    " + " ".join(experiment_command))
    else:
        print(f"Running the experiment:")
        print(f"    " + " ".join(experiment_command))
        subprocess.run(experiment_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('param_file', type=str, help='The model configuration file.')
    parser.add_argument('--name', type=str, help='A name for the experiment.')
    parser.add_argument('--layer-num', type=int, help='The layer to evaluate.')
    parser.add_argument('--max-instances', type=float,
                        help='The proportion of data to use for training the auxiliary classifier.')
    parser.add_argument('--workdir', type=str, default="/stage/allennlp",
                        help=('The absolute path to the workdir of the docker container, for '
                              'compatibility with relative paths.'))
    parser.add_argument('--spec-output-path', type=str, help='The destination to write the experiment spec.')
    parser.add_argument('--dry-run', action='store_true', help='If specified, an experiment will not be created.')
    parser.add_argument('--blueprint', type=str, help='The Blueprint to use (if unspecified one will be built)')
    parser.add_argument('--desc', type=str, help='A description for the experiment.')
    parser.add_argument('--env', action='append', default=[],
                        help='Set environment variables (e.g. NAME=value or NAME)')
    parser.add_argument('--source', action='append', default=[],
                        help='Bind a remote data source (e.g. source-id:/target/path)')
    parser.add_argument('--cpu', help='CPUs to reserve for this experiment (e.g., 0.5)')
    parser.add_argument('--gpu-count', default=1, help='GPUs to use for this experiment (e.g., 1 (default))')
    parser.add_argument('--memory', help='Memory to reserve for this experiment (e.g., 1GB)')

    args = parser.parse_args()

    main(args.param_file, args)
