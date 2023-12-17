import itertools
import os
import re
import subprocess
import tempfile
from typing import List, Tuple

import dateutil.parser as dp
import requests


def flatten(l):
    """Flatten list of lists.
    Args:
        l: A list of lists
    Returns: A flattened iterable
    """
    return itertools.chain.from_iterable(l)


def chunks(l: List, n: int):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def remap_nwo(nwo: str) -> Tuple[str, str]:
    r = requests.get("https://github.com/{}".format(nwo))
    if r.status_code not in (404, 451, 502):  # DMCA
        if "migrated" not in r.text:
            if r.history:
                return (
                    nwo,
                    "/".join(
                        re.findall(r'"https://github.com/.+"', r.history[0].text)[0]
                        .strip('"')
                        .split("/")[-2:]
                    ),
                )
            return (nwo, nwo)
    return (nwo, None)


def get_sha(tmp_dir: tempfile.TemporaryDirectory, nwo: str):
    os.chdir(os.path.join(tmp_dir.name, nwo))
    # git rev-parse HEAD
    cmd = ["git", "rev-parse", "HEAD"]
    sha = subprocess.check_output(cmd).strip().decode("utf-8")
    os.chdir("/tmp")
    return sha


def get_sha_svn(tmp_dir: tempfile.TemporaryDirectory, nwo: str):
    os.chdir(os.path.join(tmp_dir.name, nwo))
    cmd = "svn info | grep \"Revision\" | awk -F ': ' '{print $2}'"
    sha = subprocess.getoutput(cmd).strip()
    os.chdir("/tmp")
    return sha


def get_oldest_sha(tmp_dir: tempfile.TemporaryDirectory, nwo: str, sha_list):
    os.chdir(os.path.join(tmp_dir.name, nwo))
    oldest_sha = 0
    oldest_date = 0
    for sha in sha_list:
        cmd = ["git", "log", '--pretty=format:"%ct"', sha, "-1"]
        date = int(subprocess.check_output(cmd, encoding="UTF-8").replace('"', ""))
        if oldest_date == 0 or date < oldest_date:
            oldest_date = date
            oldest_sha = sha
    os.chdir("/tmp")
    return oldest_sha


def get_oldest_sha_svn(nwo: str, sha_list):
    oldest_sha = 0
    oldest_date = 0
    for sha in sha_list:
        cmd = [
            "svnlook",
            "date",
            "/home/Project/dl/AlphaRepair_finetune/defects4j/project_repos/{}".format(
                nwo
            ),
            "-r",
            sha,
        ]
        date = subprocess.check_output(cmd, encoding="UTF-8").split(" ")
        ISO_8601_date = date[0] + "T" + date[1] + date[2][:3] + ":" + date[2][3:]
        date_stamp = dp.parse(ISO_8601_date).strftime("%s")
        if oldest_date == 0 or date_stamp < oldest_date:
            oldest_date = date_stamp
            oldest_sha = sha
    return oldest_sha


def go_to_sha(tmp_dir: tempfile.TemporaryDirectory, nwo: str, sha: str):
    os.chdir(os.path.join(tmp_dir.name, nwo))
    cmd = ["git", "checkout", sha]
    command_output = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    os.chdir("/tmp")
    return tmp_dir


def download(nwo: str):
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    tmp_dir = tempfile.TemporaryDirectory()
    cmd = [
        "git",
        "clone",
        "--depth=1",
        "https://github.com/{}.git".format(nwo),
        "{}/{}".format(tmp_dir.name, nwo),
    ]
    subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    return tmp_dir


def download_local(nwo: str):
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    tmp_dir = tempfile.TemporaryDirectory()
    cmd = [
        "git",
        "clone",
        "--depth=1",
        "/home/Project/dl/AlphaRepair_finetune/defects4j/project_repos/{}.git".format(
            nwo
        ),
        "{}/{}".format(tmp_dir.name, nwo),
    ]
    command_output = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    return tmp_dir


def download_local_for_DefextsKotlin(nwo: str):
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    tmp_dir = tempfile.TemporaryDirectory()
    cmd = ["python", "defexts.py", "kotlin", "-c", nwo, "-b", "-cp", tmp_dir.name]
    command_output = subprocess.run(
        cmd,
        cwd="/home/Project/dl/defexts",
        stdin=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    return tmp_dir


def download_local_svn_with_SHA(nwo: str, sha: str):
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    tmp_dir = tempfile.TemporaryDirectory()
    cmd = [
        "svn",
        "checkout",
        "file:///"
        + "/home/Project/dl/AlphaRepair_finetune/defects4j/project_repos/{}".format(
            nwo
        ),
        "-r",
        "{}".format(sha),
        "{}/{}".format(tmp_dir.name, nwo),
    ]
    command_output = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    return tmp_dir


def walk(tmp_dir: tempfile.TemporaryDirectory, ext: str):
    results = []
    for root, _, files in os.walk(tmp_dir.name):
        for f in files:
            if ext == "kotlin":
                if f.endswith(".kt"):
                    results.append(os.path.join(root, f))
            else:
                if f.endswith("." + ext):
                    results.append(os.path.join(root, f))
    return results


def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
