"""Download and extract the official NASA C-MAPSS dataset."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath


OFFICIAL_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
OUTER_ARCHIVE_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
INNER_ARCHIVE_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"
INNER_ARCHIVE_NAME = "CMAPSSData.zip"
REQUIRED_FD001_FILES = {
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 digest of in-memory content."""

    return hashlib.sha256(content).hexdigest()


def download_archive(destination: Path) -> None:
    """Download the outer NASA archive to a temporary file."""

    print(f"Downloading official archive from:\n{OFFICIAL_URL}")
    with urllib.request.urlopen(OFFICIAL_URL, timeout=60) as response, destination.open(
        "wb"
    ) as file:
        shutil.copyfileobj(response, file)


def read_inner_archive(outer_archive: Path) -> bytes:
    """Verify the NASA archive and return the nested CMAPSSData ZIP bytes."""

    outer_content = outer_archive.read_bytes()
    outer_digest = sha256_bytes(outer_content)
    if outer_digest != OUTER_ARCHIVE_SHA256:
        raise ValueError(
            "Official archive checksum mismatch. The upstream file may have changed; "
            "review the NASA source before updating the expected checksum."
        )

    with zipfile.ZipFile(BytesIO(outer_content)) as archive:
        candidates = [
            name for name in archive.namelist() if PurePosixPath(name).name == INNER_ARCHIVE_NAME
        ]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one nested CMAPSSData.zip archive.")
        inner_content = archive.read(candidates[0])

    if sha256_bytes(inner_content) != INNER_ARCHIVE_SHA256:
        raise ValueError("Nested CMAPSSData.zip checksum mismatch.")
    return inner_content


def extract_dataset(inner_content: bytes, destination: Path) -> None:
    """Extract the verified dataset without allowing archive path traversal."""

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(inner_content)) as archive:
        for member in archive.infolist():
            relative_path = PurePosixPath(member.filename)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            archive.extract(member, destination)

    missing = REQUIRED_FD001_FILES.difference(path.name for path in destination.iterdir())
    if missing:
        raise ValueError(f"FD001 extraction is incomplete; missing: {sorted(missing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already downloaded outer NASA ZIP instead of downloading it.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Extraction directory (default: {DEFAULT_DESTINATION})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.destination.resolve()
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Destination must be a directory: {destination}")
    present = {path.name for path in destination.iterdir()} if destination.is_dir() else set()
    if REQUIRED_FD001_FILES.issubset(present):
        print(f"FD001 files already exist in {destination}")
        return
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Destination is not empty but FD001 is incomplete: {destination}"
        )

    if args.archive is not None:
        inner_content = read_inner_archive(args.archive.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="cmapss-download-") as temporary_directory:
            outer_archive = Path(temporary_directory) / "nasa-cmapss.zip"
            download_archive(outer_archive)
            inner_content = read_inner_archive(outer_archive)

    extract_dataset(inner_content, destination)
    print(f"Verified and extracted NASA C-MAPSS data to {destination}")


if __name__ == "__main__":
    main()
