#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pre-download all models required by NeMo Guardrails.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_cache_dir():
    if Path("/app").exists() and os.access("/app", os.W_OK):
        return Path("/app/.cache")
    return Path.home() / ".cache" / "nemo-guardrails"


def setup_cache():
    cache_base = get_cache_dir()
    for subdir in [
        "transformers",
        "huggingface",
        "sentence_transformers",
        "nltk_data",
        "fastembed",
    ]:
        (cache_base / subdir).mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "TRANSFORMERS_CACHE": str(cache_base / "transformers"),
            "HF_HOME": str(cache_base / "huggingface"),
            "SENTENCE_TRANSFORMERS_HOME": str(cache_base / "sentence_transformers"),
            "NLTK_DATA": str(cache_base / "nltk_data"),
            "FASTEMBED_CACHE_PATH": str(cache_base / "fastembed"),
        }
    )


def ensure_st_prefix(models):
    for model_name in models:
        if model_name.startswith("sentence-transformers/"):
            yield model_name
        else:
            yield f"sentence-transformers/{model_name}"


def download_spacy_models(models):
    for model in models:
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            logging.info(f"Downloaded SpaCy model: {model}")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else "Unknown error"
            logging.warning(f"Failed to download SpaCy model {model}: {error_msg}")


def download_sentence_transformers_models(models):
    try:
        import sentence_transformers
    except ImportError:
        logging.warning("Sentence Transformers not available - skipping")
        return
    for model_name in ensure_st_prefix(models):
        try:
            sentence_transformers.SentenceTransformer(model_name)
            logging.info(f"Downloaded Sentence Transformers model: {model_name}")
        except Exception as e:
            logging.warning(
                f"Failed to download Sentence Transformers model {model_name}: {e}"
            )


def download_fastembed_models(models):
    try:
        from fastembed import TextEmbedding
    except ImportError:
        logging.warning("FastEmbed not available - skipping")
        return
    for model_name in ensure_st_prefix(models):
        try:
            TextEmbedding(model_name)
            logging.info(f"Downloaded FastEmbed model: {model_name}")
        except Exception as e:
            logging.warning(f"Failed to download FastEmbed model {model_name}: {e}")

    cache_dir = Path(
        os.environ.get(
            "FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed")
        )
    )
    populate_fastembed_gcs_cache(cache_dir)


def populate_fastembed_gcs_cache(cache_dir: Path):
    """Populate fastembed GCS-format cache from HF-format cache.

    fastembed has two download paths: HuggingFace and Google Cloud Storage.
    During image build, models are downloaded via HF and cached in HF format
    (models--org--name/snapshots/<hash>/). At runtime with HF_HUB_OFFLINE=1,
    the HF path fails and fastembed falls back to checking for GCS-format
    directories (fast-<name>/). If these don't exist, it downloads ~80-90MB
    from storage.googleapis.com — which fails in air-gapped environments.

    This function copies model files from the HF cache snapshots into the
    GCS-format directories so both cache paths are satisfied at build time.
    """
    import shutil

    try:
        from fastembed import TextEmbedding
    except ImportError:
        return

    hf_to_model = {}
    for desc in TextEmbedding.list_supported_models():
        if desc.sources and desc.sources.hf:
            hf_to_model[desc.sources.hf] = desc.model

    for hf_dir in sorted(cache_dir.glob("models--*")):
        if not hf_dir.is_dir():
            continue

        hf_repo = "/".join(hf_dir.name.split("--")[1:])
        model_name = hf_to_model.get(hf_repo)
        if not model_name:
            logging.info(f"No fastembed model mapping for {hf_repo}, skipping")
            continue

        fast_name = f"fast-{model_name.split('/')[-1]}"
        fast_dir = cache_dir / fast_name

        if fast_dir.exists() and any(fast_dir.iterdir()):
            logging.info(f"GCS cache already populated: {fast_dir}")
            continue

        snapshots = hf_dir / "snapshots"
        if not snapshots.exists():
            continue

        snapshot_dirs = sorted(
            snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not snapshot_dirs:
            continue
        snapshot = snapshot_dirs[0]

        fast_dir.mkdir(parents=True, exist_ok=True)
        for item in snapshot.iterdir():
            dest = fast_dir / item.name
            src = item.resolve() if item.is_symlink() else item
            if src.is_file():
                shutil.copy2(str(src), str(dest))
            elif src.is_dir():
                shutil.copytree(str(src), str(dest))

        file_count = sum(1 for _ in fast_dir.iterdir())
        total_size = sum(f.stat().st_size for f in fast_dir.rglob("*") if f.is_file())
        logging.info(
            f"Populated GCS cache: {fast_dir} "
            f"({file_count} files, {total_size / 1024 / 1024:.1f} MB) "
            f"from {hf_repo}"
        )

    cache_dir = Path(
        os.environ.get(
            "FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed")
        )
    )
    populate_fastembed_gcs_cache(cache_dir)


def populate_fastembed_gcs_cache(cache_dir: Path):
    """Populate fastembed GCS-format cache from HF-format cache.

    fastembed has two download paths: HuggingFace and Google Cloud Storage.
    During image build, models are downloaded via HF and cached in HF format
    (models--org--name/snapshots/<hash>/). At runtime with HF_HUB_OFFLINE=1,
    the HF path fails and fastembed falls back to checking for GCS-format
    directories (fast-<name>/). If these don't exist, it downloads ~80-90MB
    from storage.googleapis.com — which fails in air-gapped environments.

    This function copies model files from the HF cache snapshots into the
    GCS-format directories so both cache paths are satisfied at build time.
    """
    import shutil

    try:
        from fastembed import TextEmbedding
    except ImportError:
        return

    hf_to_model = {}
    for desc in TextEmbedding.list_supported_models():
        if desc.sources and desc.sources.hf:
            hf_to_model[desc.sources.hf] = desc.model

    for hf_dir in sorted(cache_dir.glob("models--*")):
        if not hf_dir.is_dir():
            continue

        hf_repo = "/".join(hf_dir.name.split("--")[1:])
        model_name = hf_to_model.get(hf_repo)
        if not model_name:
            logging.info(f"No fastembed model mapping for {hf_repo}, skipping")
            continue

        fast_name = f"fast-{model_name.split('/')[-1]}"
        fast_dir = cache_dir / fast_name

        if fast_dir.exists() and any(fast_dir.iterdir()):
            logging.info(f"GCS cache already populated: {fast_dir}")
            continue

        snapshots = hf_dir / "snapshots"
        if not snapshots.exists():
            continue

        snapshot_dirs = sorted(
            snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not snapshot_dirs:
            continue
        snapshot = snapshot_dirs[0]

        fast_dir.mkdir(parents=True, exist_ok=True)
        for item in snapshot.iterdir():
            dest = fast_dir / item.name
            src = item.resolve() if item.is_symlink() else item
            if src.is_file():
                shutil.copy2(str(src), str(dest))
            elif src.is_dir():
                shutil.copytree(str(src), str(dest))

        file_count = sum(1 for _ in fast_dir.iterdir())
        total_size = sum(f.stat().st_size for f in fast_dir.rglob("*") if f.is_file())
        logging.info(
            f"Populated GCS cache: {fast_dir} "
            f"({file_count} files, {total_size / 1024 / 1024:.1f} MB) "
            f"from {hf_repo}"
        )


def download_huggingface_models(models):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        logging.warning("Transformers not available - skipping HuggingFace models")
        return
    try:
        from sentence_transformers import SentenceTransformer

        has_sentence_transformers = True
    except ImportError:
        has_sentence_transformers = False
    for model_name in models:
        if "/" not in model_name:
            logging.info(f"Skipping non-HuggingFace model name: {model_name}")
            continue
        try:
            logging.info(f"Downloading HuggingFace model: {model_name}")
            AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=False,
            )
            logging.info(f"Downloaded HuggingFace model: {model_name}")
        except Exception as e:
            # Some models (like Snowflake arctic-embed) work better with SentenceTransformer
            if has_sentence_transformers and "snowflake" in model_name.lower():
                try:
                    logging.info(f"Retrying with SentenceTransformer: {model_name}")
                    SentenceTransformer(model_name, trust_remote_code=True)
                    logging.info(f"✓ Downloaded via SentenceTransformer: {model_name}")
                except Exception as e2:
                    logging.warning(f"Failed to download {model_name}: {e2}")
            else:
                logging.warning(
                    f"Failed to download HuggingFace model {model_name}: {e}"
                )


def download_nltk_data():
    try:
        import nltk

        cache_dir = os.environ.get("NLTK_DATA")
        nltk.download("punkt", download_dir=cache_dir, quiet=True)
        logging.info("Downloaded NLTK punkt tokenizer")
    except ImportError:
        logging.warning("NLTK not available - skipping")
    except Exception as e:
        logging.warning(f"Failed to download NLTK data: {e}")


def get_models(profile):
    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    from discover_required_models import ModelDiscoverer

    discoverer = ModelDiscoverer(profile)
    return discoverer.discover()


def main():
    profile = os.environ.get("GUARDRAILS_PROFILE", "opensource")
    logging.info(f"Pre-downloading models for profile: {profile}")
    setup_cache()
    models = get_models(profile)
    download_spacy_models(models.get("spacy", []))
    download_sentence_transformers_models(models.get("sentence_transformers", []))
    download_fastembed_models(models.get("sentence_transformers", []))
    download_huggingface_models(models.get("huggingface", []))
    download_nltk_data()
    logging.info("Model download complete")


if __name__ == "__main__":
    main()
