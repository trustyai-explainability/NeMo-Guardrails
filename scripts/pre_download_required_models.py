#!/usr/bin/env python3
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


def download_huggingface_models(models):
    try:
        from transformers import AutoModel
    except ImportError:
        logging.warning("Transformers not available - skipping HuggingFace models")
        return
    for model_name in models:
        if "/" not in model_name:
            logging.info(f"Skipping non-HuggingFace model name: {model_name}")
            continue
        try:
            AutoModel.from_pretrained(model_name, trust_remote_code=True)
            logging.info(f"Downloaded HuggingFace model: {model_name}")
        except Exception as e:
            logging.warning(f"Failed to download HuggingFace model {model_name}: {e}")


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
