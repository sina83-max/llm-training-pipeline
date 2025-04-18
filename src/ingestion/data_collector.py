import os
import logging
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT/ 'config' / 'config.yaml'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    A class to collect text data for LLM training.

    This class handles downloading data from various sources and
    storing it in the raw data directory.
    """

    def __init__(self, config):
        """
        Initialize the DataCollector with configuration.
        Args:
            config (dict): Configuration dictionary containing data paths
                which is config.yaml file.
        """
        self.raw_data_dir = PROJECT_ROOT / config['data']['raw_dir']
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data will be stored in {self.raw_data_dir}")

    def download_file(self, url, filename):
        """
        Download a file from a URL and save it locally.

        Args:
            url (str): The URL to download from
            filename (str): The filename to save as

        Returns:
            bool: True if successful, False otherwise
        """
        output_path = self.raw_data_dir / filename

        # Don't download if the file exists
        if output_path.exists():
            logger.info(f"File {filename} exists. Skipping download.")
            return True

        try:
            logger.info(f"Downloading from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise exception for errors

            with open(output_path, 'wb') as f:
                # Process each chunk (e.g., save to file)
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                logger.info(f"Successfully download {filename}")
                return True
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return False

    def download_sample_text_dataset(self):
        """
        Download a small sample text dataset for LLM training.
        For MVP purposes, we'll use a small public domain text dataset.

        Returns:
            bool: True if successful, False otherwise
        """
        # For demonstration, let's download a few classic books from Project Gutenberg
        urls = [
            ('https://www.gutenberg.org/files/1342/1342-0.txt', 'pride_and_prejudice.txt'),
            ('https://www.gutenberg.org/files/84/84-0.txt', 'frankenstein.txt'),
            ('https://www.gutenberg.org/files/1661/1661-0.txt', 'sherlock_holmes.txt'),
        ]

        success = True
        for url, filename in urls:
            if not self.download_file(url, filename):
                success = False

        return success

    def list_collected_files(self):
        """
        List all files that have been collected.

        Returns:
            list: List of filenames in the raw data directory
        """
        return [f.name for f in self.raw_data_dir.glob('*') if f.is_file()]


if __name__ == "__main__":
    import yaml

    # Load configurations
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

        collector = DataCollector(config)

        collector.download_sample_text_dataset()

        files = collector.list_collected_files()
        logger.info(f'Collected files: {files}')
