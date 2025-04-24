"""
Preprocessing script for raw texts
"""

import os
import re
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT/ 'config' / 'config.yaml'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Class for preprocessing text data for training LLM.

    Handles cleaning and simple preprocessing actions.
    """
    def __init__(self, config):
        """
        Initialize preprocessor with config.

        :param config: Configurations dictionary containing config data
        """
        self.raw_data_dir = PROJECT_ROOT / config['data']['raw_dir']
        self.processed_data_dir = PROJECT_ROOT / config['data']['processed_dir']
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Load preprocessing options from config
        self.max_length = config['preprocessing'].get('max_length', None)
        self.lowercase = config['preprocessing'].get('lowercase', False)
        self.remove_special_chars = config['preprocessing'].get(
            'remove_special_chars', False)

        logger.info(
            f"Text Processor initialized with config: {config['preprocessing']}")

    def clean_text(self, text):
        """
        Preprocess the data with simple functions.

        :param text: raw text (str)

        :return: preprocessed text (str)
        """
        # Apply lowercase if configured
        if self.lowercase:
            text = text.lower()

        # Remove special characters if configured
        if self.remove_special_chars:
            text = re.sub(r'[^\w\s.,!?]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Truncate if max_length is set
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length]

        return text

    def process_file(self, filename):
        """
        Process a single file.

        :param filename: Name of the file to process.

        :return: (bool) Either True or File.
        """
        input_path = self.raw_data_dir / filename
        output_path = self.processed_data_dir / f"processed_{filename}"

        try:
            logger.info(f"processing file: {filename}")

            # read input file
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            # clean the text
            cleaned_text = self.clean_text(text)

            # Write the text into output
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            logger.info(f"Successfully processed file: {filename}")

            return  True

        except Exception as e:
            logger.error(f"Error processing file: {filename}")
            return False

    def split_into_chunks(self, text, chunk_size=1000, overlap=100):
        """
        Split the text into overlapping chunks of the specified size.
        :param text (Str): text to split
        :param chunk_size (Int): Size of each chunk in characters
        :param overlap (Int): Number of characters to overlap between chunks

        :return (List): List of each chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Find the nearest .,!,? to end on
            if end < len(text):
                # Look for sentence boundaries within 100 chars of the end
                search_area = text[max(end - 100, start):min(end + 100, len(text))]
                sentence_ends = [m.start() for m in re.finditer(r'[.!?]', search_area)]

                if sentence_ends:
                    # Find the closest sentence to our target endpoint
                    closest_end = min(sentence_ends, key=lambda x: abs(x - 100))
                    end = max(end - 100, start) + closest_end + 1

            chunks.append(text[start:end].strip())
            start = end - overlap

        return chunks

    def process_all_files(self):
        """
        Process all files in the raw data directory

        :return:
            int: Number of successfully processed files.
        """
        files = [f.name for f in self.raw_data_dir.glob("*")
                    if f.is_file() and not f.name.startswith(".")]

        success_count = 0

        for filename in files:
            if self.process_file(filename):
                success_count += 1

        logger.info(f"Successfully processed {success_count} out of {len(files)} files")
        return success_count

    def create_training_chunks(self, filename, output_filename, chunk_size=1000, overlap=100):
        """
        Create overlapping chunks of text for training.

        :param filename: str => Input filename
        :param output_filename: str => Output filename
        :param chunk_size: int => Size of each chunk
        :param overlap: int => Overlap between chunks

        :return:
            int => Number of created chunks
        """
        input_path = self.processed_data_dir / f'processed_{filename}'
        output_path = self.processed_data_dir / output_filename

        if not input_path.exists():
            logger.error(f"Processed file {input_path} not found")
            return 0

        try:
            chunk_count = 0
            buffer = ""

            # Read processed files.
            with open(input_path, 'r', encoding='utf-8') as fin, \
                 open(output_path, 'w', encoding='utf-8') as fout:

                while True:
                    # Read in small chunks (100kb)
                    data = fin.read(102_400)
                    if not data:
                        if buffer.strip():
                            fout.write(buffer.strip() + '\n\n---\n\n')
                            chunk_count +=1
                        break

                    buffer += data

                    while True:
                        # Find safe split point
                        chunk, remaining = self._find_chunk_split(buffer, chunk_size, overlap)
                        if not chunk:
                            break

                        fout.write(chunk + '\n\n---\n\n')
                        chunk_count += 1
                        buffer = remaining  # Keep overlap for next chunk

                    logger.info(f'Created {chunk_count} chunks from {filename}')
                    return chunk_count

        except Exception as e:
            logger.error(f"Chunking failed: {str(e)}", exc_info=True)
            return 0

    def _find_chunk_split(self, text, chunk_size, overlap):
        """Finds the best split point in text buffer"""
        if len(text) < chunk_size // 2:  # Wait for more data
            return None, text

        target_end = min(chunk_size, len(text))
        search_start = max(0, target_end - 200)
        search_end = min(len(text), target_end + 200)

        # Find nearest sentence end
        match = None
        for m in re.finditer(r'[.!?]', text[search_start:search_end]):
            if not match or abs(m.start() - (target_end - search_start)) < abs(
                    match.start() - (target_end - search_start)):
                match = m

        if match:
            actual_end = search_start + match.start() + 1
            chunk = text[:actual_end].strip()
            remaining = text[max(actual_end - overlap, 0):]
            return chunk, remaining

        # Fallback: split at chunk_size if no punctuation found
        if len(text) >= chunk_size:
            chunk = text[:chunk_size].strip()
            remaining = text[chunk_size - overlap:]
            return chunk, remaining

        return None, text


# CLI to testing the module
if __name__ == "__main__":
    import yaml

    # Load the configuration
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize the processor
    processor = TextProcessor(config)

    # Process all files
    processor.process_all_files()

    # Create training chunks for one file
    processor.create_training_chunks('pride_and_prejudice.txt', 'training_chunks.txt')











