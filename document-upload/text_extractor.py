from pathlib import Path


class TextExtractor:
    def extract(self, file_path: Path) -> str:
        if file_path.suffix == ".txt":
            return file_path.read_bytes().decode("utf-8")

        raise ValueError(f"Unsupported file type: {file_path.suffix}")
