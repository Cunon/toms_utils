import whisper
import json
from pathlib import Path
from typing import Union, List, Optional

def transcribe_media(
    source: Union[str, Path],
    output_dir: Union[str, Path] = "transcriptions",
    model_size: str = "large",
    extensions: List[str] = None,
    context_prompt: Optional[str] = None,
    **whisper_kwargs
) -> None:
    """
    Transcribe a single file or a directory of media files using OpenAI Whisper.

    Args:
        source: Path to a specific file OR a directory containing media.
        output_dir: Directory to save the resulting JSON/TXT files.
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
        extensions: List of extensions to scan for IF source is a directory (default: mp4, mp3, etc).
        context_prompt: A string to guide the model style or vocabulary.
        **whisper_kwargs: Extra arguments passed to model.transcribe (e.g. fp16=False).
    """
    # 1. Setup Paths
    source_path = Path(source).resolve()
    output_path = Path(output_dir).resolve()
    
    # 2. Gather files to process
    files_to_process = []
    
    if source_path.is_file():
        # If user explicitly points to a file, process it regardless of extension
        files_to_process.append(source_path)
        print(f"Processing single file: {source_path.name}")
        
    elif source_path.is_dir():
        # If directory, scan for supported extensions
        if extensions is None:
            extensions = [".mp4", ".mkv", ".mp3", ".wav", ".m4a", ".flac"]
            
        print(f"Scanning directory: {source_path}")
        for ext in extensions:
            # Case-insensitive search requires more complex globbing on Linux, 
            # keeping it simple here for standard extensions.
            files_to_process.extend(source_path.glob(f"*{ext}"))
            
    else:
        print(f"Error: Source path '{source_path}' does not exist.")
        return

    if not files_to_process:
        print("No media files found to process.")
        return

    # 3. Create Output Directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 4. Load Model
    print(f"Loading Whisper model ('{model_size}')...")
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 5. Process Loop
    for media_file in files_to_process:
        print(f"\nTranscribing: {media_file.name}")
        try:
            # Prepare arguments
            transcribe_args = {
                "audio": str(media_file),
                **whisper_kwargs 
            }
            
            # Inject prompt if provided
            if context_prompt:
                transcribe_args["initial_prompt"] = context_prompt

            # Transcribe
            result = model.transcribe(**transcribe_args)
            
            # Save JSON
            json_file = output_path / f"{media_file.stem}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Optional: Also save plain text for quick reading
            txt_file = output_path / f"{media_file.stem}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(result['text'].strip())
            
            print(f"✓ Saved to {json_file.name}")
            
        except Exception as e:
            print(f"✗ Failed {media_file.name}: {e}")

# Example Usage
if __name__ == "__main__":
    # Example 1: Process a single file
    # transcribe_media("my_recording.wav", model_size="base")
    
    # Example 2: Process a directory with custom config
    transcribe_media(
        source=".",  # Current directory
        model_size="small",
        fp16=False,
        context_prompt="This is a discussion about Dungeons and Dragons, ESP32, and Home Assistant."
    )