"""
Command-line interface for Cloud Vision OCR.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Cloud Vision OCR - Production-grade OCR pipeline."""
    pass


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file path (default: stdout)"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["json", "text", "hocr"]),
    default="text",
    help="Output format"
)
@click.option(
    "--detection-type", "-d",
    type=click.Choice(["TEXT_DETECTION", "DOCUMENT_TEXT_DETECTION"]),
    default="DOCUMENT_TEXT_DETECTION",
    help="Vision API detection type"
)
@click.option(
    "--language", "-l",
    multiple=True,
    help="Language hints (can be specified multiple times)"
)
@click.option(
    "--no-preprocessing",
    is_flag=True,
    help="Disable image preprocessing"
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable result caching"
)
def process(
    image_path: str,
    output: Optional[str],
    format: str,
    detection_type: str,
    language: tuple,
    no_preprocessing: bool,
    no_cache: bool
):
    """Process a single image for OCR."""
    from src.observability.logging import setup_logging
    from src.preprocessing import PreprocessingPipeline
    from src.vision import VisionClient
    from src.cache import get_cache
    
    setup_logging(log_format="console")
    
    image_path = Path(image_path)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Load image
        task = progress.add_task("Loading image...", total=None)
        
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        progress.update(task, description="Loading image... ✓")
        
        # Check cache
        if not no_cache:
            cache = get_cache()
            content_hash = cache.compute_hash(image_bytes)
            cached = cache.get(content_hash)
            
            if cached:
                progress.update(task, description="Found in cache! ✓")
                result_text = cached.get("text", "")
                _output_result(result_text, cached, format, output)
                return
        
        # Preprocessing
        if not no_preprocessing:
            progress.update(task, description="Preprocessing image...")
            pipeline = PreprocessingPipeline()
            image_bytes = pipeline.process_bytes(image_bytes)
            progress.update(task, description="Preprocessing image... ✓")
        
        # OCR
        progress.update(task, description="Calling Vision API...")
        
        with VisionClient() as client:
            result = client.detect_text(
                image_bytes,
                detection_type=detection_type,
                language_hints=list(language) if language else None
            )
        
        progress.update(task, description="Calling Vision API... ✓")
        
        # Cache result
        if not no_cache:
            cache.set(content_hash, result.to_dict())
    
    # Output
    _output_result(result.text, result.to_dict(), format, output)
    
    # Summary
    console.print()
    table = Table(title="OCR Results", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Characters", f"{result.char_count:,}")
    table.add_row("Words", f"{result.word_count:,}")
    table.add_row("Confidence", f"{result.confidence:.2%}")
    table.add_row("Language", result.language or "Unknown")
    
    console.print(table)


def _output_result(text: str, result: dict, format: str, output: Optional[str]):
    """Output the result in the specified format."""
    if format == "json":
        content = json.dumps(result, indent=2, ensure_ascii=False)
    elif format == "hocr":
        from src.vision.response_parser import OCRResult
        # Reconstruct for hOCR
        content = text  # Simplified - full hOCR requires result object
    else:
        content = text
    
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[green]✓[/green] Output saved to {output}")
    else:
        console.print(Panel(content, title="Extracted Text", expand=False))


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    required=True,
    help="Output directory for results"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format"
)
@click.option(
    "--detection-type", "-d",
    type=click.Choice(["TEXT_DETECTION", "DOCUMENT_TEXT_DETECTION"]),
    default="DOCUMENT_TEXT_DETECTION",
    help="Vision API detection type"
)
@click.option(
    "--language", "-l",
    multiple=True,
    help="Language hints"
)
@click.option(
    "--no-preprocessing",
    is_flag=True,
    help="Disable image preprocessing"
)
@click.option(
    "--workers", "-w",
    type=int,
    default=4,
    help="Number of parallel workers"
)
def batch(
    input_dir: str,
    output_dir: str,
    format: str,
    detection_type: str,
    language: tuple,
    no_preprocessing: bool,
    workers: int
):
    """Process multiple images in a directory."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.observability.logging import setup_logging
    from src.preprocessing import PreprocessingPipeline
    from src.vision import VisionClient
    
    setup_logging(log_format="console")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find images
    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
    images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
    
    if not images:
        console.print("[yellow]No images found in directory[/yellow]")
        return
    
    console.print(f"Found [cyan]{len(images)}[/cyan] images to process")
    
    pipeline = PreprocessingPipeline() if not no_preprocessing else None
    lang_hints = list(language) if language else None
    
    results = {"success": 0, "failed": 0}
    
    def process_image(image_file: Path):
        try:
            with open(image_file, "rb") as f:
                content = f.read()
            
            if pipeline:
                content = pipeline.process_bytes(content)
            
            with VisionClient() as client:
                result = client.detect_text(
                    content,
                    detection_type=detection_type,
                    language_hints=lang_hints
                )
            
            # Save result
            ext = ".json" if format == "json" else ".txt"
            output_file = output_path / f"{image_file.stem}{ext}"
            
            if format == "json":
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            else:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(result.text)
            
            return True, image_file.name
        except Exception as e:
            return False, f"{image_file.name}: {str(e)}"
    
    with Progress(console=console) as progress:
        task = progress.add_task("Processing images...", total=len(images))
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_image, img): img for img in images}
            
            for future in as_completed(futures):
                success, message = future.result()
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    console.print(f"[red]✗[/red] {message}")
                
                progress.advance(task)
    
    # Summary
    console.print()
    console.print(Panel(
        f"[green]Successful:[/green] {results['success']}\n"
        f"[red]Failed:[/red] {results['failed']}\n"
        f"[cyan]Output:[/cyan] {output_path}",
        title="Batch Processing Complete"
    ))


@cli.command()
@click.option(
    "--host", "-h",
    default="0.0.0.0",
    help="Host to bind to"
)
@click.option(
    "--port", "-p",
    default=8000,
    type=int,
    help="Port to bind to"
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development"
)
def serve(host: str, port: int, reload: bool):
    """Start the OCR API server."""
    import uvicorn
    
    console.print(Panel(
        f"Starting Cloud Vision OCR API\n"
        f"[cyan]URL:[/cyan] http://{host}:{port}\n"
        f"[cyan]Docs:[/cyan] http://{host}:{port}/docs",
        title="🚀 Server Starting"
    ))
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@cli.command()
@click.option(
    "--port", "-p",
    default=8501,
    type=int,
    help="Port to bind to"
)
def ui(port: int):
    """Start the Streamlit frontend."""
    import subprocess
    
    console.print(Panel(
        f"Starting Streamlit UI\n"
        f"[cyan]URL:[/cyan] http://localhost:{port}",
        title="🎨 UI Starting"
    ))
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/frontend/app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0"
    ])


if __name__ == "__main__":
    cli()
