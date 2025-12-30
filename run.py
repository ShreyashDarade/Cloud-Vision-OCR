"""
Cloud Vision OCR - Run Script

Start both API server and Streamlit frontend with a single command.

Usage:
    python run.py          # Start both API and Frontend
    python run.py api      # Start only API
    python run.py frontend # Start only Frontend
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

# Ensure we're in the project directory
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class ServiceManager:
    """Manage multiple services."""
    
    def __init__(self):
        self.processes = []
        self.running = True
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n🛑 Shutting down services...")
        self.running = False
        self.stop_all()
        sys.exit(0)
    
    def start_api(self, port: int = 8000):
        """Start the FastAPI server."""
        print(f"🚀 Starting API server on http://localhost:{port}")
        print(f"   📚 API Docs: http://localhost:{port}/docs")
        
        process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "src.api.main:app",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--reload"
            ],
            cwd=PROJECT_ROOT
        )
        self.processes.append(("API", process))
        return process
    
    def start_frontend(self, port: int = 8501):
        """Start the Streamlit frontend."""
        print(f"🎨 Starting Frontend on http://localhost:{port}")
        
        process = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run",
                "src/frontend/app.py",
                "--server.port", str(port),
                "--server.address", "0.0.0.0",
                "--server.headless", "true"
            ],
            cwd=PROJECT_ROOT
        )
        self.processes.append(("Frontend", process))
        return process
    
    def stop_all(self):
        """Stop all running processes."""
        for name, process in self.processes:
            if process.poll() is None:
                print(f"   Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        self.processes.clear()
    
    def wait_for_all(self):
        """Wait for all processes to complete."""
        while self.running:
            # Check if any process has died
            for name, process in self.processes:
                if process.poll() is not None:
                    print(f"⚠️  {name} process exited with code {process.returncode}")
            time.sleep(1)


def print_banner():
    """Print startup banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              ☁️  Cloud Vision OCR Pipeline                    ║
║                  Production-Grade OCR                         ║
╠═══════════════════════════════════════════════════════════════╣
║  API:      http://localhost:8000                              ║
║  Docs:     http://localhost:8000/docs                         ║
║  Frontend: http://localhost:8501                              ║
╠═══════════════════════════════════════════════════════════════╣
║  Press Ctrl+C to stop all services                            ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cloud Vision OCR Runner")
    parser.add_argument(
        "service",
        nargs="?",
        choices=["api", "frontend", "all"],
        default="all",
        help="Service to run (default: all)"
    )
    parser.add_argument("--api-port", type=int, default=8000, help="API port")
    parser.add_argument("--frontend-port", type=int, default=8501, help="Frontend port")
    
    args = parser.parse_args()
    
    manager = ServiceManager()
    
    try:
        if args.service in ["api", "all"]:
            manager.start_api(port=args.api_port)
            time.sleep(2)  # Wait for API to start
        
        if args.service in ["frontend", "all"]:
            manager.start_frontend(port=args.frontend_port)
        
        if args.service == "all":
            print_banner()
        
        # Keep running
        manager.wait_for_all()
        
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all()


if __name__ == "__main__":
    main()
