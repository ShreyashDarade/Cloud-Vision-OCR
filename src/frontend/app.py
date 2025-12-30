"""
Production-grade Streamlit frontend for Cloud Vision OCR.

This frontend communicates exclusively with the FastAPI backend via REST API.
Run with: streamlit run src/frontend/app.py
"""

import io
import time
import requests
from typing import Optional
from PIL import Image

import streamlit as st

# Configuration
API_BASE_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Cloud Vision OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #a0aec0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    /* Status badges */
    .badge-success {
        background: linear-gradient(90deg, #00b09b, #96c93d);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .badge-error {
        background: linear-gradient(90deg, #e53e3e, #c53030);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    
    /* Metric cards */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #a0aec0;
        font-size: 0.9rem;
    }
    
    /* Upload area */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px dashed rgba(102, 126, 234, 0.5) !important;
        border-radius: 12px !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(26, 26, 46, 0.95);
    }
    
    /* Text area */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #e2e8f0;
    }
    
    /* Select box */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


class OCRClient:
    """Client for communicating with the OCR API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> dict:
        """Check API health."""
        try:
            response = self.session.get(f"{self.base_url}/health/ready", timeout=5)
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def process_image(
        self,
        image_bytes: bytes,
        filename: str,
        detection_type: str = "DOCUMENT_TEXT_DETECTION",
        language_hints: Optional[str] = None,
        preprocessing_enabled: bool = True,
        output_format: str = "json"
    ) -> dict:
        """Process a single image via the sync endpoint."""
        try:
            files = {"file": (filename, image_bytes)}
            data = {
                "detection_type": detection_type,
                "preprocessing_enabled": preprocessing_enabled,
                "output_format": output_format
            }
            if language_hints:
                data["language_hints"] = language_hints
            
            response = self.session.post(
                f"{self.base_url}/ocr/sync",
                files=files,
                data=data,
                timeout=120
            )
            
            if output_format == "json":
                return response.json()
            else:
                return {
                    "success": True,
                    "text": response.text,
                    "raw_output": True
                }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def submit_batch(
        self,
        images: list,
        detection_type: str = "DOCUMENT_TEXT_DETECTION",
        language_hints: Optional[str] = None,
        preprocessing_enabled: bool = True
    ) -> dict:
        """Submit batch for async processing."""
        try:
            files = [("files", (img["name"], img["bytes"])) for img in images]
            data = {
                "detection_type": detection_type,
                "preprocessing_enabled": preprocessing_enabled
            }
            if language_hints:
                data["language_hints"] = language_hints
            
            response = self.session.post(
                f"{self.base_url}/ocr/batch",
                files=files,
                data=data,
                timeout=30
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def get_job_status(self, job_id: str) -> dict:
        """Get async job status."""
        try:
            response = self.session.get(
                f"{self.base_url}/ocr/jobs/{job_id}",
                timeout=10
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": str(e)}
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        try:
            response = self.session.get(f"{self.base_url}/cache/stats", timeout=5)
            return response.json()
        except:
            return {}


# Initialize client
@st.cache_resource
def get_client():
    return OCRClient(API_BASE_URL)


def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">🔍 Cloud Vision OCR</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Production-grade OCR powered by Google Cloud Vision API</p>',
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render the sidebar with settings."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # API Status
        client = get_client()
        health = client.health_check()
        
        if health.get("status") == "healthy":
            st.success("🟢 API Connected")
        elif health.get("status") == "degraded":
            st.warning("🟡 API Degraded")
        else:
            st.error("🔴 API Unavailable")
        
        st.markdown("---")
        
        # OCR Settings
        st.markdown("### Detection Settings")
        
        detection_type = st.selectbox(
            "Detection Type",
            options=["DOCUMENT_TEXT_DETECTION", "TEXT_DETECTION"],
            index=0,
            help="DOCUMENT_TEXT_DETECTION is optimized for documents with dense text"
        )
        
        language_hints = st.text_input(
            "Language Hints",
            placeholder="en,hi,mr",
            help="Comma-separated language codes to prioritize"
        )
        
        preprocessing_enabled = st.checkbox(
            "Enable Preprocessing",
            value=True,
            help="Apply image enhancements for better accuracy"
        )
        
        st.markdown("---")
        
        # Output Settings
        st.markdown("### Output Settings")
        
        output_format = st.selectbox(
            "Output Format",
            options=["json", "text", "hocr"],
            index=0
        )
        
        st.markdown("---")
        
        # Cache Stats
        with st.expander("📊 Cache Statistics"):
            stats = client.get_cache_stats()
            if stats:
                st.metric("Memory Cache Size", stats.get("memory_cache_size", 0))
                st.metric("Redis Available", "✓" if stats.get("redis_available") else "✗")
        
        return {
            "detection_type": detection_type,
            "language_hints": language_hints if language_hints else None,
            "preprocessing_enabled": preprocessing_enabled,
            "output_format": output_format
        }


def render_metrics(result: dict):
    """Render OCR result metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{result.get('char_count', 0):,}</div>
            <div class="metric-label">Characters</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{result.get('word_count', 0):,}</div>
            <div class="metric-label">Words</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        confidence = result.get('confidence', 0) * 100
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{confidence:.1f}%</div>
            <div class="metric-label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        time_ms = result.get('processing_time_ms', 0)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{time_ms:.0f}ms</div>
            <div class="metric-label">Processing Time</div>
        </div>
        """, unsafe_allow_html=True)


def render_single_image_tab(settings: dict):
    """Render single image processing tab."""
    client = get_client()
    
    st.markdown("### 📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "tiff", "tif", "bmp", "gif", "webp"],
        help="Supported formats: PNG, JPEG, TIFF, BMP, GIF, WEBP"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Preview")
            image = Image.open(uploaded_file)
            st.image(image, width='stretch')
            st.caption(f"Size: {image.width}×{image.height} | Format: {image.format}")
        
        with col2:
            st.markdown("#### Actions")
            
            if st.button("🚀 Extract Text", type="primary"):
                with st.spinner("Processing image..."):
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    
                    result = client.process_image(
                        image_bytes=image_bytes,
                        filename=uploaded_file.name,
                        detection_type=settings["detection_type"],
                        language_hints=settings["language_hints"],
                        preprocessing_enabled=settings["preprocessing_enabled"],
                        output_format=settings["output_format"]
                    )
                    
                    st.session_state["last_result"] = result
    
    # Display results
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        
        st.markdown("---")
        st.markdown("### 📝 Results")
        
        if result.get("success", False):
            if result.get("raw_output"):
                st.text_area("Output", result.get("text", ""), height=400)
            else:
                # Show metrics
                render_metrics(result)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["📄 Text", "📊 Structured Data", "📋 Raw JSON"])
                
                with tab1:
                    text = result.get("text", "")
                    st.text_area(
                        "Extracted Text",
                        text,
                        height=300,
                        label_visibility="collapsed"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download as TXT",
                            text,
                            file_name="ocr_result.txt",
                            mime="text/plain"
                        )
                
                with tab2:
                    if result.get("pages"):
                        for i, page in enumerate(result["pages"]):
                            with st.expander(f"Page {i+1} ({page.get('width', 0)}×{page.get('height', 0)})"):
                                for j, block in enumerate(page.get("blocks", [])):
                                    st.markdown(f"**Block {j+1}** (Confidence: {block.get('confidence', 0):.2%})")
                                    st.text(block.get("text", ""))
                    else:
                        st.info("Detailed structure not available for TEXT_DETECTION mode")
                
                with tab3:
                    import json
                    st.json(result)
                
                # Cache status
                if result.get("cached"):
                    st.info("⚡ Result served from cache")
                
                # Language detected
                if result.get("language"):
                    st.caption(f"Detected language: {result['language']}")
        
        else:
            st.error(f"❌ OCR Failed: {result.get('error', 'Unknown error')}")


def render_batch_tab(settings: dict):
    """Render batch processing tab."""
    client = get_client()
    
    st.markdown("### 📤 Upload Multiple Images")
    
    uploaded_files = st.file_uploader(
        "Choose images",
        type=["png", "jpg", "jpeg", "tiff", "tif", "bmp", "gif", "webp"],
        accept_multiple_files=True,
        help="Select multiple images for batch processing"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} images selected**")
        
        # Preview thumbnails
        cols = st.columns(min(len(uploaded_files), 5))
        for i, file in enumerate(uploaded_files[:5]):
            with cols[i]:
                image = Image.open(file)
                st.image(image, width='stretch', caption=file.name[:20])
        
        if len(uploaded_files) > 5:
            st.caption(f"...and {len(uploaded_files) - 5} more")
        
        if st.button("🚀 Process Batch", type="primary"):
            # Submit batch job
            images = []
            for file in uploaded_files:
                file.seek(0)
                images.append({
                    "name": file.name,
                    "bytes": file.read()
                })
            
            with st.spinner("Submitting batch job..."):
                job = client.submit_batch(
                    images=images,
                    detection_type=settings["detection_type"],
                    language_hints=settings["language_hints"],
                    preprocessing_enabled=settings["preprocessing_enabled"]
                )
            
            if job.get("job_id"):
                st.session_state["batch_job_id"] = job["job_id"]
                st.success(f"Batch job submitted! Job ID: {job['job_id']}")
            else:
                st.error(f"Failed to submit batch: {job.get('error', 'Unknown error')}")
    
    # Poll for job status
    if "batch_job_id" in st.session_state:
        job_id = st.session_state["batch_job_id"]
        
        st.markdown("---")
        st.markdown("### 📊 Job Status")
        
        if st.button("🔄 Refresh Status"):
            pass  # Just triggers a rerun
        
        status = client.get_job_status(job_id)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", status.get("status", "unknown").upper())
        with col2:
            st.metric("Progress", f"{status.get('progress', 0) * 100:.0f}%")
        with col3:
            st.metric("Processed", f"{status.get('processed_images', 0)}/{status.get('total_images', 0)}")
        
        if status.get("status") == "processing":
            st.progress(status.get("progress", 0))
        
        if status.get("status") == "completed" and status.get("result"):
            st.markdown("### Results")
            
            for i, result in enumerate(status["result"]):
                with st.expander(f"Image {i+1} - {'✓' if result.get('success') else '✗'}"):
                    if result.get("success"):
                        st.text_area(
                            "Text",
                            result.get("text", ""),
                            height=150,
                            key=f"batch_result_{i}"
                        )
                        st.caption(f"Words: {result.get('word_count', 0)} | Confidence: {result.get('confidence', 0):.2%}")
                    else:
                        st.error(result.get("error", "Failed"))
            
            # Download all results
            import json
            results_json = json.dumps(status["result"], indent=2)
            st.download_button(
                "📥 Download All Results (JSON)",
                results_json,
                file_name="batch_results.json",
                mime="application/json"
            )
        
        if status.get("status") == "failed":
            st.error(f"Batch job failed: {status.get('error', 'Unknown error')}")


def render_url_tab(settings: dict):
    """Render URL input tab."""
    st.markdown("### 🌐 Process Image from URL")
    
    st.info("This feature requires backend support for URL fetching. Currently, please download the image and use the upload feature.")
    
    url = st.text_input("Image URL", placeholder="https://example.com/image.png")
    
    if url and st.button("🚀 Process URL"):
        st.warning("URL processing not yet implemented. Please use file upload.")


def main():
    """Main application entry point."""
    render_header()
    settings = render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📄 Single Image", "📚 Batch Processing", "🌐 URL"])
    
    with tab1:
        render_single_image_tab(settings)
    
    with tab2:
        render_batch_tab(settings)
    
    with tab3:
        render_url_tab(settings)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #a0aec0; font-size: 0.85rem;">
            Built with ❤️ using Streamlit & Google Cloud Vision API
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
