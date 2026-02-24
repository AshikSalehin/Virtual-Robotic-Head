import os
import asyncio
import json
import base64
import subprocess
import logging
import re
import time
import hashlib
import tempfile
import wave
from typing import Dict, Any, List, Optional, Tuple
from contextlib import asynccontextmanager
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vr")

# ============================================================================
# TTS Generator - Using Working say command from original code
# ============================================================================

class TTSGenerator:
    """TTS that generates audio for frontend playback using the working say command"""
    
    def __init__(self):
        self.expression_params = {
            "neutral": {"voice": "Alex", "rate": 180},
            "happy": {"voice": "Alex", "rate": 190},
            "sad": {"voice": "Fred", "rate": 140},
            "angry": {"voice": "Bruce", "rate": 200},
            "thinking": {"voice": "Victoria", "rate": 150},
            "surprised": {"voice": "Alex", "rate": 220},
            "tensed": {"voice": "Bruce", "rate": 190},
            "disgust": {"voice": "Fred", "rate": 160},
        }
        self.audio_cache = {}
        
    def generate_audio(self, text: str, expression: str = "neutral") -> str:
        """Generate audio using say command and return base64 string"""
        try:
            params = self.expression_params.get(expression, self.expression_params["neutral"])
            
            # Clean text
            clean_text = re.sub(r'\[.*?\]', '', text).strip()
            if not clean_text or len(clean_text) < 2:
                return ""
            
            # Create cache key
            cache_key = f"{expression}_{hashlib.md5(clean_text.encode()).hexdigest()[:16]}"
            
            # Check cache first
            if cache_key in self.audio_cache:
                logger.info(f"♻️ Using cached audio for: '{clean_text[:30]}...'")
                return self.audio_cache[cache_key]
            
            logger.info(f"🔊 Generating audio: '{clean_text[:30]}...' (expression: {expression})")
            
            # Create temp file for AIFF
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
                temp_aiff = tmp.name
            
            try:
                # Generate AIFF with say command (working method from original)
                cmd = ["say", "-v", params["voice"], "-r", str(params["rate"]), "-o", temp_aiff, clean_text]
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate(timeout=10)  # 10 second timeout
                
                if process.returncode != 0:
                    logger.error(f"say command failed: {stderr.decode()}")
                    return ""
                
                # Convert AIFF to WAV using ffmpeg if available, otherwise use manual conversion
                try:
                    import subprocess as sp
                    # Create temp file for WAV
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                        temp_wav_path = tmp_wav.name
                    
                    # Convert using ffmpeg
                    cmd = ["ffmpeg", "-y", "-i", temp_aiff, "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", temp_wav_path]
                    process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
                    stdout, stderr = process.communicate(timeout=10)
                    
                    if process.returncode != 0:
                        logger.warning(f"ffmpeg conversion failed: {stderr.decode()[:100]}")
                        # Fallback: use sox if available
                        cmd = ["sox", temp_aiff, "-r", "22050", "-b", "16", "-c", "1", temp_wav_path]
                        process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
                        stdout, stderr = process.communicate(timeout=10)
                
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    # If ffmpeg/sox not available, try pydub
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_file(temp_aiff, format="aiff")
                        audio = audio.set_frame_rate(22050).set_channels(1)
                        
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                            temp_wav_path = tmp_wav.name
                        
                        audio.export(temp_wav_path, format="wav")
                    except ImportError:
                        logger.error("Neither ffmpeg, sox, nor pydub available for audio conversion")
                        # Read raw AIFF data
                        with open(temp_aiff, 'rb') as f:
                            audio_data = f.read()
                        # Encode as base64 directly (frontend might not support AIFF)
                        b64_audio = base64.b64encode(audio_data).decode('utf-8')
                        os.unlink(temp_aiff)
                        self.audio_cache[cache_key] = b64_audio
                        return b64_audio
                
                # Read WAV data
                with open(temp_wav_path, 'rb') as f:
                    audio_data = f.read()
                
                # Cleanup
                os.unlink(temp_aiff)
                if os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
                
                # Encode to base64
                b64_audio = base64.b64encode(audio_data).decode('utf-8')
                
                # Cache it
                self.audio_cache[cache_key] = b64_audio
                
                # Limit cache size
                if len(self.audio_cache) > 100:
                    # Remove oldest entry
                    oldest_key = next(iter(self.audio_cache))
                    del self.audio_cache[oldest_key]
                
                logger.info(f"✅ Generated {len(b64_audio)} bytes of audio")
                return b64_audio
                
            except subprocess.TimeoutExpired:
                logger.error(f"Audio generation timed out for: '{clean_text[:30]}...'")
                if os.path.exists(temp_aiff):
                    os.unlink(temp_aiff)
                return ""
            except Exception as e:
                logger.error(f"Audio generation error: {e}")
                if os.path.exists(temp_aiff):
                    os.unlink(temp_aiff)
                return ""
                
        except Exception as e:
            logger.error(f"Audio generation error: {e}")
            return ""
    
    def clear_cache(self):
        """Clear audio cache"""
        self.audio_cache.clear()
        logger.info("🧹 Audio cache cleared")

# ============================================================================
# Smart LLM with Conversation Memory
# ============================================================================

class SmartLLM:
    """Smart LLM with conversation memory to prevent loops"""
    
    def __init__(self):
        self.conversation_history = []
        self.max_history = 5
        self.last_response_time = 0
        self.response_cooldown = 1
    
    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """Generate response with anti-loop logic"""
        
        current_time = time.time()
        
        # Add to history
        self.conversation_history.append(f"User: {user_input}")
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        # Keyword-based responses
        user_input_lower = user_input.lower()
        
        # Check for specific phrases first
        if any(phrase in user_input_lower for phrase in ['your name', 'who are you']):
            response = "[thinking] I'm your AI assistant! [happy] You can call me Robo."
            expressions = ["thinking", "happy"]
        elif any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            response = "[happy] Hello there! [neutral] I can hear you clearly."
            expressions = ["happy", "neutral"]
        elif any(word in user_input_lower for word in ['how are you', 'how do you do']):
            response = "[happy] I'm functioning perfectly! [neutral] Thanks for asking."
            expressions = ["happy", "neutral"]
        elif any(word in user_input_lower for word in ['what can you do', 'help']):
            response = "[neutral] I can listen to you, [thinking] understand speech, [happy] and respond with expressions!"
            expressions = ["neutral", "thinking", "happy"]
        elif '?' in user_input:
            response = "[thinking] That's a good question! [neutral] Let me think..."
            expressions = ["thinking", "neutral"]
        elif any(word in user_input_lower for word in ['thank', 'thanks']):
            response = "[happy] You're welcome! [neutral] Happy to help."
            expressions = ["happy", "neutral"]
        elif any(word in user_input_lower for word in ['bye', 'goodbye', 'see you']):
            response = "[happy] Goodbye! [neutral] Talk to you again soon."
            expressions = ["happy", "neutral"]
        else:
            # Check if user is repeating robot's last response
            if self.conversation_history:
                last_robot_response = ""
                for i in range(len(self.conversation_history)-1, -1, -1):
                    if self.conversation_history[i].startswith("Assistant:"):
                        last_robot_response = self.conversation_history[i].replace("Assistant:", "").strip().lower()
                        break
                
                # Remove expression tags for comparison
                last_robot_clean = re.sub(r'\[.*?\]', '', last_robot_response).strip().lower()
                user_clean = re.sub(r'[^\w\s]', '', user_input_lower).strip()
                
                if user_clean in last_robot_clean or last_robot_clean in user_clean:
                    response = "[thinking] Yes, I just mentioned that. [neutral] Did you have a question about it?"
                    expressions = ["thinking", "neutral"]
                elif len(user_input.split()) < 3:
                    response = "[neutral] I see. [thinking] Tell me more."
                    expressions = ["neutral", "thinking"]
                else:
                    import random
                    responses = [
                        "[neutral] I understand. [thinking] That's interesting.",
                        "[happy] Got it! [neutral] What else would you like to talk about?",
                        "[thinking] Hmm, I see. [neutral] Please continue.",
                        "[happy] Interesting point! [neutral] Go on."
                    ]
                    response = random.choice(responses)
                    expressions = re.findall(r'\[(.*?)\]', response)
        
        # Add response to history
        self.conversation_history.append(f"Assistant: {response}")
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        return {
            "text": response,
            "expressions": expressions,
            "skip": False
        }

# ============================================================================
# Streaming STT Processor
# ============================================================================

class StreamingSTT:
    """STT processor that ONLY triggers LLM on final transcriptions"""
    
    def __init__(self, model_size: str = "base"):
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        self.SAMPLE_RATE = 16000
        self.CHUNK_SECONDS = 1.0
        self.HOP_SECONDS = 0.25
        self.CHUNK_FRAMES = int(self.CHUNK_SECONDS * self.SAMPLE_RATE)
        self.HOP_FRAMES = int(self.HOP_SECONDS * self.SAMPLE_RATE)
        
        # VAD parameters
        self.VAD_RMS_THRESHOLD = 0.008
        self.SILENCE_SECONDS = 1.5
        self.ACTIVATE_HOPS = 2
        self.SILENCE_HOPS = max(1, int(self.SILENCE_SECONDS / self.HOP_SECONDS))
        
        # State
        self.buffer = deque(maxlen=self.CHUNK_FRAMES)
        self.utterance = []
        self.currently_active = False
        self.active_hop_count = 0
        self.silent_hop_count = 0
        self.last_interim = ""
        self.last_final = ""
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def reset_state(self):
        """Reset all state variables"""
        self.buffer.clear()
        self.utterance = []
        self.currently_active = False
        self.active_hop_count = 0
        self.silent_hop_count = 0
        self.last_interim = ""
        
    def process_audio_chunk(self, audio_bytes: bytes) -> Tuple[str, str, bool]:
        """
        Process an audio chunk and return (interim_text, final_text, is_final)
        """
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            if len(audio_np) < self.HOP_FRAMES:
                # Too short, just add to buffer
                self.buffer.extend(audio_np.tolist())
                return "", "", False
            
            # Add to rolling buffer
            self.buffer.extend(audio_np.tolist())
            
            # Always add to utterance accumulator
            self.utterance.extend(audio_np.tolist())
            
            if len(self.buffer) < self.CHUNK_FRAMES:
                # Wait until buffer fills
                return "", "", False
            
            # Get current rolling chunk for interim transcript
            chunk = np.array(self.buffer, dtype=np.float32)
            
            # Compute RMS VAD on the current hop window
            hop_samples = chunk[-self.HOP_FRAMES:] if self.HOP_FRAMES <= len(chunk) else chunk
            rms = float(np.sqrt(np.mean(hop_samples.astype(np.float64) ** 2)))
            
            # Hop-counter VAD logic
            if rms > self.VAD_RMS_THRESHOLD:
                self.active_hop_count += 1
                self.silent_hop_count = 0
            else:
                self.silent_hop_count += 1
                self.active_hop_count = max(0, self.active_hop_count - 1)
            
            # Activate when we've seen enough consecutive active hops
            if not self.currently_active and self.active_hop_count >= self.ACTIVATE_HOPS:
                self.currently_active = True
                # Include recent buffer as start of utterance
                self.utterance = list(self.buffer)
                logger.info("🎤 Speech started")
            
            interim_text = ""
            final_text = ""
            is_final = False
            
            # Generate interim transcription if active (FOR DISPLAY ONLY)
            if self.currently_active:
                # Run interim transcription in background
                samples_np = chunk.copy()
                
                def transcribe_interim():
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp_path = tmp.name
                        
                        with wave.open(tmp_path, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(self.SAMPLE_RATE)
                            wf.writeframes((samples_np * 32767).astype(np.int16).tobytes())
                        
                        result = self.model.transcribe(tmp_path, language="en", fp16=False, temperature=0.0)
                        os.unlink(tmp_path)
                        return result.get("text", "").strip()
                    except Exception as e:
                        logger.error(f"Interim transcription error: {e}")
                        return ""
                
                # Submit to executor
                future = self.executor.submit(transcribe_interim)
                try:
                    interim_text = future.result(timeout=0.5)
                    if interim_text and interim_text != self.last_interim:
                        self.last_interim = interim_text
                except:
                    pass
            
            # Check for finalization (user stopped speaking)
            if self.currently_active and self.silent_hop_count >= self.SILENCE_HOPS:
                if len(self.utterance) > 0:
                    # Final transcription
                    samples = np.array(self.utterance, dtype=np.float32)
                    
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    with wave.open(tmp_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self.SAMPLE_RATE)
                        wf.writeframes((samples * 32767).astype(np.int16).tobytes())
                    
                    # Transcribe
                    result = self.model.transcribe(tmp_path, language="en", fp16=False, temperature=0.0)
                    os.unlink(tmp_path)
                    
                    final_text = result.get("text", "").strip()
                    is_final = True
                    self.last_final = final_text
                    
                    logger.info(f"🎤 Final transcription: '{final_text}'")
                    
                    # Reset state
                    self.currently_active = False
                    self.utterance = []
                    self.active_hop_count = 0
                    self.silent_hop_count = 0
                    self.last_interim = ""
            
            return interim_text, final_text, is_final
            
        except Exception as e:
            logger.error(f"STT processing error: {e}")
            return "", "", False

# ============================================================================
# Core Backend with Frontend Audio Playback
# ============================================================================

class ConnectionState:
    """Per-connection state management"""
    def __init__(self):
        self.cancel_event = asyncio.Event()
        self.is_processing = False
        self.is_speaking = False
        self.active_task = None
        self.stt_processor = StreamingSTT("base")
        self.last_interim = ""
        self.last_final_time = 0  # Will be set when first transcription happens
        self.min_final_interval = 1.0  # Reduced to 1 second
        self.last_robot_response = ""
        self.connection_time = time.time()
        
    def reset(self):
        """Reset state for new input"""
        self.cancel_event.set()
        self.cancel_event = asyncio.Event()
        self.is_speaking = False
        self.is_processing = False

class RoboticHeadBackend:
    """Holistic robotic head backend"""
    
    def __init__(self):
        self.llm = SmartLLM()
        self.tts = TTSGenerator()
        self.active_connections = {}  # websocket -> ConnectionState
    
    def _parse_sentences_with_expressions(self, text: str) -> List[Tuple[str, str]]:
        """Parse text into (expression, sentence) tuples"""
        parts = re.split(r'(\[.*?\])', text)
        current_expr = "neutral"
        sentences = []
        buffer = ""
        
        for part in parts:
            if not part:
                continue
            if part.startswith("[") and part.endswith("]"):
                # Flush buffer
                if buffer.strip():
                    for s in re.split(r'(?<=[.!?])\s+', buffer.strip()):
                        if s:
                            sentences.append((current_expr, s))
                    buffer = ""
                current_expr = part.strip("[]")
            else:
                buffer += (" " + part) if buffer else part
        
        if buffer.strip():
            for s in re.split(r'(?<=[.!?])\s+', buffer.strip()):
                if s:
                    sentences.append((current_expr, s))
        
        return sentences
    
    async def process_audio_chunk(self, websocket: WebSocket, audio_bytes: bytes, state: ConnectionState):
        """Process incoming audio chunk"""
        try:
            # Process with STT
            interim, final, is_final = state.stt_processor.process_audio_chunk(audio_bytes)
            
            # Send interim results (for display only)
            if interim and interim != state.last_interim:
                state.last_interim = interim
                await websocket.send_json({
                    "type": "interim_transcription",
                    "text": interim
                })
            
            # Handle final transcription (triggers LLM)
            if is_final and final:
                current_time = time.time()
                
                # For first transcription after connection, allow it immediately
                if state.last_final_time == 0:
                    # First transcription, allow it
                    state.last_final_time = current_time
                else:
                    # Rate limiting for subsequent transcriptions
                    time_since_last = current_time - state.last_final_time
                    if time_since_last < state.min_final_interval:
                        logger.info(f"⏱️ Rate limiting ({time_since_last:.1f}s < {state.min_final_interval}s): skipping '{final[:30]}...'")
                        return
                    
                    state.last_final_time = current_time
                
                logger.info(f"🎤 Final speech detected: '{final}'")
                
                # Cancel ongoing processing if any
                if state.is_processing or state.is_speaking:
                    logger.info("⚡ Interrupting ongoing response for new speech")
                    self.cancel_for_connection(websocket)
                    await asyncio.sleep(0.1)
                
                # Process the final transcription
                state.active_task = asyncio.create_task(
                    self._process_transcription(websocket, final, state)
                )
                
        except Exception as e:
            logger.error(f"Audio chunk processing error: {e}")
    
    async def _process_transcription(self, websocket: WebSocket, transcription: str, state: ConnectionState):
        """Process final transcription and send audio to frontend"""
        try:
            # Skip if too short
            if len(transcription.strip()) < 2:
                logger.info(f"⚠️ Skipping short transcription: '{transcription}'")
                return
                
            state.is_processing = True
            
            # Check for echo - if user is repeating robot's last response
            if state.last_robot_response:
                robot_clean = re.sub(r'[^\w\s]', '', state.last_robot_response.lower()).split()
                user_clean = re.sub(r'[^\w\s]', '', transcription.lower()).split()
                
                # Check for significant overlap
                # if user_clean and robot_clean:
                #     common_words = set(robot_clean) & set(user_clean)
                #     similarity = len(common_words) / max(len(user_clean), 1)
                #     if similarity > 0.6:  # If >60% similar
                #         logger.info(f"🔄 Ignoring possible echo (similarity {similarity:.2f}): '{transcription[:30]}...'")
                #         state.is_processing = False
                #         return
            
            # Send transcription to frontend
            await websocket.send_json({
                "type": "transcription",
                "text": transcription
            })
            
            # Generate LLM response
            llm_response = self.llm.generate_response(transcription)
            
            if llm_response.get("skip", False):
                state.is_processing = False
                return
            
            # Send LLM response text
            await websocket.send_json({
                "type": "llm_response",
                "text": llm_response["text"],
                "expressions": llm_response["expressions"]
            })
            
            # Parse sentences
            sentences = self._parse_sentences_with_expressions(llm_response["text"])
            
            if not sentences:
                sentences = [("neutral", llm_response["text"])]
            
            logger.info(f"📝 Processing {len(sentences)} sentences")
            
            # Store robot's response for echo detection
            clean_response = ' '.join([s for _, s in sentences])
            state.last_robot_response = clean_response
            
            # Process each sentence
            for idx, (expr, sentence) in enumerate(sentences):
                if state.cancel_event.is_set():
                    break
                
                logger.info(f"🔊 Generating audio for sentence {idx+1}/{len(sentences)}: '{sentence[:50]}...'")
                
                # Generate audio
                audio_data = self.tts.generate_audio(sentence, expr)
                
                if not audio_data:
                    logger.error(f"Failed to generate audio for: '{sentence[:30]}...'")
                    continue
                
                # Send audio to frontend
                await websocket.send_json({
                    "type": "audio_talking",
                    "expression": expr,
                    "text": sentence,
                    "audio_data": audio_data,
                    "sentence_index": idx,
                    "total_sentences": len(sentences)
                })
                
                # Estimate duration and wait (frontend will play the audio)
                # Estimate: 0.1 seconds per 10 characters, minimum 0.8 seconds
                estimated_duration = max(0.8, len(sentence) / 100)
                logger.info(f"⏱️ Estimated audio duration: {estimated_duration:.1f}s")
                
                # Wait for the audio to play
                await asyncio.sleep(estimated_duration)
                
                if state.cancel_event.is_set():
                    break
            
            # Completion
            state.is_processing = False
            state.is_speaking = False
            
            await websocket.send_json({
                "type": "pipeline_complete",
                "message": "Response completed"
            })
            
            await websocket.send_json({
                "type": "listening",
                "message": "ready"
            })
            
            logger.info("✅ Processing complete")
            
        except asyncio.CancelledError:
            logger.info("Processing cancelled")
            state.is_processing = False
            state.is_speaking = False
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                })
                await websocket.send_json({
                    "type": "listening",
                    "message": "ready"
                })
            except:
                pass
            state.is_processing = False
            state.is_speaking = False
    
    def cancel_for_connection(self, websocket: WebSocket):
        """Cancel ongoing processing for a connection"""
        state = self.active_connections.get(websocket)
        if not state:
            return
        
        logger.info("Cancelling ongoing tasks")
        state.reset()
        
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
            logger.info("Active task cancelled")

# ============================================================================
# FastAPI Application
# ============================================================================

backend = RoboticHeadBackend()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("=" * 60)
    logger.info("🤖 ROBOTIC HEAD BACKEND - FRONTEND AUDIO v4.0")
    logger.info("=" * 60)
    logger.info("Features:")
    logger.info("  ✅ Streaming STT with Whisper")
    logger.info("  ✅ Smart LLM responses with memory")
    logger.info("  ✅ Frontend audio playback (NO ECHO!)")
    logger.info("  ✅ Audio caching for performance")
    logger.info("  ✅ Expression-based TTS generation")
    logger.info("=" * 60)
    yield
    logger.info("🛑 Shutting down")

app = FastAPI(
    title="Robotic Head Backend",
    description="Holistic virtual robotic head with streaming STT, LLM, and frontend TTS",
    version="4.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML"""
    try:
        with open("templates/indexSimplified.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Robotic Head Backend</title>
        </head>
        <body>
            <h1>🤖 Robotic Head Backend v4.0</h1>
            <p>Frontend should be at <a href="/">http://localhost:8000/</a></p>
            <p>Make sure indexSimplified.html is in the same directory as this Python file.</p>
        </body>
        </html>
        """)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "4.0",
        "active_connections": len(backend.active_connections),
        "tts_cache_size": len(backend.tts.audio_cache),
        "timestamp": time.time()
    }

@app.post("/api/speak")
async def speak_direct(data: dict):
    """Direct TTS generation endpoint"""
    try:
        text = data.get("text", "Hello")
        expression = data.get("expression", "neutral")
        
        audio_data = backend.tts.generate_audio(text, expression)
        
        if not audio_data:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        return {
            "status": "generated",
            "text": text,
            "expression": expression,
            "audio_size": len(audio_data),
            "has_audio": bool(audio_data)
        }
        
    except Exception as e:
        logger.error(f"Speak error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop")
async def stop_tts():
    """Stop any ongoing processing"""
    for state in backend.active_connections.values():
        state.reset()
    return {"status": "stopped"}

@app.post("/api/clear_cache")
async def clear_cache():
    """Clear TTS cache"""
    backend.tts.clear_cache()
    return {"status": "cache_cleared"}

# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bidirectional communication.
    """
    await websocket.accept()
    
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(f"🔌 New WebSocket connection from {client_host}")
    
    # Initialize per-connection state
    state = ConnectionState()
    backend.active_connections[websocket] = state
    
    try:
        # Send initialization data
        await websocket.send_json({
            "type": "system",
            "message": "Backend ready with frontend audio playback. No echo!",
            "version": "4.0",
            "sample_rate": 16000,
            "rate_limit": state.min_final_interval
        })
        logger.info("✅ Ready signal sent to client")
        
        # Main message loop
        while True:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(websocket.receive_json(), timeout=300.0)
                
                message_type = data.get("type", "")
                
                if message_type == "audio":
                    # Handle audio chunk
                    audio_data_b64 = data.get("data", "")
                    
                    if not audio_data_b64:
                        continue
                    
                    # Decode audio
                    audio_data = base64.b64decode(audio_data_b64)
                    
                    # Process audio chunk in background
                    asyncio.create_task(
                        backend.process_audio_chunk(websocket, audio_data, state)
                    )
                    
                elif message_type == "expression":
                    # Handle expression change
                    expression = data.get("expression", "neutral")
                    await websocket.send_json({
                        "type": "expression_ack",
                        "expression": expression
                    })
                    
                elif message_type == "stop_speaking":
                    # Stop current speech
                    logger.info("⏹️ Stop speaking requested")
                    backend.cancel_for_connection(websocket)
                    await websocket.send_json({
                        "type": "stopped",
                        "message": "Speech stopped by user"
                    })
                    
                elif message_type == "clear_history":
                    # Clear conversation history
                    logger.info("🧹 Clearing conversation history")
                    backend.llm.conversation_history.clear()
                    await websocket.send_json({
                        "type": "history_cleared",
                        "message": "Conversation history cleared"
                    })
                    
                elif message_type == "audio_done":
                    # Frontend finished playing audio
                    logger.debug(f"🎵 Audio playback complete")
                    
                elif message_type == "ping":
                    # Keepalive
                    await websocket.send_json({"type": "pong"})
                    
                elif message_type == "test":
                    # Test message
                    test_msg = data.get("message", "test")
                    logger.info(f"🧪 Test message: {test_msg}")
                    await websocket.send_json({
                        "type": "test_response",
                        "message": f"Received: {test_msg}"
                    })
                    
                else:
                    logger.warning(f"Unknown message type: {message_type}")
                    
            except asyncio.TimeoutError:
                # Send keepalive
                try:
                    await websocket.send_json({"type": "keepalive", "timestamp": time.time()})
                except:
                    pass
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"🚨 WebSocket error: {e}")
    finally:
        # Clean up
        backend.active_connections.pop(websocket, None)
        logger.info(f"🔌 Connection from {client_host} cleaned up")

# ============================================================================
# Startup
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Check for required external commands
    logger.info("Checking system requirements...")
    
    # Check if 'say' command is available (macOS)
    try:
        result = subprocess.run(["which", "say"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ 'say' command found at: {result.stdout.strip()}")
        else:
            logger.warning("⚠️ 'say' command not found. TTS will not work on macOS.")
    except Exception as e:
        logger.error(f"Error checking for 'say' command: {e}")
    
    # Check for audio conversion tools
    tools = ["ffmpeg", "sox"]
    for tool in tools:
        try:
            result = subprocess.run(["which", tool], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ {tool} found at: {result.stdout.strip()}")
            else:
                logger.warning(f"⚠️ {tool} not found. Audio conversion may use fallback methods.")
        except:
            pass
    
    logger.info("🚀 Starting server on http://0.0.0.0:8000")
    logger.info("💡 Access the interface at http://localhost:8000")
    logger.info("📡 WebSocket endpoint: ws://localhost:8000/ws")
    logger.info("")
    logger.info("📝 First-time setup notes:")
    logger.info("   1. Whisper model will download on first use (may take a minute)")
    logger.info("   2. First audio generation may be slow as cache warms up")
    logger.info("   3. Ensure microphone permissions are granted in browser")
    
    uvicorn.run(
        "vrs:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )