import argparse
import os
import time
import json
import random
import gradio as gr
import traceback
import threading
import queue
import gc
import re
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto
from acestep.pipeline_ace_step import ACEStepPipeline
import librosa


# Constants and Configuration
SUPPORTED_LANGUAGES = {
    "English": "English",
    "Spanish": "Spanish", 
    "French": "French",
    "German": "German",
    "Italian": "Italian",
    "Portuguese": "Portuguese",
    "Japanese": "Japanese",
    "Chinese": "Chinese",
    "Russian": "Russian",
    "Arabic": "Arabic",
    "Hindi": "Hindi",
    "Korean": "Korean",
    "Finnish": "Finnish"
}

# Genre-specific configurations
GENRE_DURATIONS = {
    # Existing genres
    "pop": 180,
    "rock": 210,
    "hip hop": 210,
    "electronic": 300,
    "lofi": 120,
    "jazz": 240,
    "classical": 360,
    "ambient": 420,
    "country": 210,
    
    # New genres
    "metal": 240,        # Longer for solos
    "death metal": 210,  # More concise brutality
    "doom metal": 420,   # Extended heavy passages
    "reggae": 240,       # Extended dub sections
    "dub": 300,          # Even longer instrumental
    "blues": 240,        # 12-bar repetitions
    "delta blues": 300,  # Storytelling format
    "funk": 210,         # Tight grooves
    "disco": 270,        # Extended dance mixes
    "punk": 150,         # Short and fast
    
    "default": 180
}

GENRE_TEMPOS = {
    # Existing genres
    "pop": 120,
    "rock": 140,
    "hip hop": 95,
    "electronic": 128,
    "lofi": 85,
    "jazz": 110,
    "classical": 80,
    "ambient": 70,
    "country": 100,
    
    # New genres
    "metal": 160,       # Faster for thrash/speed metal
    "death metal": 180, # Extreme metal subgenre
    "doom metal": 60,   # Slower, heavier variant
    "reggae": 90,       # Laid-back offbeat rhythm
    "dub": 85,          # Even slower reggae variant
    "blues": 100,       # Classic 12-bar shuffle
    "delta blues": 80,  # Slower acoustic blues
    "funk": 115,        # Groove-oriented
    "disco": 120,       # Danceable four-on-the-floor
    "punk": 180,        # Fast and aggressive
    
    "default": 120
}

THEME_SUGGESTIONS = {
    "pop": [
        "summer love", "heartbreak", "dancing", "celebrity crush", "night out",
        "young and free", "last goodbye", "party all night", "secret admirer",
        "teenage dreams", "radio hit", "glitter and gold", "midnight kiss",
        "fame and fortune", "broken promises"
    ],
    "rock": [
        "rebellion", "road trip", "broken dreams", "rock and roll lifestyle",
        "guitar hero", "against the machine", "burning bridges", "small town blues",
        "arena anthem", "leather and spikes", "last train home", "bar fight",
        "whiskey rebellion", "soundcheck at midnight"
    ],
    "hip hop": [
        "street life", "hustle", "success story", "city nights",
        "hood chronicles", "money and power", "block party", "trap dreams",
        "lyrical warfare", "24/7 grind", "bounce to this", "gold chains",
        "concrete jungle", "from nothing to something"
    ],
    "electronic": [
        "neon lights", "cosmic journey", "digital dreams", "midnight drive",
        "rave till dawn", "synthetic love", "cyberpunk city", "bass drop",
        "laser fantasy", "underground warehouse", "alien transmission",
        "metropolis pulse", "glitch in the matrix", "afterhours euphoria"
    ],
    "lofi": [
        "rainy day", "coffee shop", "study session", "chill vibes",
        "late night thoughts", "window seat", "old cassette tapes",
        "sunday morning", "vinyl crackle", "city skyline at dusk",
        "nostalgic memories", "quiet bookstore", "snowfall at midnight"
    ],
    "jazz": [
        "smooth nights", "blue mood", "speakeasy", "saxophone dreams",
        "moonlight serenade", "smoky bar", "improvisation", "uptown swing",
        "jazz age", "walking bassline", "after hours jam", "velvet voice",
        "trumpet solo", "harlem renaissance"
    ],
    "classical": [
        "moonlight", "spring morning", "winter tale", "royal ball",
        "symphony no. 9", "opera drama", "baroque garden", "piano concerto",
        "sunrise sonata", "gothic cathedral", "waltz of the flowers",
        "requiem mass", "orchestral storm", "string quartet"
    ],
    "ambient": [
        "ocean waves", "forest walk", "mountain sunrise", "deep space",
        "arctic winds", "desert mirage", "underwater caves", "celestial bodies",
        "morning mist", "ancient ruins", "silent meditation", "bioluminescent bay",
        "northern lights", "empty train station at night"
    ],
    "country": [
        "small town", "truck driving", "lost love", "whiskey nights", "backroads",
        "dusty boots", "honky tonk angel", "rodeo clown", "front porch swing",
        "blue jeans", "pickup truck", "county fair", "dirt road diary",
        "grandpa's guitar", "southern comfort"
    ],
    "metal": [
        "battle cries", "apocalypse now", "dragon slayer", "demonic possession",
        "forge of souls", "viking conquest", "doomsday prophecy", "blackened sky",
        "mosh pit madness", "necrotic ritual", "shred till death",
        "iron maiden voyage", "blood moon rising"
    ],
    "reggae": [
        "island breeze", "one love", "ganja farmer", "sunshine state of mind",
        "beach bonfire", "rasta revolution", "dub plate special",
        "tropical storm", "bob marley tribute", "steel drum sunset",
        "kingston nights", "irie vibrations"
    ],
    "blues": [
        "crossroads deal", "mississippi delta", "empty whiskey bottle",
        "train whistle blues", "juke joint", "broken guitar string",
        "stormy monday", "dust my broom", "hard luck woman",
        "12-bar confession", "slide guitar tears"
    ],
    "default": [
        "love", "dreams", "adventure", "nostalgia",
        "second chances", "hidden truth", "forbidden fruit",
        "eternal youth", "parallel universe", "fading memories"
    ]
}

# Enums and Data Classes
class RadioState(Enum):
    STOPPED = auto()
    BUFFERING = auto()
    PLAYING = auto()
    PAUSED = auto()

@dataclass
class Song:
    title: str
    artist: str
    genre: str
    theme: str
    duration: float
    lyrics: str
    language: str 
    prompt: str
    audio_path: str
    generation_time: float
    timestamp: float
    metadata: dict

@dataclass
class StationIdentity:
    name: str
    slogan: str
    
    @classmethod
    def generate_identity(cls, genre: str, theme: str):
        """Generate cohesive station branding based on genre and theme"""
        # Format the station name
        theme = theme.replace(" radio", "").replace("Radio", "").title()
        suffixes = {
            "hip hop": "FM",
            "electronic": "Radio",
            "classical": "Classical",
            "jazz": "Jazz",
            "lofi": "Lo-Fi",
            "country": "Country",
            "metal": "Metal",
            "reggae": "Reggae",
            "blues": "Blues",
            "default": "Radio"
        }
        suffix = suffixes.get(genre.lower(), suffixes["default"])
        name = f"{genre.title()} {theme} {suffix}"
        
        # Generate slogan
        slogans = {
            "pop": f"The hottest {theme} hits!",
            "rock": f"Loud {theme} anthems 24/7",
            "hip hop": f"Real {theme} sounds",
            "electronic": f"{theme} frequencies",
            "lofi": f"{theme} beats to relax to",
            "jazz": f"Smoothest {theme} in town",
            "classical": f"{theme} masterpieces",
            "ambient": f"{theme} soundscapes",
            "country": f"Real {theme} country music",
            "metal": f"Brutal {theme} sounds",
            "reggae": f"Irie {theme} vibrations",
            "blues": f"Authentic {theme} blues",
            "default": f"Your {theme} soundtrack"
        }
        slogan = slogans.get(genre.lower(), slogans["default"])
        
        
        return cls(name, slogan)

class AIRadioStation:
    def __init__(self, ace_step_pipeline: ACEStepPipeline, model_path: str = "gemma-3-12b-it-abliterated.q4_k_m", ollama: bool = False):
        """
        Initialize the AI Radio Station with continuous generation.
        
        Args:
            ace_step_pipeline: Initialized ACEStepPipeline for music generation
            model_path: Path to LLM model for lyric generation
            ollama: Whether to use Ollama for lyric generation (default: False)
        """
        self._pipeline = ace_step_pipeline  # Store the original pipeline reference
        self.random_mode = False 
        self.llm_model_path = model_path
        self.ollama = ollama
        self.llm = None
        self._first_play = True  
        self.pipeline_args = {
            'checkpoint_dir': ace_step_pipeline.checkpoint_dir,
            'dtype': "bfloat16",
            'torch_compile': ace_step_pipeline.torch_compile
        }
        self.current_pipeline = None  # Will be initialized when needed
        self.identity: Optional[StationIdentity] = None
        self.current_song: Optional[Song] = None
        self.playlist: List[Song] = []
        self.history: List[Song] = []
        self.song_queue = queue.Queue()
        self.state = RadioState.STOPPED
        self.stop_event = threading.Event()
        self.min_buffer_size = 1
        self.generation_thread = None
        self.playback_thread = None
        self.playback_paused = threading.Event()
        self.max_history_size = 50
        self.cache_cleanup_interval = 60
        self.last_cache_cleanup = time.time()
        self.language = "English"
        self.tempo = 120
        self.intensity = "medium"
        self.mood = "upbeat"
        self.generation_progress = 0.0
        self.current_song_elapsed = 0.0
        


    def load_llm(self):
        """Load the LLM model into memory"""
        self.release_pipeline()
        self.unload_llm()
        gc.collect()
        if self.llm is None:
            if self.ollama:
                self.llm = True
            else:
                print("Loading LLM model...")
                try:
                    from llama_cpp import Llama
                    self.llm = Llama(
                        model_path=self.llm_model_path,
                        n_ctx=2048,
                        n_threads=4,
                        n_gpu_layers=-1,
                        seed = -1 # random seed for random lyrics

                    )
                except ImportError:
                    print("Warning: llama-cpp-python not installed, using simple lyric generation")
                    self.llm = None

    def unload_llm(self):
        """Unload the LLM model from memory"""
        if self.llm is not None:
            if not self.ollama:
                print("Unloading LLM model...")
                del self.llm
            self.llm = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def get_pipeline(self):
        """Always create a fresh pipeline with proper cleanup"""
        if self.current_pipeline is not None:
            self.release_pipeline()  # Ensure previous pipeline is cleaned up
        
        # Force cleanup before creating new pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Initializing fresh pipeline...")
        self.current_pipeline = ACEStepPipeline(**self.pipeline_args)
        return self.current_pipeline

    def release_pipeline(self):
        """Clean up pipeline resources thoroughly"""
        if self.current_pipeline is not None:
            print("Releasing pipeline resources...")
            
            # First delete ACE-specific resources
            if hasattr(self.current_pipeline, 'ace_model'):
                if hasattr(self.current_pipeline.ace_model, 'cpu'):
                    try:
                        self.current_pipeline.ace_model.cpu()
                    except:
                        pass
                del self.current_pipeline.ace_model
                
            # Now delete the full pipeline
            del self.current_pipeline
            self.current_pipeline = None
            
            # Force cleanup
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

    def clean_all_memory(self):
        """More aggressive memory cleanup"""
        # First unload LLM if loaded
        self.unload_llm()
        
        # Then release pipeline resources
        self.release_pipeline()
        
        # Force CUDA cleanup with synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all kernels to finish
            torch.cuda.empty_cache()  # Clear cache
            torch.cuda.reset_peak_memory_stats()  # Reset tracking
        
        # Multiple GC passes
        for _ in range(3):
            gc.collect()



    def _playback_worker(self):
        """Simplified playback worker that switches songs when they end"""
        self.state = RadioState.BUFFERING
        print("Playback worker started - buffering songs...")
        
        last_cache_clean = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Cache cleanup (non-blocking check)
                now = time.time()
                if now - last_cache_clean > self.cache_cleanup_interval:
                    self._cleanup_cache()
                    last_cache_clean = now
                
                # Handle paused state
                if self.playback_paused.is_set():
                    self.state = RadioState.PAUSED
                    time.sleep(0.1)  # Shorter sleep
                    continue
                
                # First play buffer check
                if self._first_play:
                    if self.song_queue.qsize() >= self.min_buffer_size:
                        print(f"Initial buffer filled with {self.song_queue.qsize()} songs - starting playback!")
                        self.state = RadioState.PLAYING
                        self._first_play = False
                    else:
                        # Wait for buffer to fill
                        time.sleep(0.5)
                        continue
                
                # Main playback logic
                try:
                    song = self.song_queue.get_nowait()  # Non-blocking get
                    self._play_song(song)
                    # No need to track elapsed time - just play each song and move on
                    
                except queue.Empty:
                    # Tiny sleep when empty to prevent CPU spin
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Playback error: {e}")
                traceback.print_exc()
                time.sleep(1)  # Prevent crash loops

    def _play_song(self, song):
        """Simplified song playback - just play the full song without tracking time"""
        self.current_song = song
        self.history.append(song)
        # Enforce max history size
        # if len(self.history) > self.max_history_size:
            # Remove oldest song(s) - could do more than one if needed
            # self.history.pop(0)
        
        print(f"\n=== Now Playing ===\n"
            f"Title: {song.title}\n"
            f"Duration: {song.duration:.1f}s\n")
        
        # Simply sleep for the song duration
        # Use the requested duration from the song object
        target_duration = float(song.duration)
        
        print(f"Playing song for full duration: {target_duration:.1f}s")
        
        # Single sleep for the full song duration
        start_time = time.time()
        
        # Wait until song finishes or playback is interrupted
        while (time.time() - start_time) < target_duration:
            if self.stop_event.is_set() or self.playback_paused.is_set():
                break
            time.sleep(0.1)  # Short sleep to be responsive to interruptions
        
        # Song completed
        if not (self.stop_event.is_set() or self.playback_paused.is_set()):
            print(f"Song complete - played for full duration: {target_duration:.1f}s")

    def _cleanup_cache(self):
        """More aggressive VRAM cleanup"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Reset the pipeline completely
        if self.current_pipeline is not None:
            del self.current_pipeline
            self.current_pipeline = None
        
        # Force Python to release memory
        for _ in range(3):
            gc.collect()
        
        # Clean up temporary files
        try:
            temp_dir = os.path.join(os.getcwd(), "temp")
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    file_age = time.time() - os.path.getmtime(file_path)
                    if file_age > 3600 and os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
        except Exception:
            pass
        
        self.last_cache_cleanup = time.time()

    def start_radio(self, genre: str, theme: str, duration: Optional[float] = None, 
                buffer_size: int = 1, language: str = "English", max_history: int = 50, 
                tempo: Optional[int] = None, intensity: Optional[str] = None, 
                mood: Optional[str] = None, random_mode: bool = False, random_languages: bool = False):
        """Start the radio station with smart defaults"""
        self.stop_radio()
        self.stop_event.clear()
        self.playback_paused.clear()
        
        # Set smart defaults if parameters not provided
        duration = duration or GENRE_DURATIONS.get(genre.lower(), GENRE_DURATIONS["default"])
        tempo = tempo or GENRE_TEMPOS.get(genre.lower(), GENRE_TEMPOS["default"])
        intensity = intensity or "medium"
        mood = mood or self._detect_mood(theme)
        
        print(f"\n=== Starting song generation ===")
        print(f"Genre: {genre}, Theme: {theme}, Duration: {duration}s")
        if tempo is not None:
            print(f"Tempo: {tempo} BPM")
        if intensity is not None:
            print(f"Intensity: {intensity}")
        if mood is not None:
            print(f"Mood: {mood}")
        
        # Store parameters for lyric generation
        self.tempo = tempo or self.tempo
        self.intensity = intensity or self.intensity
        self.mood = mood or self.mood
        # Generate station identity (use original genre/theme even in random mode for branding)
        self.identity = StationIdentity.generate_identity(genre, theme)
        self.min_buffer_size = buffer_size
        self.max_history_size = max_history
        self.language = language
        self.random_mode = random_mode
        self.random_languages=random_languages 
        self.playlist = []
        self.history = []
        self.state = RadioState.BUFFERING
        self.generation_progress = 0.0
        
        print(f"\n=== Starting {self.identity.name} ===")
        print(f"Slogan: {self.identity.slogan}")
        print(f"Random Mode: {'ON' if random_mode else 'OFF'}")
        print(f"Buffering {buffer_size} songs before playback...")
        print(f"Language: {SUPPORTED_LANGUAGES.get(language, language)}")
        print(f"Tempo: {tempo} BPM, Intensity: {intensity}, Mood: {mood}")
        
        # Start background workers
        self.generation_thread = threading.Thread(
            target=self._generate_song_worker,
            args=(genre, theme, duration),
            daemon=True
        )
        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True
        )
        
        self.generation_thread.start()
        self.playback_thread.start()

    def _detect_mood(self, theme: str) -> str:
        """Detect mood from theme keywords"""
        theme_lower = theme.lower()
        if any(word in theme_lower for word in ["love", "happy", "celebrate", "party"]):
            return "happy"
        elif any(word in theme_lower for word in ["sad", "heartbreak", "lonely", "tears"]):
            return "sad"
        elif any(word in theme_lower for word in ["angry", "rage", "fight", "protest"]):
            return "angry"
        elif any(word in theme_lower for word in ["chill", "relax", "calm", "peace"]):
            return "chill"
        elif any(word in theme_lower for word in ["dream", "thought", "memory", "reflect"]):
            return "reflective"
        return "upbeat"

    def generate_lyrics_and_prompt(self, genre: str, theme: str, language: str = "English") -> Tuple[str, str]:
        """Generate song lyrics with genre-specific structures and retry logic"""
        structures = {
            "country": (
                "[Steel Guitar Intro]\n\n"
                "[Verse 1] (storytelling)\n{lyrics}\n\n"
                "[Chorus] (big melody)\n{lyrics}\n\n"
                "[Verse 2] (develop story)\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Fiddle Solo] (8 bars)\n\n"
                "[Bridge] (emotional peak)\n{lyrics}\n\n"
                "[Double Chorus] (with harmonies)"
            ),
            "pop": (
                "[Verse 1]\n{lyrics}\n\n"
                "[Pre-Chorus]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Pre-Chorus]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge]\n{lyrics}\n\n"
                "[Final Chorus] (with ad-libs)"
            ),
            "rock": (
                "[Guitar Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Guitar Solo] (8-16 bars)\n\n"
                "[Bridge]\n{lyrics}\n\n"
                "[Double Chorus] (big finish)"
            ),
            "hip hop": (
                "[Intro Hook]\n{lyrics}\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge] (optional rap breakdown)\n\n"
                "[Outro] (fade with ad-libs)"
            ),
            "electronic": (
                "[Atmospheric Intro] (16 bars)\n\n"
                "[Build-Up]\n{lyrics}\n\n"
                "[Drop]\n{lyrics}\n\n"
                "[Breakdown Verse]\n{lyrics}\n\n"
                "[Build-Up]\n{lyrics}\n\n"
                "[Drop]\n{lyrics}\n\n"
                "[Outro] (beat fade)"
            ),
            "lofi": (
                "[Ambient Intro] (with vinyl noise)\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chill Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chill Chorus]\n{lyrics}\n\n"
                "[Instrumental Break] (8 bars)\n\n"
                "[Outro] (fade with rain sounds)"
            ),
            "jazz": (
                "[Piano Intro] (improvised)\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Swing Chorus]\n{lyrics}\n\n"
                "[Instrumental Break] (sax solo)\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Swing Chorus]\n{lyrics}\n\n"
                "[Outro] (group improv)"
            ),
            "classical": (
                "[Orchestral Introduction]\n\n"
                "[Theme A]\n{lyrics}\n\n"
                "[Theme B] (variation)\n\n"
                "[Development Section]\n\n"
                "[Recapitulation]\n{lyrics}\n\n"
                "[Coda] (grand finale)"
            ),
            "ambient": (
                "[Textural Intro] (2-4 minutes)\n\n"
                "[Drone Section]\n{lyrics}\n\n"
                "[Modulation]\n\n"
                "[Resolution Section]\n{lyrics}\n\n"
                "[Fade Out] (gradual)"
            ),
            "metal": (
                "[Shredding Intro] (fast picking)\n\n"
                "[Verse 1] (growled vocals)\n{lyrics}\n\n"
                "[Chorus] (clean vocals)\n{lyrics}\n\n"
                "[Guitar Solo] (tapping)\n\n"
                "[Breakdown] (chugging riffs)\n\n"
                "[Final Blast] (double bass)"
            ),
            "reggae": (
                "[Skank Guitar Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus] (call-and-response)\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Dub Section] (instrumental)\n\n"
                "[Final Chorus] (with harmonies)"
            ),
            "blues": (
                "[Guitar Lick Intro] (12-bar)\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Response] (guitar answers vocal)\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Harmonica Solo] (12-bar)\n\n"
                "[Outro] (repeat and fade)"
            ),
            "default": (
                "[Intro]\n{lyrics}\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge/Middle 8]\n{lyrics}\n\n"
                "[Chorus] (variation)\n\n"
                "[Outro] (optional vamp)"
            )
        }

        structure = structures.get(genre.lower(), structures["default"])
        
        prompt_addons = {
            "pop": "radio-ready, catchy hooks, polished production",
            "rock": "electric guitars, driving drums, raw energy",
            "hip hop": "punchy beats, rhythmic flow, urban vibe",
            "electronic": "synthesizers, pulsing bass, euphoric drops", 
            "lofi": "chill beats, vinyl crackle, relaxed vibe",
            "jazz": "smooth saxophone, walking bass, improvisational solos",
            "classical": "orchestral arrangements, dynamic phrasing, emotional depth",
            "ambient": "atmospheric pads, subtle textures, immersive soundscapes",
            "country": "steel guitar, fiddle, storytelling, twangy vocals",
            "metal": "distorted guitars, double bass drums, aggressive vocals",
            "reggae": "offbeat rhythms, organ skanks, dub effects",
            "blues": "bent notes, 12-bar structure, soulful vocals",
            "default": "melodic, emotionally expressive, professional mix"
        }
        
        intensity_modifiers = {
            "soft": "gentle, subtle, delicate, mellow, calm",
            "medium": "balanced, moderate, steady",
            "high": "energetic, powerful, intense, loud, aggressive"
        }
        
        mood_modifiers = {
            "happy": "uplifting, joyful, positive, cheerful",
            "sad": "melancholic, sorrowful, emotional, touching",
            "reflective": "thoughtful, introspective, contemplative",
            "angry": "passionate, intense, raw, powerful",
            "upbeat": "lively, optimistic, vibrant, spirited",
            "chill": "relaxed, laid-back, peaceful, smooth"
        }

        prompt = (
            f"Write a {genre} song in {language} about '{theme}' using this exact structure:\n"
            f"{structure}\n\n"
            "STRICT REQUIREMENTS:\n"
            "1. Write ONLY the song lyrics - no translations, no explanations, no notes\n"
            "2. Use ONLY the specified language: {language}\n"
            "3. Follow the structure EXACTLY as shown\n"
            "4. Format each section header exactly as shown (e.g. [Verse 1])\n"
            "5. Never include any text outside the lyrics structure\n\n"
            "STYLE GUIDELINES:\n"
            f"- {prompt_addons.get(genre.lower(), prompt_addons['default'])}\n"
            f"- {intensity_modifiers.get(self.intensity, intensity_modifiers['medium'])} feel\n"
            f"- {mood_modifiers.get(self.mood, mood_modifiers['upbeat'])} mood\n"
            "- Use vivid imagery and emotional resonance\n"
            "- Match the rhythm and phrasing to {genre} conventions\n"
            f"{'- Incorporate country idioms and themes' if genre.lower() == 'country' else ''}"
            f"{'- Use rap flow patterns and urban vocabulary' if genre.lower() == 'hip hop' else ''}\n\n"
        )
        
        print(f"\nLyric generation prompt:\n{prompt}")
        
        lyrics = ""
        max_retries = 2  # Number of retries before falling back
        retry_count = 0
        
        if self.llm_model_path:  # Only try if we have a model path
            while retry_count <= max_retries:
                try:
                    # Load model right before use
                    self.release_pipeline()
                    self.load_llm()
                    
                    if self.llm:  # Check if load was successful
                        print(f"Using LLM for lyric generation (attempt {retry_count + 1}/{max_retries + 1})...")
                        if self.ollama:
                            import ollama
                            output = ollama.chat(
                                model=self.llm_model_path,
                                messages=[
                                    {
                                        'role': 'user',
                                        'content': prompt,
                                    },
                                ]
                            )
                            lyrics = output.message.content.strip()
                        else:
                            output = self.llm(
                                prompt,
                                max_tokens=700,
                                temperature=0.7,
                                top_p=0.9,
                                repeat_penalty=1.1,
                                stop=["[End]", "\n\n\n"],
                                echo=False,
                                seed=-1
                            )
                            lyrics = output["choices"][0]["text"].strip()

                        print(f"Generated lyrics:\n{lyrics}")
                        
                        # Validate lyrics quality
                        if not lyrics or len(lyrics.splitlines()) < 6:
                            raise ValueError("Lyrics too short or empty")
                            
                        lyrics = lyrics.replace("[Inst]", "").strip()
                        break  # Success - exit retry loop
                        
                except Exception as e:
                    print(f"Lyric generation attempt {retry_count + 1} failed: {str(e)}")
                    retry_count += 1
                    if retry_count <= max_retries:
                        print("Retrying...")
                        time.sleep(1)  # Brief delay before retry
                    else:
                        print("Max retries reached, using fallback lyrics")
                        lyrics = self._fallback_lyrics(genre, theme)
                finally:
                    # Always unload after generation attempt
                    self.unload_llm()
        else:
            lyrics = self._fallback_lyrics(genre, theme)
        
        music_prompt = (
            f"{genre} song about {theme}, professional production, "
            f"high quality, {self.tempo} BPM, "
            f"{self.intensity} intensity, {self.mood} mood, "
            "clear vocals, catchy melody"
        )
        
        lyrics = self._clean_lyrics(lyrics)
        return lyrics, music_prompt

    def _clean_lyrics(self, lyrics_text: str) -> str:
        """Clean lyrics by removing instructional text and formatting"""
        cleaned = re.sub(r'\([^)]*\)', '', lyrics_text)  # Remove (instructions)
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Normalize line breaks
        return cleaned.strip()

    def _fallback_lyrics(self, genre: str, theme: str) -> str:
        """Simple fallback lyric generation if LLM fails"""
        return (
            f"[Verse 1]\nSinging about {theme} in {genre} style\n"
            "The AI is making music all the while\n\n"
            f"[Chorus]\nThis is {genre} radio\n"
            "Generated just for you\n"
            "Listen close and you might know\n"
            "What these lyrics try to do\n\n"
            f"[Verse 2]\n{theme} is what we're singing\n"
            "In this digital creation\n"
            "No human hands were wringing\n"
            "Just pure AI inspiration"
        )

    def _generate_song_worker(self, genre: str, theme: str, duration: float):
        """Background worker for continuous song generation"""
        first_song_in_random_mode = True  # Track if this is the first song in random mode
        
        while not self.stop_event.is_set():
            try:
                self.clean_all_memory()
                # Always generate if queue is below buffer size
                if self.song_queue.qsize() < self.min_buffer_size:
                    # Handle random language selection independently of random_mode
                    current_language = self.language
                    if self.random_languages and not first_song_in_random_mode:
                        current_language = random.choice(list(SUPPORTED_LANGUAGES.keys()))
                    
                    # If in random mode, generate random parameters for each song AFTER the first one
                    if self.random_mode and not first_song_in_random_mode:
                        genres = list(THEME_SUGGESTIONS.keys())
                        current_genre = random.choice(genres)
                        current_theme = random.choice(THEME_SUGGESTIONS.get(current_genre, THEME_SUGGESTIONS["default"]))
                        current_tempo = GENRE_TEMPOS.get(current_genre, GENRE_TEMPOS["default"])
                        current_intensity = random.choice(["soft", "medium", "high"])
                        current_mood = random.choice(["happy", "sad", "reflective", "upbeat", "chill"])
                        
                        song = self.generate_song(
                            genre=current_genre,
                            theme=current_theme,
                            tempo=current_tempo,
                            intensity=current_intensity,
                            mood=current_mood,
                            language=current_language  # Use the potentially randomized language
                        )
                    else:
                        # Use the selected parameters for the first song or when not in random mode
                        song = self.generate_song(
                            genre=genre,
                            theme=theme,
                            duration=duration,
                            language=current_language  # Use the potentially randomized language
                        )
                        first_song_in_random_mode = False  # After first song, we can randomize if enabled
                        
                    self.song_queue.put(song)
                    self.playlist.append(song)
                    
                    if len(self.playlist) % 5 == 0:
                        self._save_state()
                    
                    # Small delay between generations
                    time.sleep(1)
                else:
                    # If buffer is full, check more frequently
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error in generation worker: {e}")
                traceback.print_exc()
                # Ensure cleanup on error
                self.clean_all_memory()
                # Longer delay on error
                time.sleep(5)
            finally:
                # Always clean after generation attempt
                self.clean_all_memory()

    def _save_state(self):
        """Save current radio state to disk"""
        state = {
            "current_station": self.identity.name if self.identity else "",
            "current_song": self.current_song.title if self.current_song else None,
            "playlist": [song.title for song in self.playlist],
            "history": [song.title for song in self.history],
            "state": self.state.name,
            "timestamp": time.time()
        }
        
        with open("radio_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def stop_radio(self):
        """Stop all radio operations and clean up resources"""
        self.stop_event.set()
        self.playback_paused.clear()
        self.state = RadioState.STOPPED
        self._first_play = True  # Reset the flag
        
        # Clean up both pipeline and LLM
        self.clean_all_memory()
        
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=5)
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=5)
        
        self._save_state()


    def generate_song(self, genre: str, theme: str, duration: float = 120.0, 
                    tempo: Optional[int] = None, intensity: Optional[str] = None,
                    mood: Optional[str] = None, language: Optional[str] = None) -> Song:
        """
        Generate a complete song with lyrics and music.
        
        Args:
            genre: Music genre (pop, rock, etc.)
            theme: Song theme/topic
            duration: Song length in seconds
            tempo: Beats per minute (optional)
            intensity: Song intensity level (soft/medium/high)
            mood: Emotional mood of the song
            language: Language for lyrics
            
        Returns:
            Song: Generated song object
            
        Raises:
            Exception: If generation fails at any stage
        """
        language = language or self.language
        # Make sure we're using the requested duration
        duration = float(duration)
        
        print(f"\n=== Starting song generation ===")
        print(f"Genre: {genre}, Theme: {theme}, Language: {language}, Target Duration: {duration}s")        
        
        # Initialize progress tracking
        self.generation_progress = 0.0
        song = None
        
        try:
            # Stage 1: Generate lyrics
            self.clean_all_memory()
            print("\n[1/3] Generating lyrics...")
            self.generation_progress = 0.33
            lyrics, music_prompt = self.generate_lyrics_and_prompt(genre, theme, language)
            self.clean_all_memory()
            
            # Stage 2: Generate music
            print(f"\n[2/3] Generating music with ACEStepPipeline (target duration: {duration}s)...")
            self.generation_progress = 0.66
            start_time = time.time()
            
            # Get pipeline instance (initializes if needed)
            pipeline = self.get_pipeline()
            
            try:
                # Add torch.inference_mode() here to disable gradient tracking
                with torch.inference_mode():
                    results = pipeline(
                        audio_duration=duration,
                        prompt=music_prompt,
                        lyrics=lyrics,
                        infer_step=27,
                        guidance_scale=15.0,
                        scheduler_type="euler",
                        cfg_type="apg",
                        omega_scale=10.0,
                        batch_size=1
                    )
                
                audio_path = results[0]
                metadata = results[-1]
                
                # Get actual duration of generated audio file
                try:
                    import librosa
                    actual_duration = librosa.get_duration(filename=audio_path)
                    print(f"Generated audio duration: {actual_duration:.2f}s (target was {duration}s)")
                    
                    # If duration is significantly different, adjust metadata
                    if abs(actual_duration - duration) > 5.0:
                        print(f"Note: Actual duration differs from target by {abs(actual_duration - duration):.2f}s")
                        metadata['actual_duration'] = actual_duration
                        metadata['target_duration'] = duration
                except Exception as e:
                    print(f"Couldn't measure audio duration: {e}")
                    actual_duration = duration  # Fallback to target duration
                
                # Stage 3: Finalize song
                generation_time = time.time() - start_time
                print(f"\n[3/3] Song generated in {generation_time:.2f} seconds")
                self.generation_progress = 1.0
                
                # Generate artist name based on genre
                artist_names = {
                    "pop": "Pop AI",
                    "rock": "Rock Bot",
                    "hip hop": "MC Algorithm",
                    "electronic": "Synth AI",
                    "lofi": "Lo-Fi Generator",
                    "jazz": "Jazz AI",
                    "classical": "Classical Composer AI",
                    "ambient": "Ambient Creator",
                    "metal": "Metal Machine",
                    "reggae": "Digital Rasta",
                    "blues": "Blues Bot"
                }
                artist = artist_names.get(genre.lower(), "AI Musician")
                
                # Create song title from theme
                title = f"{theme.title()}"
                if random.random() > 0.5:  # 50% chance to add a numeric suffix
                    title += f" {random.randint(1, 100)}"
                
                # Store the actual duration in the Song object
                song = Song(
                    title=title,
                    artist=artist,
                    genre=genre,
                    theme=theme,
                    duration=actual_duration,  # Use actual duration here
                    lyrics=lyrics,
                    language=language,
                    prompt=music_prompt,
                    audio_path=audio_path,
                    generation_time=generation_time,
                    timestamp=time.time(),
                    metadata=metadata
                )
                
                print(f"\n=== Song Generated ===")
                print(f"Title: {song.title}")
                print(f"Artist: {song.artist}")
                print(f"Duration: {song.duration:.2f}s")
                print(f"Audio Path: {song.audio_path}")
                
                return song
                
            except Exception as e:
                print(f"Error during music generation: {e}")
                traceback.print_exc()
                raise
            finally:
                # Always release pipeline resources after use
                self.clean_all_memory()
                
        except Exception as e:
            print(f"Error during song generation: {e}")
            traceback.print_exc()
            raise
        finally:
            # Final cleanup regardless of success/failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Ensure progress is marked complete if we have a song
            if song is not None:
                self.generation_progress = 1.0
            else:
                self.generation_progress = 0.0
                    

def create_radio_interface(radio: AIRadioStation):
    """Create Gradio interface for the AI Radio Station"""
    
    def update_display():
        """Update the UI display with current radio status (without time tracking)"""
        current_song = radio.current_song
        
        # Format audio for playback if playing
        song_audio = None
        if current_song:
            song_audio = gr.Audio(
                value=current_song.audio_path,
                autoplay=True if radio.state == RadioState.PLAYING else False,
                waveform_options={
                    'sample_rate': 44100,  # Adjust if your audio uses different sample rate
                    'show_controls': True,
                }
            )

        # Format history for display with actual durations
        history_data = [
            [song.title, song.genre, song.theme, song.language, 
            f"{song.duration:.1f}s", 
            time.strftime('%H:%M:%S', time.localtime(song.timestamp))]
            for song in radio.history[-14:]  # Show last 14 songs
        ]
        
        # Create status messages
        buffer_msg = (
            f"Buffering: {radio.song_queue.qsize()}/{radio.min_buffer_size} songs" 
            if radio.state == RadioState.BUFFERING 
            else f"Buffer: {radio.song_queue.qsize()} songs"
        )
        
        # Get song data
        song_lyrics = current_song.lyrics if current_song else "No song playing"
        song_metadata = current_song.metadata if current_song else {}
        
        # Format duration display
        if current_song:
            # Convert duration to minutes:seconds format
            minutes = int(current_song.duration // 60)
            seconds = int(current_song.duration % 60)
            duration_display = f"{minutes}:{seconds:02d}"
        else:
            duration_display = "0:00"
        
        # Create status text with duration
        status_text = f"{radio.state.name}"
        if radio.state == RadioState.PLAYING and current_song:
            status_text += f" - {current_song.title} ({current_song.duration:.1f}s)"
        status_text += f" - {radio.song_queue.qsize()} songs queued"
        
        # Calculate progress percentage (0-100)
        progress_percent = radio.generation_progress * 100 if hasattr(radio, 'generation_progress') else 0
        
        return [
            radio.identity.name if radio.identity else "",
            status_text,
            radio.song_queue.qsize(),
            radio.state.name,
            song_audio,
            song_lyrics,
            song_metadata,
            duration_display,  # Formatted duration display
            buffer_msg,
            progress_percent,
            history_data
        ]

    def start_radio(genre, theme, duration, buff_size, model_path, language, max_history, tempo, intensity, mood, random_mode, random_languages):
        """Start the radio with the specified parameters"""
        # Update model if changed
        if model_path and hasattr(radio, 'llm') and radio.llm and model_path != radio.llm.model_path:
            try:
                from llama_cpp import Llama
                radio.llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=-1,
                    seed = -1 # random seed for random lyrics 
                )
                print(f"Loaded new LLM model from: {model_path}")
            except Exception as e:
                print(f"Failed to load new LLM model: {e}")
        
        # Convert duration to float explicitly
        duration = float(duration)
        print(f"Starting radio with requested duration: {duration}s")
        
        # Start the radio station
        radio.start_radio(
            genre=genre,
            theme=theme,
            duration=duration,
            buffer_size=buff_size,
            language=language,
            max_history=max_history,
            tempo=tempo,
            intensity=intensity,
            mood=mood,
            random_mode=random_mode,
            random_languages=random_languages 
        )
        
        return update_display()
    
    def surprise_me():
        """Generate random genre and theme combinations"""
        genres = list(THEME_SUGGESTIONS.keys())
        genre = random.choice(genres)
        theme = random.choice(THEME_SUGGESTIONS.get(genre, THEME_SUGGESTIONS["default"]))
        
        # duration = GENRE_DURATIONS.get(genre, GENRE_DURATIONS["default"])
        # buffer_size = 1
        language = random.choice(list(SUPPORTED_LANGUAGES.keys()))  
        # max_history = 50
        tempo = GENRE_TEMPOS.get(genre, GENRE_TEMPOS["default"])
        intensity = random.choice(["soft", "medium", "high"])
        mood = random.choice(["happy", "sad", "reflective", "upbeat", "chill"])
        
        current_model_path = radio.llm.model_path if hasattr(radio, 'llm') and radio.llm else "gemma-3-12b-it-abliterated.q4_k_m.gguf"
        
        return (
            genre, theme, current_model_path, language, tempo, intensity, mood,
            True,  # random_mode
            True,  # random_languages
            *update_display()
        )
    
    def stop_radio():
        """Stop all radio operations and clean up resources"""
        radio.stop_radio()
            # Return values that will stop the Gradio player
        return [
            "",  # station_output
            "STOPPED",  # status_output
            0,  # queue_size
            "STOPPED",  # state_display
            None,  # current_song_output (None stops playback)
            "No song playing",  # lyrics_output
            {},  # song_info
            "",  # playback_pos
            "Buffer: 0 songs",  # buffer_status
            0,  # generation_progress
            []  # history_display
        ]

    
    
    def update_theme_suggestions(genre):
        """Update theme suggestions based on selected genre"""
        return gr.Dropdown(
            choices=THEME_SUGGESTIONS.get(genre.lower(), THEME_SUGGESTIONS["default"]),
            value=random.choice(THEME_SUGGESTIONS.get(genre.lower(), THEME_SUGGESTIONS["default"]))
        )
    
    # Create the Gradio interface
    with gr.Blocks(title="AI Radio Station", theme="soft", css="""
        .custom-btn {
            background: var(--button-secondary-background-fill);
        }
        .custom-btn:hover {
            background: var(--button-secondary-background-fill-hover) !important;
        }
        .station-header {
            font-size: 1.5em;
            font-weight: bold;
            color: var(--primary-500);
        }
        .progress-bar {
            margin-top: 5px;
        }
    """
    ) as demo:
        gr.Markdown("# 🎵 AI Radio Station")
        gr.Markdown("Continuous AI-powered music generation using ACE")
        if radio.ollama:
            gr.Markdown("#### Ollama Model for Lyric Generation: `" + radio.llm_model_path + "`")

        # Add a timer component for automatic updates
        timer = gr.Timer(0.5, active=True)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Station Controls
                with gr.Group():
                    with gr.Tab("Basic Settings"):
                        genre_input = gr.Dropdown(
                            choices=list(THEME_SUGGESTIONS.keys()),
                            value="pop",
                            label="Music Genre"
                        )
                        theme_input = gr.Dropdown(
                            choices=THEME_SUGGESTIONS["pop"],
                            value="summer love",
                            label="Station Theme"
                        )
                        duration_input = gr.Slider(30, 600, value=120, label="Song Duration (seconds)")
                        buffer_size = gr.Slider(1, 10, value=1, step=1, label="Buffer Size (songs)")
                        random_mode = gr.Checkbox(label="Continuous Random Mode (after the first song)", value=True)
                        random_languages = gr.Checkbox(label="Randomize Languages (after the first song)", value=False)
                        if radio.ollama:
                            model_path_input = gr.File(
                                label="GGUF Model File (Ollama model used instead)",
                                file_types=[".gguf"],
                                value="gemma-3-12b-it-abliterated.q4_k_m.gguf"
                            )
                        else:
                            model_path_input = gr.File(
                                label="GGUF Model File",
                                file_types=[".gguf"],
                                value="gemma-3-12b-it-abliterated.q4_k_m.gguf"
                            )

                    with gr.Tab("Advanced Settings"):
                        language_input = gr.Dropdown(
                            choices=list(SUPPORTED_LANGUAGES.keys()),
                            value="English",
                            label="Lyrics Language"
                        )
                        max_history = gr.Slider(10, 200, value=12, step=10, label="Max History Size")
                        tempo_input = gr.Slider(60, 200, value=120, step=5, label="Tempo (BPM)")
                        intensity_input = gr.Dropdown(
                            choices=["soft", "medium", "high"],
                            value="medium",
                            label="Intensity"
                        )
                        mood_input = gr.Dropdown(
                            choices=["happy", "sad", "reflective", "angry", "upbeat", "chill"],
                            value="upbeat",
                            label="Mood"
                        )
                    
                    with gr.Row():
                        start_btn = gr.Button("Start Radio", elem_classes="custom-btn")
                        stop_btn = gr.Button("Stop", elem_classes="custom-btn")
                    
                    with gr.Row():
                        refresh_btn = gr.Button("Refresh Display")
                        surprise_btn = gr.Button("Surprise Me!")
                
                # Status Display
                with gr.Group():
                    station_output = gr.Textbox(label="Current Station", elem_classes="current_station")
                    status_output = gr.Textbox(label="Status")
                    queue_size = gr.Number(label="Songs in Queue", visible=False)
                    state_display = gr.Textbox(label="Player State", visible=False)
                    playback_pos = gr.Textbox(label="Playback Position", visible=False)
                    buffer_status = gr.Textbox(label="Buffer Status", visible=False)
                    
                    # gr.Markdown("### Generation Progress")
                    generation_progress = gr.Slider(0, 100, value=0, interactive=False, elem_classes="progress-bar", visible=False)
            
            with gr.Column(scale=2):
                # Now Playing Display
                with gr.Group():
                    current_song_output = gr.Audio(
                    label="Now Playing",
                    interactive=False,
                    autoplay=True,
                    visible=True,
                    waveform_options={
                    'sample_rate': 44100,  # Or your actual sample rate
                    'show_controls': True,
                    }
                    )
                    with gr.Tabs():
                        with gr.TabItem("Lyrics"):
                            lyrics_output = gr.Textbox(label="Lyrics", lines=20, interactive=False)
                        with gr.TabItem("Song Details"):
                            song_info = gr.JSON(label="Song Info")
                        with gr.TabItem("History"):
                            history_display = gr.Dataframe(
                                headers=["Title", "Genre", "Theme", "Language", "Duration", "Generated"],
                                interactive=False,
                                label="Recent Songs"
                            )

        # Define output components list
        output_components = [
            station_output, 
            status_output, 
            queue_size, 
            state_display,
            current_song_output,
            lyrics_output,
            song_info,
            playback_pos,
            buffer_status,
            generation_progress,
            history_display
        ]

        # Connect UI elements
        start_btn.click(
            fn=start_radio,
            inputs=[genre_input, theme_input, duration_input, buffer_size, model_path_input, 
                language_input, max_history, tempo_input, intensity_input, mood_input, random_mode, random_languages],
            outputs=output_components
        )
        
        stop_btn.click(
            fn=stop_radio,
            outputs=output_components
        )
        
        
        refresh_btn.click(
            fn=update_display,
            outputs=output_components
        )
        
        surprise_btn.click(
            fn=surprise_me,
            outputs=[genre_input, theme_input, model_path_input,
                    language_input, tempo_input, intensity_input, mood_input, random_mode, random_languages,
                    *output_components]
        )
        
        # Update theme suggestions when genre changes
        genre_input.change(
            fn=update_theme_suggestions,
            inputs=genre_input,
            outputs=theme_input
        )
        
        # Set up automatic updates
        timer.tick(
            fn=update_display,
            outputs=output_components
        )
        
        # Initial state
        demo.load(
            fn=lambda: ["", "STOPPED - 0 songs queued", 0, "STOPPED", 
                    None, "No song playing", {}, "0s / 0s", "Buffer: 0 songs", 0, []],  # Add empty list for history
            outputs=output_components
        )
        
    return demo

def main():
    parser = argparse.ArgumentParser(description="AI Radio Station using ACE")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints",
                       help="Path to ACEStepPipeline checkpoints")
    parser.add_argument("--model_path", type=str, default="gemma-3-12b-it-abliterated.q4_k_m.gguf",
                       help="Path to LLM model for lyric generation")
    parser.add_argument("--server_name", type=str, default="0.0.0.0",
                       help="Server address to bind to")
    parser.add_argument("--port", type=int, default=7865,
                       help="Port to run the server on")
    parser.add_argument("--device_id", type=int, default=0,
                       help="GPU device ID to use")
    parser.add_argument("--share", default=False,
                       help="Share the Gradio interface publicly")
    parser.add_argument("--bf16", default=True,
                       help="Use bfloat16 precision")
    parser.add_argument("--torch_compile", default=False,
                       help="Enable torch compilation for faster inference")
    parser.add_argument("--ollama", default=False,
                       help="Enable Ollama for lyric generation")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    
    # Initialize ACE-Step pipeline
    print("Initializing ACEStepPipeline...")
    pipeline = ACEStepPipeline(
        checkpoint_dir=args.checkpoint_path,
        dtype="bfloat16" if args.bf16 else "float32",
        torch_compile=args.torch_compile
    )
    

    # Initialize AI Radio Station
    print("Initializing AI Radio Station...")
    radio = AIRadioStation(
        ace_step_pipeline=pipeline,
        model_path=args.model_path,
        ollama=args.ollama
    )
    
    # Create and launch interface
    print("Launching interface...")
    interface = create_radio_interface(radio)
    interface.queue()
    interface.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share
    )

if __name__ == "__main__":
    main()
