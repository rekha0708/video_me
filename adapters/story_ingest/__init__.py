"""Story ingestion: turn user-provided story text into a TranscribeResult.

Used by the dashboard worker to pre-seed the transcribe artifact for
kind="story"/"story_images" jobs so the pipeline skips yt-dlp + Whisper.
"""
