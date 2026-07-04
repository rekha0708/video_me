## Run 2026-07-04 01:57 — Phase: all — Job: 20260704-015745-b7o
Source: file:///workspace/downloads/learn_body_parts_with_rosie_fun_kids_act.f399.mp4
Adapter override: VIDEO_ME_VIDEO_ADAPTER=wan (dashboard API + worker restarted to apply; no lip-sync expected per confirmed MuseTalk limitation on cartoon faces)
Status: submitted

## Run 2026-07-04 01:57 — Phase: all — Job: 20260704-015745-b7o
Status: failed → fixed
Stage failed: fetch_media  Error: FileNotFoundError
Fix: Not a code bug — source.url pointed at .f399.mp4 which only existed in project-local /workspace/video_me/downloads/, not canonical /workspace/downloads/ (core/config.py:34). Resubmitted pointing at the correctly-placed canonical file.
Lesson written: yes
Retry outcome: see job 20260704-020418-33o

## Run 2026-07-04 02:04 — Phase: all — Job: 20260704-020418-33o
Source: file:///workspace/downloads/learn_body_parts_with_rosie_fun_kids_act.mp4
Adapter override: VIDEO_ME_VIDEO_ADAPTER=wan (confirmed via /api/config/defaults)
Status: submitted
