#!/usr/bin/env python3
"""
Generate LoRA training images using Flux 2.0 + qwen3.6:35b with human approval loop.

Usage:
    python scripts/generate_training_images.py --character max
    python scripts/generate_training_images.py --character zoe
    python scripts/generate_training_images.py --character mom
    python scripts/generate_training_images.py --character dad
    python scripts/generate_training_images.py --character max --auto-approve  # CI mode

This script:
1. Reads prompts from assets/kids_duo/training/{character}_prompts.txt
2. Uses qwen3.6:35b to refine prompts for Flux 2.0 compatibility
3. Generates images via ComfyUI + Flux 2.0 (no LoRA, base model)
4. Shows human approval web UI for each image
5. Saves approved images to assets/kids_duo/training/images/{character}/
6. Auto-generates caption .txt files for each approved image

Flow:
    Load prompts → LLM refine → Flux generate → Human review → Save approved
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_app_config
from core.models.common import HealthStatus

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


async def check_services() -> bool:
    """Verify Ollama + ComfyUI are running."""
    try:
        import httpx
    except ImportError:
        logger.error("❌ httpx not installed. Run: pip install httpx")
        return False

    checks = {
        "Ollama (qwen3.6:35b)": "http://localhost:11434/api/tags",
        "ComfyUI (Flux 2.0)": "http://localhost:8188/system_stats",
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in checks.items():
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                logger.info(f"✅ {name} — OK")
            except Exception as exc:
                logger.error(f"❌ {name} — FAILED: {exc}")
                logger.error(f"   Start services: bash scripts/start_services.sh")
                return False
    return True


def load_prompts(character: str) -> list[tuple[str, str]]:
    """Load prompts from {character}_prompts.txt.
    
    Returns: [(prompt_id, prompt_text), ...]
    Example: [("001", "full body front view, kids_duo_max..."), ...]
    """
    prompt_file = Path(f"assets/kids_duo/training/{character}_prompts.txt")
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    prompts = []
    content = prompt_file.read_text()
    
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            # Format: [001] prompt text here
            if "]" in line:
                prompt_id = line[1:line.index("]")].strip()
                prompt_text = line[line.index("]")+1:].strip()
                prompts.append((prompt_id, prompt_text))
    
    logger.info(f"📋 Loaded {len(prompts)} prompts for character '{character}'")
    return prompts


async def refine_prompt_for_flux(prompt: str, character: str) -> str:
    """Use qwen3.6:35b to adapt the prompt for Flux 2.0 base model (no LoRA).
    
    The original prompts were written for Leonardo.ai. We need to:
    - Remove the trigger token (kids_duo_max / kids_duo_zoe) since we're using base Flux
    - Keep all visual descriptors (appearance, clothing, pose, expression)
    - Add Flux-specific quality boosters
    - Maintain consistency markers
    """
    try:
        import httpx
    except ImportError:
        # Fallback: return original prompt
        return prompt
    
    system_prompt = f"""You are a prompt engineer for Flux 2.0 Dev image generation.

Task: Adapt training image prompts from Leonardo.ai format to Flux 2.0 base model format.

Rules:
1. REMOVE the trigger token "kids_duo_{character}" (we're using base Flux, not LoRA)
2. KEEP ALL visual descriptors exactly as written (skin tone, hair, clothing, pose, expression)
3. ADD Flux quality boosters: "high quality digital illustration, professional character design"
4. Keep "clean white background" requirement
5. Output ONLY the refined prompt, no explanation

Input prompt format example:
"[001] full body front view, kids_duo_max, 5-year-old cartoon boy, round friendly face..."

Output format (remove [001] prefix and kids_duo_max token):
"full body front view, 5-year-old cartoon boy, round friendly face... high quality digital illustration, professional character design, clean white background"
"""
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": "qwen3.6:35b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Refine this prompt:\n{prompt}"},
                ],
                "temperature": 0.3,
                "max_tokens": 512,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        refined = data["choices"][0]["message"]["content"].strip()
        
        # Remove any [NNN] prefix if LLM kept it
        if refined.startswith("["):
            refined = refined[refined.index("]")+1:].strip()
        
        return refined


async def generate_image_flux(prompt: str, seed: int, output_path: Path) -> Path:
    """Generate image using ComfyUI + Flux 2.0 base model (no LoRA).
    
    Uses the workflow template but with empty LoRA name to use base model only.
    """
    import httpx
    import uuid
    import time
    
    workflow_template = Path("assets/comfyui_workflows/flux_lora_txt2img.json")
    if not workflow_template.exists():
        raise FileNotFoundError(f"Workflow template not found: {workflow_template}")
    
    workflow = json.loads(workflow_template.read_text())
    
    # Find nodes by title and update them
    for node_id, node in workflow.items():
        title = node.get("_meta", {}).get("title", "")
        
        if title == "__PROMPT__":
            node["inputs"]["text"] = prompt
        elif title == "__LORA_NAME__":
            # Empty string = use base model only
            node["inputs"]["lora_name"] = ""
            node["inputs"]["strength_model"] = 0.0
            node["inputs"]["strength_clip"] = 0.0
        elif title == "__WIDTH__":
            node["inputs"]["width"] = 768
        elif title == "__HEIGHT__":
            node["inputs"]["height"] = 768
        elif title == "__STEPS__":
            node["inputs"]["steps"] = 20
        elif title == "__SEED__":
            node["inputs"]["seed"] = seed
    
    # Submit to ComfyUI
    client_id = str(uuid.uuid4())
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            "http://localhost:8188/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
        
        logger.info(f"  ⏳ Generating image (prompt_id: {prompt_id})...")
        
        # Poll for completion
        for _ in range(60):  # 60 * 5s = 5 min timeout
            await asyncio.sleep(5)
            
            resp = await client.get(f"http://localhost:8188/history/{prompt_id}")
            resp.raise_for_status()
            history = resp.json()
            
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        images = node_output["images"]
                        if images:
                            # Download first image
                            img = images[0]
                            img_url = f"http://localhost:8188/view?filename={img['filename']}&subfolder={img.get('subfolder', '')}&type={img['type']}"
                            
                            img_resp = await client.get(img_url)
                            img_resp.raise_for_status()
                            
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            output_path.write_bytes(img_resp.content)
                            logger.info(f"  ✅ Image saved: {output_path}")
                            return output_path
        
        raise TimeoutError("Image generation timed out after 5 minutes")


async def show_approval_ui(image_path: Path, prompt: str, prompt_id: str) -> tuple[bool, Optional[str]]:
    """Show web UI for human to approve/reject image.
    
    Returns: (approved: bool, notes: Optional[str])
    """
    from fastapi import FastAPI, Form
    from fastapi.responses import HTMLResponse, FileResponse
    import uvicorn
    import asyncio
    
    app = FastAPI()
    result = {"approved": None, "notes": None}
    
    @app.get("/")
    async def show_image():
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Image Approval — {prompt_id}</title>
            <style>
                body {{ font-family: system-ui; max-width: 1200px; margin: 40px auto; padding: 20px; }}
                img {{ max-width: 100%; border: 2px solid #ddd; border-radius: 8px; }}
                .prompt {{ background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .buttons {{ margin: 30px 0; }}
                button {{ padding: 12px 30px; font-size: 16px; margin-right: 10px; border-radius: 6px; cursor: pointer; }}
                .approve {{ background: #28a745; color: white; border: none; }}
                .reject {{ background: #dc3545; color: white; border: none; }}
                textarea {{ width: 100%; padding: 10px; margin: 10px 0; border-radius: 6px; }}
            </style>
        </head>
        <body>
            <h1>Training Image Approval</h1>
            <h2>Prompt ID: {prompt_id}</h2>
            
            <img src="/image" alt="Generated image">
            
            <div class="prompt">
                <strong>Prompt:</strong><br>
                {prompt}
            </div>
            
            <form method="POST" action="/approve">
                <div class="buttons">
                    <button type="submit" class="approve">✅ Approve (Use for Training)</button>
                </div>
            </form>
            
            <form method="POST" action="/reject">
                <label><strong>Rejection reason (optional):</strong></label>
                <textarea name="notes" rows="3" placeholder="e.g., wrong pose, anatomy issues, background not clean..."></textarea>
                <div class="buttons">
                    <button type="submit" class="reject">❌ Reject (Regenerate Later)</button>
                </div>
            </form>
        </body>
        </html>
        """
        return HTMLResponse(html)
    
    @app.get("/image")
    async def get_image():
        return FileResponse(image_path)
    
    @app.post("/approve")
    async def approve():
        result["approved"] = True
        return HTMLResponse("<h1>✅ Approved!</h1><p>Window will close...</p><script>setTimeout(() => window.close(), 1000)</script>")
    
    @app.post("/reject")
    async def reject(notes: str = Form("")):
        result["approved"] = False
        result["notes"] = notes
        return HTMLResponse("<h1>❌ Rejected</h1><p>Window will close...</p><script>setTimeout(() => window.close(), 1000)</script>")
    
    # Start server
    config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="warning")
    server = uvicorn.Server(config)
    
    # Run server in background, wait for approval
    task = asyncio.create_task(server.serve())
    
    logger.info(f"\n🌐 Open browser: http://localhost:8765")
    logger.info(f"   Approve or reject the image for prompt {prompt_id}\n")
    
    # Wait for user decision
    while result["approved"] is None:
        await asyncio.sleep(0.5)
    
    # Shutdown server
    server.should_exit = True
    await task
    
    return result["approved"], result["notes"]


async def main():
    parser = argparse.ArgumentParser(description="Generate LoRA training images with Flux 2.0")
    parser.add_argument("--character", required=True, choices=["max", "zoe", "mom", "dad"], help="Character to generate")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve all images (CI mode)")
    parser.add_argument("--start-from", type=str, help="Resume from specific prompt ID (e.g., '005')")
    args = parser.parse_args()
    
    logger.info(f"\n🎨 Training Image Generator — {args.character.upper()}\n")
    
    # Check services
    if not await check_services():
        sys.exit(1)
    
    # Load prompts
    prompts = load_prompts(args.character)
    
    # Filter if resuming
    if args.start_from:
        prompts = [(pid, p) for pid, p in prompts if pid >= args.start_from]
        logger.info(f"📍 Resuming from prompt {args.start_from}")
    
    output_dir = Path(f"assets/kids_duo/training/images/{args.character}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    approved_count = 0
    rejected_count = 0
    
    for prompt_id, original_prompt in prompts:
        logger.info(f"\n{'='*70}")
        logger.info(f"Prompt {prompt_id}/{len(prompts)}: {original_prompt[:80]}...")
        
        # Step 1: Refine prompt for Flux
        refined_prompt = await refine_prompt_for_flux(original_prompt, args.character)
        logger.info(f"  🔧 Refined: {refined_prompt[:100]}...")
        
        # Step 2: Generate image
        output_path = output_dir / f"{args.character}_{prompt_id}.png"
        seed = int(prompt_id) + (1000 if args.character == "zoe" else 0)
        
        await generate_image_flux(refined_prompt, seed, output_path)
        
        # Step 3: Human approval (or auto-approve)
        if args.auto_approve:
            approved = True
            notes = None
            logger.info(f"  ✅ Auto-approved (--auto-approve mode)")
        else:
            approved, notes = await show_approval_ui(output_path, refined_prompt, prompt_id)
        
        if approved:
            # Save caption file
            caption_path = output_path.with_suffix(".txt")
            caption_path.write_text(refined_prompt)
            logger.info(f"  ✅ APPROVED — saved with caption")
            approved_count += 1
        else:
            # Delete rejected image
            output_path.unlink()
            logger.info(f"  ❌ REJECTED — {notes or 'no reason given'}")
            rejected_count += 1
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ Complete! Approved: {approved_count} | Rejected: {rejected_count}")
    logger.info(f"📁 Images saved to: {output_dir}")
    logger.info(f"\n📋 Next step: Train LoRA with kohya_ss")
    logger.info(f"   cd /workspace/sd-scripts")
    logger.info(f"   accelerate launch flux_train_network.py --config_file ../video_me/assets/kids_duo/training/kohya_config.toml")


if __name__ == "__main__":
    asyncio.run(main())
