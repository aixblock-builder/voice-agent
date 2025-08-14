from services_manager import *

tts_config = {"engine_name": "kokoro", "engine_kwargs": {"voice": "af_heart"}}


async def start_tts_service():
    tts_id = await start_service(
        name="tts",
        config=tts_config,
        health_url="http://127.0.0.1:1006/health-check",
        run_fn_blocking=run_tts_app_func,
        stop_fn=stop_tts_app,
    )

    print(f"[TTS] Service started with ID: {tts_id}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(start_tts_service())