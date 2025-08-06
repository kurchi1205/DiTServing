import asyncio
import aiohttp

async def start_background_process():
    url = "http://localhost:8000/start_background_process"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params={"model": ""}) as resp:
                print("Started background process:", await resp.text())
    except Exception as e:
        print("Error:", e)

if __name__=="__main__":
    asyncio.run(start_background_process())
