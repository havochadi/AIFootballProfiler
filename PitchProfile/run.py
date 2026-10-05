"""Launch the local application. No cloud account or paid service is required."""
import argparse
import asyncio
import sys
import uvicorn

if __name__=="__main__":
    # Browser video seeks cancel range requests. Windows' Proactor loop reports
    # connection-reset callback errors for these normal cancellations.
    if sys.platform=="win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    p=argparse.ArgumentParser(description="Run PitchProfile locally")
    p.add_argument("--port",type=int,default=8000)
    p.add_argument("--host",default="127.0.0.1")
    a=p.parse_args()
    uvicorn.run("football_profiler.app:app",host=a.host,port=a.port)
