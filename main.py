"""
Main entry point for the blockchain stablecoin explorer application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Blockchain Stablecoin Explorer",
    description="Autonomous agent for collecting and analyzing stablecoin usage data",
    version="1.0.0"
)

# CORS configuration will be added when implementing API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will be configured from environment variables
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "ok",
        "service": "Blockchain Stablecoin Explorer",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
