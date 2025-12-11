from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WS connected!")

    while True:
        try:
            data = await websocket.receive_text()
            print("Received chunk:", len(data))
        except Exception as e:
            print("WS error:", e)
            break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=9000)

