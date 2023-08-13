import asyncio # default lib
from typing import AsyncIterable, Awaitable

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel

from langchain.schema import HumanMessage, AIMessage

app = FastAPI()


origins = [
    "http://localhost:3000",  # Adjust this to your frontend's address
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def send_message(messages) -> AsyncIterable[str]:
    """
    Logic of streaming response:
        1. Asynccall back gets the output from llm 
        2. It generates iterative tokens from the output and yields the tokens to the stream
    """
    callback = AsyncIteratorCallbackHandler()
    
    llm = ChatOpenAI(
        streaming=True,
        # verbose=True,
        callbacks=[callback],
    )

    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            # TODO: handle exception
            print(f"Caught exception: {e}")
        finally:
            # Signal the aiter to stop.
            event.set()

    # Begin a task that runs in the background.
    task = asyncio.create_task(wrap_done(
        llm.agenerate(messages=[messages]),
        callback.done),
    )

    async for token in callback.aiter():
        # Use server-sent-events to stream the response
        yield f"{token}"

    await task


class StreamRequest(BaseModel):
    """Request body for streaming."""
    message: str


@app.post("/api/chat")
async def stream(request: Request):
    data = await request.json()
    messages = data['messages']
    print("messages", messages)
    previewToken = data['previewToken']

    # need authentication layer here ##############
    # like we did in js
    # const userId = "uid" + (await auth())?.user.id
    # if (!userId) {
    #     return new Response('Unauthorized', {
    #     status: 401
    #     })
    # }
    ################################################

    if previewToken:
        openai_api_key = previewToken

    llm_messages = [
        HumanMessage(content=m['content']) if m['role'] == "user" 
        else AIMessage(content=m['content'])
        for m in messages
    ]

    print("llm_messages", llm_messages)

    return StreamingResponse(send_message(llm_messages), media_type="text/event-stream")
