"""Исправление вводного примера: ID не зависит от числа живых записей."""

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

app = FastAPI()
ITEMS = {}
NEXT_ID = 1


class ItemIn(BaseModel):
    name: str = Field(min_length=1)
    price: float = Field(gt=0)


@app.post("/items", status_code=201)
def create_item(item: ItemIn):
    global NEXT_ID
    item_id = NEXT_ID
    NEXT_ID += 1
    ITEMS[item_id] = item.model_dump()
    return {"id": item_id, **ITEMS[item_id]}


@app.delete("/items/{item_id}", status_code=204)
def delete_item(item_id: int):
    if item_id not in ITEMS:
        raise HTTPException(status_code=404, detail="Not found")
    del ITEMS[item_id]
    return Response(status_code=204)
