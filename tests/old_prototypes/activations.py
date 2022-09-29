import asyncio

class Producer:
    async def produce(self):
        i = 0
        while True:
            i+=1
            await asyncio.sleep(1)
            yield i

class Consumer:
    async def consume(self,producer):
        while True:
            r = await producer.produce()
            print(r)

async def example2():
    p = Producer()
    c = Consumer()
    await c.consume(p)

async def count():
    print("One")
    await asyncio.sleep(1)
    print("Two")

async def main():
    await asyncio.gather(count(), count(), count())
def example1():
    import time
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.3f} seconds.")

if __name__ == "__main__":
    asyncio.run(example2())