import asyncio

async def main():
    n_queues = 3
    qs = [asyncio.Queue() for i in range(n_queues)]
    values = list(range(n_queues))

    tasks = [asyncio.create_task(processing(q)) for q in qs]
    iterations = 10
    for i in range(n_queues):
        await qs[i].put(iterations)

    for i in range(iterations):
        for i in range(n_queues):
            await qs[i].put(values[i])


async def processing(q):
    m = 0
    i=0
    n = await q.get()
    for i in range(n):
        v = await q.get()
        m+=v
        i+=1
    print(m)


if __name__ == '__main__':
    asyncio.run(main())