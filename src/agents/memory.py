from mem0 import Memory



class MemoryInstance():
    def __init__(self):
        self.mem = Memory()

    def add_to_memory(self, inputs: str, user_id: str, meta_data: dict):
        self.mem.add(inputs, user_id=user_id, meta_data=meta_data)

    def search(self, query: str, user_id: str):
        related_memories = self.mem.search(query=query, user_id=user_id)
        memory_context = "\n".join(memory['memory'] for memory in related_memories )
        return memory_context

# mem = Memory()

# result = mem.add("Likes to play cricket on weekends", user_id="alice", metadata={"category": "hobbies"})
# related_memories = mem.search(query="What are Alice's hobbies?", user_id="alice")
#
# print(related_memories)