import time
import uuid
import json
import gzip
import numpy as np
from collections import defaultdict
from datetime import datetime
from sentence_transformers import SentenceTransformer

class MemoryNode:
    def __init__(self, id: str, text: str, timestamp: float, embedding: list):
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.embedding = embedding

class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}
        self.index = defaultdict(list)

    def add_node(self, embedding: list, text: str, timestamp: float) -> str:
        node_id = str(uuid.uuid4())
        self.nodes[node_id] = MemoryNode(node_id, text, timestamp, embedding)
        self.index['default'].append(node_id)
        return node_id

    def search(self, query_embed: list, alpha=0.7, beta=0.3, base_decay=0.05) -> list:
        results = []
        current_time = time.time()
        
        for node_id in self.index['default']:
            node = self.nodes[node_id]
            sim = self.cosine_similarity(query_embed, node.embedding)
            decay = base_decay * (1 + np.log1p(current_time - node.timestamp))
            recency = np.exp(-decay)
            score = alpha * sim + beta * recency
            results.append((score, node))
            
        return sorted(results, key=lambda x: x[0], reverse=True)

    @staticmethod
    def cosine_similarity(a: list, b: list) -> float:
        a_np, b_np = np.array(a), np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

class ToxicityDetector:
    def __init__(self, threshold=0.85):
        self.threshold = threshold
    
    def evaluate(self, text):
        score = self.mock_score(text)
        return score < self.threshold

    def mock_score(self, text):
        text_lower = text.lower()
        if "explosif" in text_lower or "danger" in text_lower:
            return 0.92
        return 0.15

class ContextMemory:
    def __init__(self):
        self.graph = KnowledgeGraph()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.toxicity_detector = ToxicityDetector()

    def store(self, text: str) -> str:
        if not self.toxicity_detector.evaluate(text):
            return None
        embedding = self.encoder.encode(text).tolist()
        return self.graph.add_node(embedding, text, time.time())

    def recall(self, query: str, top_k: int = 3) -> list:
        query_embed = self.encoder.encode(query).tolist()
        return self.graph.search(query_embed)[:top_k]

    def feedback(self, node_id: str, correction: str) -> bool:
        if node_id not in self.graph.nodes or not self.toxicity_detector.evaluate(correction):
            return False
        self.graph.nodes[node_id].text = correction
        return True
