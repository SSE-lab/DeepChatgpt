from flask import Flask, request, jsonify
from memory import ContextMemory

app = Flask(__name__)
mem = ContextMemory()

@app.route("/store", methods=["POST"])
def store():
    data = request.json
    node_id = mem.store(data.get("text", ""))
    return jsonify({
        "status": "success" if node_id else "rejected",
        "node_id": node_id
    })

@app.route("/recall", methods=["GET"])
def recall():
    query = request.args.get("q", "")
    results = mem.recall(query)
    return jsonify([{
        "id": node.id,
        "text": node.text[:100],
        "score": float(score)
    } for score, node in results])

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    success = mem.feedback(
        data.get("node_id", ""),
        data.get("correction", "")
    )
    return jsonify({"status": "success" if success else "failed"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
