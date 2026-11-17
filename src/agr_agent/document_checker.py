def collect_documents(matches):
    docs = []
    for item in matches:
        for doc in item.get("required_documents", []):
            if doc not in docs:
                docs.append(doc)
    return docs
