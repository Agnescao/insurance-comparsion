from pymilvus import MilvusClient
if __name__ == "__main__":
    c = MilvusClient(uri="tcp://121.41.85.215:19530", user="root", password="Milvus")
    print(c.list_collections())