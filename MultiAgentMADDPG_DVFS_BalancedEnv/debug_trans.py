import json

path = r"results\models_1hugcnn_1cnn15_1miniresnet_1resnet18_1vgg11_1deepcnn_1lenet_p3_linFL_t0.7_e0.5k_a0.4_b0.6\sl_1p00\episode_impact.jsonl"
try:
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} episodes.")
    for ep_idx in [0, 10, 100, 1000, 4999]:
        if ep_idx < len(data):
            row = data[ep_idx]
            agent_sums = {}
            for alloc in row.get("allocations", []):
                a = alloc.get("agent")
                if a not in agent_sums:
                    agent_sums[a] = 0.0
                agent_sums[a] += alloc.get("trans_data", 0.0)
            print(f"Episode {row.get('episode')} (Failed: {row.get('ep_failed')}) Trans data sums: {agent_sums}")
            
except Exception as e:
    print("Error:", e)
