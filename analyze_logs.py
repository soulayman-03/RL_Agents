import json
from collections import Counter
import sys

def analyze_fails(log_path):
    total_episodes = 0
    failed_episodes = 0
    fail_reasons_counter = Counter()
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                total_episodes += 1
                if data.get("ep_failed", False):
                    failed_episodes += 1
                    reasons = data.get("fail_reasons", {})
                    for reason, count in reasons.items():
                        fail_reasons_counter[reason] += count
                        
        print(f"Total Episodes: {total_episodes}")
        print(f"Failed Episodes: {failed_episodes}")
        print(f"Success Rate: {(1 - failed_episodes/total_episodes)*100:.2f}%")
        print("\nFail Reasons (Total counts across all failed episodes):")
        for reason, count in fail_reasons_counter.most_common():
            print(f"- {reason}: {count}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_fails(sys.argv[1])
    else:
        print("Please provide a log path.")
