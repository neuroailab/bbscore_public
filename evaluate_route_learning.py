import sys
import numpy as np
import matplotlib.pyplot as plt

# Optional dependency check
try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False

from benchmarks import BENCHMARK_REGISTRY

def main():
    print("Initializing Route Learning Brain-to-Brain Analysis...")
    
    # We will analyze Exp1 subject pairs. 
    # To keep the analysis fast, we select the first 10 subjects.
    exp1_subjects = [f"{i:02d}" for i in range(1, 11)]
    
    results = np.zeros((len(exp1_subjects), len(exp1_subjects)))
    
    print(f"Running cross-subject benchmarks for {len(exp1_subjects)} subjects...")
    for i, src in enumerate(exp1_subjects):
        for j, tgt in enumerate(exp1_subjects):
            if i == j:
                results[i, j] = 1.0 # Perfect consistency with oneself
                continue
            
            benchmark_name = f"RL_Exp1Sub{src}_to_Exp1Sub{tgt}"
            if benchmark_name not in BENCHMARK_REGISTRY:
                print(f"Warning: {benchmark_name} not found in registry. Skipping.")
                continue
                
            try:
                # Instantiate mapping benchmark
                scorer_cls = BENCHMARK_REGISTRY[benchmark_name]
                scorer = scorer_cls(debug=False)
                
                # In BBScore, regression metrics like 'pls' or 'ridge' are commonly used for 
                # assembly-to-assembly mappings. Let's use 'pls' for cross-subject predictions.
                scorer.add_metric('pls')
                
                # Run the benchmark. `run()` automatically calls get_assembly on both source and target 
                # and computes the added metrics.
                out = scorer.run()
                
                # Extract the score from the dictionary that `run()` returns
                # Format is typically {'metrics': {'pls': score_object}, 'ceiling': array...}
                score = out['metrics']['pls']
                
                # Average the raw pearson correlation slice (if it's a BrainScore-like object or float)
                if hasattr(score, 'raw'):
                    raw_score = score.raw.values.mean()
                else:
                    raw_score = float(score)  # Fallback if metric returns a scalar
                
                results[i, j] = raw_score
                
                print(f"[{src} -> {tgt}] Average Score: {raw_score:.4f}")
            except Exception as e:
                print(f"[{src} -> {tgt}] Error: {e}")
                results[i, j] = np.nan
                
    # Plot results
    plt.figure(figsize=(10, 8))
    labels = [f"Sub{s}" for s in exp1_subjects]
    
    if SNS_AVAILABLE:
        sns.heatmap(results, annot=True, fmt=".2f", cmap="viridis",
                    xticklabels=labels, yticklabels=labels)
    else:
        plt.imshow(results, cmap="viridis")
        plt.colorbar()
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
        plt.yticks(ticks=np.arange(len(labels)), labels=labels)
        
        # Add text annotations if seaborn isn't available
        for i in range(len(labels)):
            for j in range(len(labels)):
                if not np.isnan(results[i, j]):
                    plt.text(j, i, f"{results[i, j]:.2f}", 
                             ha="center", va="center", color="w" if results[i, j] < 0.5 else "k")

    plt.title("Brain-to-Brain Mapping Consistency (Early vs Late Learning)")
    plt.xlabel("Target Subject")
    plt.ylabel("Source Subject")
    plt.tight_layout()
    plt.savefig("route_learning_mapping_consistency.png")
    print("Saved plot to route_learning_mapping_consistency.png")

if __name__ == "__main__":
    main()
