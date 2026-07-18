import time
import tracemalloc
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.utils.logger import get_logger

logger = get_logger(__name__)

class Profiler:
    """Measures pipeline performance: runtime and memory usage."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Profiler, cls).__new__(cls)
            cls._instance.stages = []
            cls._instance.current_stage = None
            cls._instance.start_time = None
            tracemalloc.start()
        return cls._instance

    def start_stage(self, stage_name: str) -> None:
        if self.current_stage is not None:
            self.end_stage()
            
        self.current_stage = stage_name
        self.start_time = time.time()
        tracemalloc.reset_peak()
        logger.info(f"Profiler: Started stage '{stage_name}'")

    def end_stage(self) -> None:
        if self.current_stage is None:
            return
            
        elapsed = time.time() - self.start_time
        _, peak_mem = tracemalloc.get_traced_memory()
        
        self.stages.append({
            "stage": self.current_stage,
            "runtime_sec": elapsed,
            "peak_memory_mb": peak_mem / (1024 * 1024)
        })
        
        logger.info(f"Profiler: Ended stage '{self.current_stage}' ({elapsed:.2f}s, {peak_mem / (1024 * 1024):.2f}MB peak)")
        self.current_stage = None
        self.start_time = None

    def generate_report(self, output_dir: Path) -> None:
        self.end_stage() # Ensure last stage is closed
        
        logger.info("Generating Performance Profile")
        md_path = output_dir / "reports" / "performance_profile.md"
        
        total_time = sum(s["runtime_sec"] for s in self.stages)
        max_mem = max((s["peak_memory_mb"] for s in self.stages), default=0)
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Pipeline Performance Profile\n\n")
            
            f.write("## Overall Metrics\n")
            f.write(f"- **Total Measured Runtime**: {total_time:.2f} seconds\n")
            f.write(f"- **Global Peak Memory**: {max_mem:.2f} MB\n\n")
            
            f.write("## Stage Breakdown\n")
            f.write("| Stage | Runtime (s) | Peak Memory (MB) | % of Total Time |\n")
            f.write("|---|---|---|---|\n")
            for s in self.stages:
                pct = (s["runtime_sec"] / total_time * 100) if total_time > 0 else 0
                f.write(f"| {s['stage']} | {s['runtime_sec']:.2f} | {s['peak_memory_mb']:.2f} | {pct:.1f}% |\n")
                
            f.write("\n## Observations\n")
            if not self.stages:
                f.write("No stages were profiled.\n")
            else:
                slowest = max(self.stages, key=lambda x: x["runtime_sec"])
                hungriest = max(self.stages, key=lambda x: x["peak_memory_mb"])
                f.write(f"The slowest stage was **{slowest['stage']}** taking {slowest['runtime_sec']:.2f} seconds. ")
                f.write(f"The most memory-intensive stage was **{hungriest['stage']}** peaking at {hungriest['peak_memory_mb']:.2f} MB.\n")
                
            f.write("\n## Interpretation\n")
            f.write("Performance bottlenecks typically arise in Feature Engineering (due to rolling/expanding windows) and Model Training (due to Optuna trials). ")
            f.write("Memory peaks usually occur during feature matrix materialization and cross-validation folding.\n")
            
            f.write("\n## Recommendations\n")
            f.write("1. If Feature Engineering is the bottleneck, consider vectorizing rolling operations or moving to Polars/DuckDB.\n")
            f.write("2. If Model Training is slow, reduce the number of Optuna trials or restrict `TimeSeriesSplit` to fewer folds.\n")
            f.write("3. If memory is exhausting, use `chunksize` for SQLite reads and garbage collect aggressively.\n")
            
        logger.info("Performance profile generated successfully.")

# Global instance for easy access
profiler = Profiler()
