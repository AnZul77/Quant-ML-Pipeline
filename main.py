"""
Main CLI entry point for the CrowdWisdomTrading Quantitative ML Pipeline.
"""

import datetime
import os
import random
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file before anything else
load_dotenv()

from src.utils.config import PipelineConfig
from src.utils.logger import get_logger
from src.evaluation.profiler import profiler
from src.evaluation.sanity_checks import SanityChecker

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _get_db(config: PipelineConfig):
    """Factory to get the appropriate database client."""
    if config.db_engine == "sqlite":
        from src.database.sqlite import SQLiteClient
        db = SQLiteClient(config.db_sqlite_path)
    elif config.db_engine == "postgres":
        from src.database.postgres import PostgresClient
        pg_conf = config.db_postgres
        db = PostgresClient(
            host=pg_conf["host"],
            port=pg_conf["port"],
            database=pg_conf["database"],
            user=pg_conf["user"],
            password=pg_conf["password"]
        )
    else:
        raise ValueError(f"Unknown database engine: {config.db_engine}")
    return db


@click.group()
@click.option("--config", type=click.Path(exists=True), default=None, help="Path to config.yaml")
@click.option("--seed", type=int, default=None, help="Random seed override")
@click.option("--model", type=str, default=None, help="Default model override")
@click.option("--train-window", type=int, default=None, help="Train window override")
@click.option("--test-window", type=int, default=None, help="Test window override")
@click.pass_context
def cli(ctx, config, seed, model, train_window, test_window):
    """CrowdWisdomTrading Quantitative ML Pipeline CLI."""
    overrides = {}
    if seed is not None:
        overrides["seed"] = seed
    if model is not None:
        overrides["model"] = model
    if train_window is not None:
        overrides["train_window"] = train_window
    if test_window is not None:
        overrides["test_window"] = test_window

    conf = PipelineConfig(config_path=config, overrides=overrides)
    set_seed(conf.seed)
    
    ctx.ensure_object(dict)
    ctx.obj["CONFIG"] = conf


@cli.command()
@click.pass_context
def scrape(ctx):
    """Download macroeconomic calendar data from Trading Economics."""
    logger.info("Starting scrape command")
    profiler.start_stage("scrape")
    config = ctx.obj["CONFIG"]
    db = _get_db(config)
    try:
        db.connect()
        db.initialize_schema()
        
        from src.data.ingestion import MacroIngestion
        ingestion = MacroIngestion(config, db)
        ingestion.scrape_macro_events()
    finally:
        db.close()
        profiler.end_stage()
    logger.info("Finished scrape command")


@cli.command()
@click.pass_context
def ingest(ctx):
    """Load raw CSV trading logs into database."""
    logger.info("Starting ingest command")
    profiler.start_stage("ingest")
    config = ctx.obj["CONFIG"]
    db = _get_db(config)
    try:
        db.connect()
        db.initialize_schema()
        

        if config.raw_logs_path.exists():
            df = pd.read_csv(config.raw_logs_path)
            db.execute("DELETE FROM trades")
            db.write_dataframe(df, "trades", if_exists="append")
            logger.info("Loaded %d rows from %s into trades table", len(df), config.raw_logs_path)
        else:
            logger.warning("Trades raw file not found at %s", config.raw_logs_path)
    finally:
        db.close()
        profiler.end_stage()
    logger.info("Finished ingest command")


@cli.command()
@click.pass_context
def clean(ctx):
    """Run deduplication and account filtering on trades."""
    logger.info("Starting clean command")
    profiler.start_stage("clean")
    config = ctx.obj["CONFIG"]
    db = _get_db(config)
    try:
        db.connect()
        
        from src.data.cleaning import TradeCleaner
        cleaner = TradeCleaner(config, db)
        cleaner.clean()
        cleaner.generate_quality_report()
    finally:
        db.close()
        profiler.end_stage()
    logger.info("Finished clean command")


@cli.command()
@click.pass_context
def features(ctx):
    """Compute features and write to feature_store."""
    logger.info("Starting features command")
    profiler.start_stage("features")
    config = ctx.obj["CONFIG"]
    db = _get_db(config)
    try:
        db.connect()
        
        from src.features.feature_engineer import FeatureEngineer
        fe = FeatureEngineer(config, db)
        fe.build_features()
        
        # Fail fast: check feature matrix
        df = db.read_sql("SELECT * FROM feature_store")
        feature_cols = [c for c in df.columns if c not in ["trade_id", "timestamp", "pnl", "target", "strategy", "account_id"] and pd.api.types.is_numeric_dtype(df[c])]
        SanityChecker.check_feature_matrix(df, feature_cols)
        
    finally:
        db.close()
        profiler.end_stage()
    logger.info("Finished features command")


@cli.command()
@click.argument("experiment_id", required=False)
@click.pass_context
def train(ctx, experiment_id):
    """Run walk-forward validation and tune models."""
    logger.info("Starting train command")
    profiler.start_stage("train")
    config = ctx.obj["CONFIG"]
    db = _get_db(config)
    try:
        db.connect()
        
        if experiment_id is None:
            experiment_id = f"exp_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')}"
        
        out_dir = config.get_experiment_dir(experiment_id)
        (out_dir / "models").mkdir(parents=True, exist_ok=True)
        (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (out_dir / "reports").mkdir(parents=True, exist_ok=True)
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        
        from src.models.trainer import WalkForwardTrainer
        trainer = WalkForwardTrainer(config, db)
        results = trainer.train(experiment_id)
        
        # Store results dict in context for evaluate if pipeline runs
        ctx.obj["TRAIN_RESULTS"] = results
        ctx.obj["EXPERIMENT_ID"] = experiment_id
    finally:
        db.close()
        profiler.end_stage()
    logger.info("Finished train command for experiment %s", experiment_id)


@cli.command()
@click.argument("experiment_id", required=True)
@click.pass_context
def evaluate(ctx, experiment_id):
    """Run performance metrics, equity curves, heatmaps, and error analysis."""
    logger.info("Starting evaluate command")
    profiler.start_stage("evaluate")
    config = ctx.obj["CONFIG"]
    db = _get_db(config)
    try:
        db.connect()
        
        out_dir = config.get_experiment_dir(experiment_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "reports").mkdir(exist_ok=True)
        (out_dir / "figures").mkdir(exist_ok=True)
        
        from src.evaluation.evaluator import PipelineEvaluator
        evaluator = PipelineEvaluator(config, db)
        
        results = ctx.obj.get("TRAIN_RESULTS", {})
        evaluator.run_evaluation(experiment_id, out_dir, results)
        
        # Run Phase 1 Diagnostics
        from src.database.auditor import DatabaseAuditor
        auditor = DatabaseAuditor(config, db)
        auditor.generate_audit(out_dir)
        
        from src.evaluation.feature_diagnostics import FeatureDiagnoser
        fd = FeatureDiagnoser(config, db)
        fd.generate_diagnostics(experiment_id, out_dir)
        
        from src.evaluation.target_diagnostics import TargetDiagnoser
        td = TargetDiagnoser(config, db)
        td.generate_diagnostics(experiment_id, out_dir)
        
        # Generate performance profile after evaluation
        profiler.generate_report(out_dir)
    finally:
        db.close()
        profiler.end_stage()
    logger.info("Finished evaluate command")


@cli.command()
@click.argument("experiment_id", required=True)
@click.pass_context
def predict(ctx, experiment_id):
    """Score candidate permutations."""
    logger.info("Starting predict command")
    config = ctx.obj["CONFIG"]
    db = _get_db(config)
    try:
        db.connect()
        
        from src.models.predictor import PermutationPredictor
        predictor = PermutationPredictor(config, db)
        
        df = predictor.predict_best_permutations(experiment_id)
        matrix = predictor.generate_recommendation_matrix(experiment_id)
        
        out_dir = config.get_experiment_dir(experiment_id) / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(out_dir / "best_permutations.csv", index=False)
        matrix.to_csv(out_dir / "recommendation_matrix.csv")
    finally:
        db.close()
    logger.info("Finished predict command")


@cli.command()
@click.pass_context
def pipeline(ctx):
    """Run the entire sequence end-to-end."""
    logger.info("Starting full pipeline")
    config = ctx.obj["CONFIG"]
    
    if not config.raw_logs_path.exists():
        raise FileNotFoundError(
            f"Raw trades file not found at {config.raw_logs_path}. "
            "Please provide trading logs or use the developer utility scripts/generate_synthetic.py manually."
        )
        
    # Download macroeconomic data from Trading Economics
    ctx.invoke(scrape)
    ctx.invoke(ingest)
    ctx.invoke(clean)
    ctx.invoke(features)
    
    experiment_id = f"exp_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')}"
    ctx.invoke(train, experiment_id=experiment_id)
    ctx.invoke(evaluate, experiment_id=experiment_id)
    ctx.invoke(predict, experiment_id=experiment_id)
    
    logger.info("Finished full pipeline")


if __name__ == "__main__":
    cli()
