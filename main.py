import sys
import time
from source import train_linear_regression_model
from source import train_DT_model
from source import train_XGBoost_model


def display_results_table(results: dict) -> None:
    """Renders a beautifully aligned ASCII validation grid summary map."""
    print("\n" + "=" * 108)
    print("                             FINAL MODEL PERFORMANCE SUMMARY                              ")
    print("=" * 108)
    
    header = (
        f"{'Model Name':<20} | {'Train R²':<10} | {'Test R²':<10} | "
        f"{'Overfit Drop':<12} | {'Test MAE':<12} | {'Test MSE':<16} | {'Test RMSE':<12}"
    )
    print(header)
    print("-" * 108)
 
    for model_name, metrics in results.items():
        row = (
            f"{model_name:<20} | {metrics['Train_R2']:<10} | {metrics['Test_R2']:<10} | "
            f"{metrics['Overfit']:<12} | {metrics['MAE']:<12} | {metrics['MSE']:<16} | {metrics['RMSE']:<12}"
        )
        print(row) 
        
    print("=" * 108 + "\n")


def run_pipeline() -> None:
    """Master workflow orchestrator managing cross-model training execution."""
    print("=" * 60)
    print("        STARTING MASTER MACHINE LEARNING PIPELINE        ")
    print("=" * 60)
    
    pipeline_start_time = time.time()
    final_results = {}
    
    try:
        print("🚀 [1/3] Training Linear Regression...")
        final_results["Linear Regression"] = train_linear_regression_model.train_and_evaluate()
        
        print("\n🚀 [2/3] Tuning Decision Tree (GridSearchCV)...")
        final_results["Decision Tree"] = train_DT_model.train_and_evaluate()
        
        print("\n🚀 [3/3] Optimizing XGBoost (GridSearchCV)...")
        final_results["XGBoost"] = train_XGBoost_model.train_and_evaluate()
        
    except Exception as e:
        print(f"\n❌ PIPELINE RUNTIME FAILURE: {str(e)}")
        sys.exit(1)

    total_duration = time.time() - pipeline_start_time
    display_results_table(final_results)
    print(f"Total Pipeline Execution Time: {total_duration:.2f} seconds\n")


if __name__ == "__main__":
    run_pipeline()
