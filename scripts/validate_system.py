#!/usr/bin/env python3
"""
System validation script for Tigrinya TinyLlama training pipeline.

This script performs comprehensive validation of the entire system including:
- Configuration validation
- Hardware compatibility
- Model and tokenizer functionality
- Data pipeline validation
- Training engine validation
- Inference engine validation
- System integration validation
- Performance validation
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from validation.system_integration import run_system_validation, SystemIntegrationValidator
from config.manager import ConfigManager
from utils.logging import setup_logging, get_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate Tigrinya TinyLlama training system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to JSON configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_output",
        help="Directory to save validation results"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (skip time-consuming tests)"
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        choices=[
            "config", "hardware", "model", "data", 
            "training", "inference", "integration", "performance"
        ],
        help="Specific validation categories to run (default: all)"
    )
    
    parser.add_argument(
        "--fix-issues",
        action="store_true",
        help="Attempt to automatically fix detected issues"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        default=True,
        help="Generate detailed validation report"
    )
    
    return parser.parse_args()


def setup_validation_environment(args):
    """Setup validation environment."""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "validation.log"
    setup_logging(
        level=args.log_level,
        log_file=str(log_file),
        console_output=True
    )
    
    return output_dir


def run_targeted_validation(config_path: str, categories: list, output_dir: Path) -> dict:
    """Run validation for specific categories only."""
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        # Create validator
        validator = SystemIntegrationValidator(config)
        
        # Run specific validations
        if "config" in categories:
            validator._validate_configuration()
        
        if "hardware" in categories:
            validator._validate_hardware()
        
        if "model" in categories:
            validator._validate_model_and_tokenizer()
        
        if "data" in categories:
            validator._validate_data_pipeline()
        
        if "training" in categories:
            validator._validate_training_engine()
        
        if "inference" in categories:
            validator._validate_inference_engine()
        
        if "integration" in categories:
            validator._validate_system_integration()
        
        if "performance" in categories:
            validator._validate_performance()
        
        # Determine overall status
        validator._determine_overall_status()
        
        return validator.validation_results
        
    except Exception as e:
        logger.error(f"Targeted validation failed: {e}")
        return {"overall_status": "failed", "error": str(e)}


def attempt_issue_fixes(validation_results: dict, config_path: str) -> dict:
    """Attempt to automatically fix detected issues."""
    logger = get_logger(__name__)
    
    fixes_applied = []
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        # Fix hardware configuration issues
        hardware_validation = validation_results.get("hardware_validation", {})
        if not hardware_validation.get("memory_estimate_fits", True):
            logger.info("Attempting to fix memory issues...")
            
            # Reduce batch size
            original_batch_size = config.training_params.batch_size
            config.training_params.batch_size = max(1, original_batch_size // 2)
            
            # Increase gradient accumulation
            config.training_params.gradient_accumulation_steps *= 2
            
            # Enable gradient checkpointing
            config.training_params.gradient_checkpointing = True
            
            fixes_applied.append("reduced_batch_size_and_enabled_checkpointing")
        
        # Fix training configuration issues
        training_validation = validation_results.get("training_validation", {})
        if not training_validation.get("mixed_precision_setup", True):
            logger.info("Attempting to fix mixed precision issues...")
            
            # Use FP16 for better compatibility
            config.training_params.mixed_precision = "fp16"
            fixes_applied.append("switched_to_fp16")
        
        # Save fixed configuration
        if fixes_applied:
            fixed_config_path = config_path.replace(".json", "_fixed.json")
            config_manager.save_config(config, fixed_config_path)
            logger.info(f"Fixed configuration saved to: {fixed_config_path}")
        
        return {
            "fixes_applied": fixes_applied,
            "fixed_config_path": fixed_config_path if fixes_applied else None
        }
        
    except Exception as e:
        logger.error(f"Issue fixing failed: {e}")
        return {"fixes_applied": [], "error": str(e)}


def generate_validation_report(validation_results: dict, output_dir: Path, config_path: str):
    """Generate comprehensive validation report."""
    logger = get_logger(__name__)
    
    try:
        # Save detailed JSON report
        json_report_path = output_dir / "detailed_validation_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # Generate HTML report
        html_report_path = output_dir / "validation_report.html"
        generate_html_report(validation_results, html_report_path, config_path)
        
        # Generate text summary
        text_summary_path = output_dir / "validation_summary.txt"
        generate_text_summary(validation_results, text_summary_path)
        
        logger.info(f"Validation reports generated in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")


def generate_html_report(validation_results: dict, output_path: Path, config_path: str):
    """Generate HTML validation report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tigrinya TinyLlama System Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .status-passed {{ color: green; font-weight: bold; }}
            .status-failed {{ color: red; font-weight: bold; }}
            .status-warning {{ color: orange; font-weight: bold; }}
            .category {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .check {{ margin: 5px 0; padding: 5px; }}
            .check-pass {{ background-color: #e8f5e8; }}
            .check-fail {{ background-color: #ffe8e8; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Tigrinya TinyLlama System Validation Report</h1>
            <p><strong>Configuration:</strong> {config_path}</p>
            <p><strong>Validation Time:</strong> {validation_results.get('timestamp', 'Unknown')}</p>
            <p><strong>Overall Status:</strong> 
                <span class="status-{validation_results['overall_status'].replace('_', '-')}">
                    {validation_results['overall_status'].upper()}
                </span>
            </p>
        </div>
    """
    
    # Add validation categories
    for category, results in validation_results.items():
        if category in ["timestamp", "overall_status"]:
            continue
        
        html_content += f"""
        <div class="category">
            <h2>{category.replace('_', ' ').title()}</h2>
        """
        
        if isinstance(results, dict) and "error" not in results:
            html_content += "<table><tr><th>Check</th><th>Status</th></tr>"
            for check_name, status in results.items():
                status_class = "check-pass" if status else "check-fail"
                status_text = "PASS" if status else "FAIL"
                html_content += f"""
                <tr class="{status_class}">
                    <td>{check_name.replace('_', ' ').title()}</td>
                    <td>{status_text}</td>
                </tr>
                """
            html_content += "</table>"
        elif "error" in results:
            html_content += f'<p class="status-failed">ERROR: {results["error"]}</p>'
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def generate_text_summary(validation_results: dict, output_path: Path):
    """Generate text summary of validation results."""
    with open(output_path, 'w') as f:
        f.write("TIGRINYA TINYLLAMA SYSTEM VALIDATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Overall Status: {validation_results['overall_status'].upper()}\n\n")
        
        # Count passed/failed checks
        total_checks = 0
        passed_checks = 0
        
        for category, results in validation_results.items():
            if category in ["timestamp", "overall_status"]:
                continue
            
            if isinstance(results, dict) and "error" not in results:
                for check_name, status in results.items():
                    total_checks += 1
                    if status:
                        passed_checks += 1
        
        f.write(f"Summary: {passed_checks}/{total_checks} checks passed\n")
        f.write(f"Success Rate: {(passed_checks/total_checks)*100:.1f}%\n\n")
        
        # Write category details
        for category, results in validation_results.items():
            if category in ["timestamp", "overall_status"]:
                continue
            
            f.write(f"{category.replace('_', ' ').title()}:\n")
            f.write("-" * 30 + "\n")
            
            if isinstance(results, dict) and "error" not in results:
                category_passed = sum(1 for status in results.values() if status)
                category_total = len(results)
                f.write(f"  Status: {category_passed}/{category_total} checks passed\n")
                
                for check_name, status in results.items():
                    status_str = "✓ PASS" if status else "✗ FAIL"
                    f.write(f"  {status_str}: {check_name.replace('_', ' ')}\n")
            elif "error" in results:
                f.write(f"  ✗ ERROR: {results['error']}\n")
            
            f.write("\n")


def main():
    """Main validation function."""
    args = parse_arguments()
    
    # Setup validation environment
    output_dir = setup_validation_environment(args)
    logger = get_logger(__name__)
    
    logger.info("Starting Tigrinya TinyLlama system validation...")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Validate configuration file exists
        if not os.path.exists(args.config):
            logger.error(f"Configuration file not found: {args.config}")
            return 1
        
        # Run validation
        if args.categories:
            logger.info(f"Running targeted validation for categories: {args.categories}")
            validation_results = run_targeted_validation(args.config, args.categories, output_dir)
        else:
            logger.info("Running complete system validation...")
            validation_results = run_system_validation(args.config, str(output_dir))
        
        # Attempt to fix issues if requested
        if args.fix_issues and validation_results["overall_status"] != "passed":
            logger.info("Attempting to fix detected issues...")
            fix_results = attempt_issue_fixes(validation_results, args.config)
            validation_results["fix_results"] = fix_results
            
            if fix_results["fixes_applied"]:
                logger.info(f"Applied fixes: {fix_results['fixes_applied']}")
                
                # Re-run validation with fixed configuration
                if fix_results.get("fixed_config_path"):
                    logger.info("Re-running validation with fixed configuration...")
                    fixed_validation_results = run_system_validation(
                        fix_results["fixed_config_path"], 
                        str(output_dir / "fixed_validation")
                    )
                    validation_results["fixed_validation_results"] = fixed_validation_results
        
        # Generate reports
        if args.generate_report:
            logger.info("Generating validation reports...")
            generate_validation_report(validation_results, output_dir, args.config)
        
        # Print summary
        overall_status = validation_results["overall_status"]
        logger.info(f"Validation completed with status: {overall_status.upper()}")
        
        if overall_status == "passed":
            logger.info("✓ System is ready for training!")
            return 0
        elif overall_status == "passed_with_warnings":
            logger.warning("⚠ System can be used but has some issues")
            logger.info("Check validation report for details")
            return 0
        else:
            logger.error("✗ System has critical issues that must be addressed")
            logger.info("Check validation report for details")
            return 1
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())