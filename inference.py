#!/usr/bin/env python3
"""
Inference script for Tigrinya TinyLlama bilingual text generation.

This script provides both interactive and batch text generation modes with
configurable generation parameters and quality validation.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.inference.engine import BilingualInferenceEngine
from src.inference.quality import BilingualQualityValidator, GenerationParameterOptimizer
from src.utils.logging import setup_logging, get_logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text using trained Tigrinya TinyLlama model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and tokenizer arguments
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--tokenizer-path", "-t",
        type=str,
        required=True,
        help="Path to tokenizer directory"
    )
    
    # Generation mode arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "batch", "single"],
        help="Generation mode"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Single prompt for text generation (single mode)"
    )
    
    parser.add_argument(
        "--input-file", "-i",
        type=str,
        help="Input file with prompts (batch mode)"
    )
    
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        help="Output file for generated text (batch mode)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0.1 to 2.0)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum generation length"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Minimum generation length"
    )
    
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty factor"
    )
    
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help="Number of sequences to generate per prompt"
    )
    
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=True,
        help="Use sampling for generation"
    )
    
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    
    # Configuration file support
    parser.add_argument(
        "--config",
        type=str,
        help="JSON configuration file for generation parameters"
    )
    
    # Quality validation
    parser.add_argument(
        "--validate-quality",
        action="store_true",
        help="Validate generation quality"
    )
    
    parser.add_argument(
        "--optimize-parameters",
        action="store_true",
        help="Optimize generation parameters for quality"
    )
    
    parser.add_argument(
        "--test-prompts",
        type=str,
        help="File with test prompts for parameter optimization"
    )
    
    # Hardware and logging
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output except for generated text"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with generation statistics"
    )
    
    return parser.parse_args()


def load_generation_config(config_path: str) -> Dict[str, Any]:
    """Load generation configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate configuration
        valid_params = {
            'temperature', 'top_k', 'top_p', 'max_length', 'min_length',
            'repetition_penalty', 'length_penalty', 'do_sample', 
            'num_return_sequences', 'early_stopping'
        }
        
        generation_config = {}
        for key, value in config.items():
            if key in valid_params:
                generation_config[key] = value
            else:
                logger.warning(f"Unknown generation parameter: {key}")
        
        return generation_config
        
    except Exception as e:
        logger.error(f"Failed to load generation config: {e}")
        raise


def setup_inference_engine(args: argparse.Namespace) -> BilingualInferenceEngine:
    """Setup and configure the inference engine."""
    logger = get_logger(__name__)
    
    try:
        # Initialize inference engine
        logger.info("Initializing bilingual inference engine...")
        engine = BilingualInferenceEngine(device=args.device)
        
        # Load model and tokenizer
        logger.info(f"Loading model from: {args.model_path}")
        logger.info(f"Loading tokenizer from: {args.tokenizer_path}")
        engine.load_trained_model(args.model_path, args.tokenizer_path)
        
        # Configure generation parameters
        generation_params = {}
        
        # Load from config file if provided
        if args.config:
            logger.info(f"Loading generation config from: {args.config}")
            generation_params.update(load_generation_config(args.config))
        
        # Override with command line arguments
        generation_params.update({
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'max_length': args.max_length,
            'min_length': args.min_length,
            'repetition_penalty': args.repetition_penalty,
            'num_return_sequences': args.num_return_sequences,
            'do_sample': not args.no_sample if args.no_sample else args.do_sample
        })
        
        # Configure the engine
        engine.configure_generation(**generation_params)
        
        if not args.quiet:
            logger.info("Inference engine setup completed")
            if args.verbose:
                model_info = engine.get_model_info()
                logger.info(f"Model info: {json.dumps(model_info, indent=2, default=str)}")
        
        return engine
        
    except Exception as e:
        logger.error(f"Failed to setup inference engine: {e}")
        raise


def run_single_generation(engine: BilingualInferenceEngine, 
                         prompt: str, 
                         args: argparse.Namespace) -> None:
    """Run single text generation."""
    logger = get_logger(__name__)
    
    try:
        if not args.quiet:
            print(f"\nPrompt: {prompt}")
            print("-" * 50)
        
        # Generate text
        generated_text = engine.generate_text(prompt)
        
        # Print generated text
        print(generated_text)
        
        # Show statistics if verbose
        if args.verbose:
            stats = engine.get_generation_stats(prompt, generated_text)
            print(f"\nGeneration Statistics:")
            print(f"  Input language: {stats['input_language']}")
            print(f"  Output language: {stats['output_language']}")
            print(f"  Prompt tokens: {stats['prompt_tokens']}")
            print(f"  Generated tokens: {stats['generated_tokens']}")
            print(f"  Total tokens: {stats['total_tokens']}")
            print(f"  Language consistency: {stats['language_consistency']}")
        
        # Validate quality if requested
        if args.validate_quality:
            validator = BilingualQualityValidator()
            quality_result = validator.validate_generation_quality(prompt, generated_text)
            
            print(f"\nQuality Assessment:")
            print(f"  Overall quality: {quality_result['overall_quality']}")
            print(f"  Quality score: {quality_result['quality_score']:.3f}")
            
            if quality_result['issues']:
                print(f"  Issues: {', '.join(quality_result['issues'])}")
            
            if quality_result['recommendations']:
                print(f"  Recommendations: {', '.join(quality_result['recommendations'])}")
        
    except Exception as e:
        logger.error(f"Single generation failed: {e}")
        raise


def run_interactive_mode(engine: BilingualInferenceEngine, 
                        args: argparse.Namespace) -> None:
    """Run interactive text generation mode."""
    logger = get_logger(__name__)
    
    print("=" * 60)
    print("Tigrinya TinyLlama Interactive Text Generation")
    print("=" * 60)
    print("Enter prompts to generate text. Type 'quit' or 'exit' to stop.")
    print("Type 'help' for available commands.")
    print("Type 'config' to see current generation parameters.")
    print("-" * 60)
    
    validator = BilingualQualityValidator() if args.validate_quality else None
    
    try:
        while True:
            try:
                # Get user input
                prompt = input("\nPrompt: ").strip()
                
                if not prompt:
                    continue
                
                # Handle special commands
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif prompt.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  help - Show this help message")
                    print("  config - Show current generation parameters")
                    print("  set <param> <value> - Set generation parameter")
                    print("  quit/exit/q - Exit interactive mode")
                    print("\nGeneration parameters you can set:")
                    print("  temperature, top_k, top_p, max_length, repetition_penalty")
                    continue
                
                elif prompt.lower() == 'config':
                    config = engine.generation_config.to_dict()
                    print("\nCurrent generation parameters:")
                    for key, value in config.items():
                        if key in ['temperature', 'top_k', 'top_p', 'max_length', 'repetition_penalty']:
                            print(f"  {key}: {value}")
                    continue
                
                elif prompt.lower().startswith('set '):
                    parts = prompt.split()
                    if len(parts) >= 3:
                        param_name = parts[1]
                        try:
                            param_value = float(parts[2]) if '.' in parts[2] else int(parts[2])
                            
                            # Update parameter
                            current_config = engine.generation_config.to_dict()
                            if param_name in current_config:
                                current_config[param_name] = param_value
                                engine.configure_generation(**{param_name: param_value})
                                print(f"Set {param_name} = {param_value}")
                            else:
                                print(f"Unknown parameter: {param_name}")
                        except ValueError:
                            print("Invalid parameter value")
                    else:
                        print("Usage: set <parameter> <value>")
                    continue
                
                # Generate text
                print("\nGenerating...")
                generated_text = engine.generate_text(prompt)
                
                print(f"\nGenerated text:")
                print("-" * 30)
                print(generated_text)
                print("-" * 30)
                
                # Show statistics if verbose
                if args.verbose:
                    stats = engine.get_generation_stats(prompt, generated_text)
                    print(f"\nStatistics: {stats['input_language']} → {stats['output_language']}, "
                          f"{stats['generated_tokens']} tokens")
                
                # Validate quality if requested
                if validator:
                    quality_result = validator.validate_generation_quality(prompt, generated_text)
                    print(f"Quality: {quality_result['overall_quality']} "
                          f"({quality_result['quality_score']:.2f})")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
                continue
            except Exception as e:
                logger.error(f"Generation error: {e}")
                print(f"Error: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Interactive mode failed: {e}")
        raise


def run_batch_mode(engine: BilingualInferenceEngine, 
                  args: argparse.Namespace) -> None:
    """Run batch text generation mode."""
    logger = get_logger(__name__)
    
    try:
        # Load prompts from input file
        if not args.input_file:
            raise ValueError("Input file is required for batch mode")
        
        logger.info(f"Loading prompts from: {args.input_file}")
        
        prompts = []
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    prompts.append(line)
        
        if not prompts:
            raise ValueError("No valid prompts found in input file")
        
        logger.info(f"Loaded {len(prompts)} prompts")
        
        # Generate text for all prompts
        logger.info("Starting batch generation...")
        
        results = []
        for i, prompt in enumerate(prompts, 1):
            if not args.quiet:
                print(f"Processing prompt {i}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                generated_text = engine.generate_text(prompt)
                
                result = {
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'success': True,
                    'error': None
                }
                
                # Add statistics if verbose
                if args.verbose:
                    stats = engine.get_generation_stats(prompt, generated_text)
                    result['statistics'] = stats
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Generation failed for prompt {i}: {e}")
                results.append({
                    'prompt': prompt,
                    'generated_text': '',
                    'success': False,
                    'error': str(e)
                })
        
        # Validate quality if requested
        if args.validate_quality:
            logger.info("Validating generation quality...")
            validator = BilingualQualityValidator()
            
            successful_results = [r for r in results if r['success']]
            if successful_results:
                prompts_for_validation = [r['prompt'] for r in successful_results]
                texts_for_validation = [r['generated_text'] for r in successful_results]
                
                quality_results = validator.batch_validate_quality(
                    prompts_for_validation, texts_for_validation
                )
                
                # Add quality results
                for result, quality in zip(successful_results, quality_results):
                    result['quality'] = quality
                
                # Get quality summary
                quality_summary = validator.get_quality_summary(quality_results)
                logger.info(f"Quality summary: {quality_summary}")
        
        # Save results
        if args.output_file:
            logger.info(f"Saving results to: {args.output_file}")
            
            # Determine output format based on file extension
            output_path = Path(args.output_file)
            
            if output_path.suffix.lower() == '.json':
                # Save as JSON
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                # Save as text
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for i, result in enumerate(results, 1):
                        f.write(f"=== Prompt {i} ===\n")
                        f.write(f"{result['prompt']}\n\n")
                        f.write(f"=== Generated Text ===\n")
                        if result['success']:
                            f.write(f"{result['generated_text']}\n")
                        else:
                            f.write(f"ERROR: {result['error']}\n")
                        f.write("\n" + "="*50 + "\n\n")
        else:
            # Print results to console
            for i, result in enumerate(results, 1):
                print(f"\n=== Prompt {i} ===")
                print(result['prompt'])
                print("\n=== Generated Text ===")
                if result['success']:
                    print(result['generated_text'])
                    
                    if args.verbose and 'statistics' in result:
                        stats = result['statistics']
                        print(f"\nStats: {stats['input_language']} → {stats['output_language']}, "
                              f"{stats['generated_tokens']} tokens")
                    
                    if args.validate_quality and 'quality' in result:
                        quality = result['quality']
                        print(f"Quality: {quality['overall_quality']} ({quality['quality_score']:.2f})")
                else:
                    print(f"ERROR: {result['error']}")
                
                print("=" * 50)
        
        # Print summary
        successful_count = sum(1 for r in results if r['success'])
        logger.info(f"Batch generation completed: {successful_count}/{len(results)} successful")
        
    except Exception as e:
        logger.error(f"Batch mode failed: {e}")
        raise


def optimize_generation_parameters(engine: BilingualInferenceEngine,
                                 args: argparse.Namespace) -> None:
    """Optimize generation parameters for better quality."""
    logger = get_logger(__name__)
    
    try:
        if not args.test_prompts:
            raise ValueError("Test prompts file is required for parameter optimization")
        
        # Load test prompts
        logger.info(f"Loading test prompts from: {args.test_prompts}")
        test_prompts = []
        with open(args.test_prompts, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    test_prompts.append(line)
        
        if not test_prompts:
            raise ValueError("No valid test prompts found")
        
        logger.info(f"Loaded {len(test_prompts)} test prompts")
        
        # Initialize optimizer
        optimizer = GenerationParameterOptimizer()
        
        # Run optimization
        logger.info("Starting parameter optimization...")
        optimization_result = optimizer.optimize_parameters(
            engine, test_prompts, target_quality=0.7, max_iterations=3
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("Parameter Optimization Results")
        print("=" * 60)
        
        if optimization_result['best_parameters']:
            print(f"Best parameters found:")
            for param, value in optimization_result['best_parameters'].items():
                print(f"  {param}: {value}")
            
            print(f"\nBest quality score: {optimization_result['best_quality_score']:.3f}")
            print(f"Target reached: {optimization_result['target_reached']}")
        else:
            print("No optimal parameters found")
        
        # Save optimization history if output file specified
        if args.output_file:
            output_path = Path(args.output_file)
            optimization_file = output_path.parent / f"{output_path.stem}_optimization.json"
            
            with open(optimization_file, 'w', encoding='utf-8') as f:
                json.dump(optimization_result, f, indent=2)
            
            logger.info(f"Optimization results saved to: {optimization_file}")
        
    except Exception as e:
        logger.error(f"Parameter optimization failed: {e}")
        raise


def main():
    """Main inference function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        if args.quiet:
            log_level = "ERROR"
        elif args.verbose:
            log_level = "DEBUG"
        else:
            log_level = args.log_level
        
        setup_logging(level=log_level, console_output=not args.quiet)
        logger = get_logger(__name__)
        
        if not args.quiet:
            logger.info("Starting Tigrinya TinyLlama Inference")
        
        # Setup inference engine
        engine = setup_inference_engine(args)
        
        # Run parameter optimization if requested
        if args.optimize_parameters:
            optimize_generation_parameters(engine, args)
            return 0
        
        # Run generation based on mode
        if args.mode == "single":
            if not args.prompt:
                raise ValueError("Prompt is required for single mode")
            run_single_generation(engine, args.prompt, args)
            
        elif args.mode == "interactive":
            run_interactive_mode(engine, args)
            
        elif args.mode == "batch":
            run_batch_mode(engine, args)
            
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        if not args.quiet:
            logger.info("Inference completed successfully")
        
        return 0
        
    except KeyboardInterrupt:
        if not args.quiet:
            print("\nInterrupted by user")
        return 1
        
    except Exception as e:
        if 'logger' in locals() and not args.quiet:
            logger.error(f"Inference failed: {e}")
        else:
            print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())