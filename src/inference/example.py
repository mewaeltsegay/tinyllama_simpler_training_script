"""Example usage of the bilingual inference engine."""

import logging
from typing import List
from .engine import create_inference_engine
from .quality import create_quality_validator, create_parameter_optimizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_generation():
    """Example of basic text generation."""
    print("=== Basic Text Generation Example ===")
    
    # Create inference engine
    engine = create_inference_engine(device='auto')
    
    # Load model and tokenizer (paths would need to be updated for actual use)
    try:
        # engine.load_trained_model(
        #     checkpoint_path="path/to/trained/model",
        #     tokenizer_path="tokenizer/"
        # )
        print("Note: Update checkpoint_path and tokenizer_path for actual usage")
        return
    except Exception as e:
        print(f"Model loading failed (expected in example): {e}")
        return
    
    # Configure generation parameters
    engine.configure_generation(
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        max_length=100,
        repetition_penalty=1.1
    )
    
    # Test prompts in different languages
    test_prompts = [
        "Hello, how are you today?",
        "ሰላም! ከመይ ኣሎኻ?",
        "Tell me a story about",
        "ብዛዕባ ትግራይ"
    ]
    
    # Generate text for each prompt
    for prompt in test_prompts:
        try:
            generated = engine.generate_text(prompt)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}")
            print("-" * 50)
        except Exception as e:
            print(f"Generation failed for '{prompt}': {e}")


def example_batch_generation():
    """Example of batch text generation."""
    print("=== Batch Generation Example ===")
    
    engine = create_inference_engine()
    
    # Batch prompts
    prompts = [
        "The weather today is",
        "ሎሚ ኣየር ሃዋ",
        "Machine learning is",
        "ኮምፒተር ትምህርቲ"
    ]
    
    try:
        # Generate batch
        results = engine.batch_generate(
            prompts,
            temperature=0.7,
            max_length=50
        )
        
        for prompt, result in zip(prompts, results):
            print(f"Prompt: {prompt}")
            print(f"Generated: {result}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Batch generation failed (expected without loaded model): {e}")


def example_quality_validation():
    """Example of quality validation."""
    print("=== Quality Validation Example ===")
    
    # Create quality validator
    validator = create_quality_validator()
    
    # Example generated texts (simulated)
    test_cases = [
        {
            'prompt': "Tell me about artificial intelligence",
            'generated': "Artificial intelligence is a fascinating field that involves creating machines capable of performing tasks that typically require human intelligence. These systems can learn, reason, and make decisions."
        },
        {
            'prompt': "ብዛዕባ ኣርቲፊሻል ኢንተለጀንስ ንገረኒ",
            'generated': "ኣርቲፊሻል ኢንተለጀንስ ሓደ ዘደንቕ ዓውዲ እዩ። እዚ ናይ ሰብ ኣእምሮ ዝሓትት ዕማማት ክፍጽማ ዝኽእላ ማሽናት ምፍጣር እዩ።"
        },
        {
            'prompt': "Repeat this word: hello hello hello",
            'generated': "hello hello hello hello hello hello hello hello hello"  # High repetition
        }
    ]
    
    # Validate each case
    for i, case in enumerate(test_cases):
        print(f"Test Case {i+1}:")
        print(f"Prompt: {case['prompt']}")
        print(f"Generated: {case['generated']}")
        
        # Validate quality
        result = validator.validate_generation_quality(
            case['prompt'], 
            case['generated']
        )
        
        print(f"Quality Score: {result['quality_score']:.2f}")
        print(f"Overall Quality: {result['overall_quality']}")
        print(f"Language Analysis: {result['language_analysis']['prompt_language']} -> {result['language_analysis']['output_language']}")
        
        if result['issues']:
            print(f"Issues: {', '.join(result['issues'])}")
        
        if result['recommendations']:
            print(f"Recommendations: {', '.join(result['recommendations'][:2])}")
        
        print("-" * 50)


def example_parameter_optimization():
    """Example of parameter optimization."""
    print("=== Parameter Optimization Example ===")
    
    # This would require a loaded model, so we'll just show the structure
    print("Parameter optimization requires a loaded model.")
    print("Example usage:")
    print("""
    engine = create_inference_engine()
    engine.load_trained_model(checkpoint_path, tokenizer_path)
    
    optimizer = create_parameter_optimizer()
    
    test_prompts = [
        "Hello, how are you?",
        "ሰላም! ከመይ ኣሎኻ?",
        "Tell me about the weather",
        "ብዛዕባ ኣየር ሃዋ ንገረኒ"
    ]
    
    results = optimizer.optimize_parameters(
        engine, 
        test_prompts, 
        target_quality=0.7
    )
    
    print(f"Best parameters: {results['best_parameters']}")
    print(f"Best quality score: {results['best_quality_score']}")
    """)


def main():
    """Run all examples."""
    print("Bilingual Inference Engine Examples")
    print("=" * 50)
    
    example_basic_generation()
    print("\n")
    
    example_batch_generation()
    print("\n")
    
    example_quality_validation()
    print("\n")
    
    example_parameter_optimization()


if __name__ == "__main__":
    main()