"""Quality validation and assessment for bilingual text generation."""

import re
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter
import torch
import torch.nn.functional as F
from ..utils.logging import get_logger
from .engine import LanguageDetector

logger = get_logger(__name__)


class TextQualityMetrics:
    """Metrics for assessing text generation quality."""
    
    @staticmethod
    def calculate_perplexity(model, tokenizer, text: str, device: str = 'cuda') -> float:
        """Calculate perplexity of generated text.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            text: Text to evaluate
            device: Device for computation
            
        Returns:
            Perplexity score
        """
        try:
            # Tokenize text
            token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(token_ids) < 2:
                return float('inf')
            
            input_tensor = torch.tensor([token_ids]).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor, labels=input_tensor)
                loss = outputs.loss.item()
            
            return math.exp(loss)
            
        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {e}")
            return float('inf')
    
    @staticmethod
    def calculate_repetition_score(text: str, n: int = 4) -> float:
        """Calculate repetition score based on n-gram repetition.
        
        Args:
            text: Text to analyze
            n: N-gram size
            
        Returns:
            Repetition score (0.0 = no repetition, 1.0 = high repetition)
        """
        if not text or len(text.split()) < n:
            return 0.0
        
        words = text.split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        if not ngrams:
            return 0.0
        
        ngram_counts = Counter(ngrams)
        repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)
        
        return repeated_ngrams / len(ngrams)
    
    @staticmethod
    def calculate_diversity_score(text: str) -> Dict[str, float]:
        """Calculate lexical diversity metrics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with diversity metrics
        """
        if not text:
            return {'ttr': 0.0, 'unique_words': 0, 'total_words': 0}
        
        words = text.lower().split()
        unique_words = set(words)
        
        # Type-Token Ratio
        ttr = len(unique_words) / len(words) if words else 0.0
        
        return {
            'ttr': ttr,
            'unique_words': len(unique_words),
            'total_words': len(words)
        }
    
    @staticmethod
    def calculate_coherence_score(text: str) -> float:
        """Calculate basic coherence score based on sentence structure.
        
        Args:
            text: Text to analyze
            
        Returns:
            Coherence score (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        # Split into sentences
        sentences = re.split(r'[.!?።]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        coherence_factors = []
        
        # Check sentence length variation
        lengths = [len(s.split()) for s in sentences]
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            length_score = 1.0 / (1.0 + length_variance / 100)  # Normalize
            coherence_factors.append(length_score)
        
        # Check for complete sentences (basic heuristic)
        complete_sentences = 0
        for sentence in sentences:
            words = sentence.split()
            if len(words) >= 3:  # Minimum reasonable sentence length
                complete_sentences += 1
        
        completeness_score = complete_sentences / len(sentences) if sentences else 0.0
        coherence_factors.append(completeness_score)
        
        return sum(coherence_factors) / len(coherence_factors) if coherence_factors else 0.0


class BilingualQualityValidator:
    """Quality validation for bilingual text generation."""
    
    def __init__(self):
        """Initialize bilingual quality validator."""
        self.language_detector = LanguageDetector()
        self.quality_metrics = TextQualityMetrics()
        
        # Quality thresholds
        self.thresholds = {
            'min_length': 5,
            'max_repetition': 0.3,
            'min_diversity': 0.3,
            'min_coherence': 0.4,
            'max_perplexity': 100.0
        }
        
        logger.info("BilingualQualityValidator initialized")
    
    def validate_generation_quality(
        self, 
        prompt: str, 
        generated_text: str,
        model=None,
        tokenizer=None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """Validate quality of generated text.
        
        Args:
            prompt: Input prompt
            generated_text: Generated text
            model: Optional model for perplexity calculation
            tokenizer: Optional tokenizer for perplexity calculation
            device: Device for computation
            
        Returns:
            Quality validation results
        """
        results = {
            'overall_quality': 'unknown',
            'quality_score': 0.0,
            'issues': [],
            'metrics': {},
            'language_analysis': {},
            'recommendations': []
        }
        
        try:
            # Language analysis
            results['language_analysis'] = self._analyze_language_quality(prompt, generated_text)
            
            # Basic quality metrics
            results['metrics'] = self._calculate_quality_metrics(
                generated_text, model, tokenizer, device
            )
            
            # Quality assessment
            quality_score, issues = self._assess_quality(results['metrics'])
            results['quality_score'] = quality_score
            results['issues'] = issues
            
            # Overall quality rating
            results['overall_quality'] = self._determine_overall_quality(quality_score)
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            logger.debug(f"Quality validation completed: {results['overall_quality']}")
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            results['issues'].append(f"Validation error: {str(e)}")
        
        return results
    
    def _analyze_language_quality(self, prompt: str, generated_text: str) -> Dict[str, Any]:
        """Analyze language-specific quality aspects.
        
        Args:
            prompt: Input prompt
            generated_text: Generated text
            
        Returns:
            Language analysis results
        """
        prompt_language = self.language_detector.detect_language(prompt)
        output_language = self.language_detector.detect_language(generated_text)
        
        analysis = {
            'prompt_language': prompt_language,
            'output_language': output_language,
            'language_consistency': prompt_language == output_language,
            'language_switch_appropriate': False,
            'tigrinya_quality': {},
            'english_quality': {}
        }
        
        # Analyze Tigrinya-specific quality
        if 'tigrinya' in [prompt_language, output_language]:
            analysis['tigrinya_quality'] = self._analyze_tigrinya_quality(generated_text)
        
        # Analyze English-specific quality
        if 'english' in [prompt_language, output_language]:
            analysis['english_quality'] = self._analyze_english_quality(generated_text)
        
        # Check if language switch is appropriate
        if prompt_language != output_language:
            analysis['language_switch_appropriate'] = self._is_language_switch_appropriate(
                prompt, generated_text, prompt_language, output_language
            )
        
        return analysis
    
    def _analyze_tigrinya_quality(self, text: str) -> Dict[str, Any]:
        """Analyze Tigrinya-specific text quality.
        
        Args:
            text: Tigrinya text to analyze
            
        Returns:
            Tigrinya quality metrics
        """
        quality = {
            'script_consistency': True,
            'punctuation_usage': True,
            'character_validity': True,
            'word_formation': True
        }
        
        if not text:
            return quality
        
        # Check script consistency (all Ethiopic characters should be valid)
        ethiopic_chars = 0
        invalid_chars = 0
        
        for char in text:
            char_code = ord(char)
            if 0x1200 <= char_code <= 0x137F or 0x1380 <= char_code <= 0x139F:
                ethiopic_chars += 1
            elif not char.isspace() and not char in '.,!?;:()[]{}"\'-0123456789':
                if char_code < 128:  # ASCII characters are okay for mixed text
                    continue
                invalid_chars += 1
        
        quality['character_validity'] = invalid_chars == 0
        
        # Check punctuation usage (Tigrinya has specific punctuation marks)
        tigrinya_punctuation = ['።', '፣', '፤', '፥', '፦', '፧', '፨']
        has_tigrinya_punct = any(punct in text for punct in tigrinya_punctuation)
        has_latin_punct = any(punct in text for punct in '.!?;:,')
        
        # Mixed punctuation is acceptable, but prefer Tigrinya punctuation for Tigrinya text
        quality['punctuation_usage'] = has_tigrinya_punct or has_latin_punct
        
        return quality
    
    def _analyze_english_quality(self, text: str) -> Dict[str, Any]:
        """Analyze English-specific text quality.
        
        Args:
            text: English text to analyze
            
        Returns:
            English quality metrics
        """
        quality = {
            'grammar_basic': True,
            'capitalization': True,
            'punctuation': True,
            'word_validity': True
        }
        
        if not text:
            return quality
        
        # Basic capitalization check
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        capitalized_sentences = 0
        for sentence in sentences:
            if sentence and sentence[0].isupper():
                capitalized_sentences += 1
        
        quality['capitalization'] = (capitalized_sentences / len(sentences)) > 0.7 if sentences else True
        
        # Basic punctuation check
        quality['punctuation'] = bool(re.search(r'[.!?]', text))
        
        # Word validity (basic check for reasonable English words)
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if words:
            # Check for reasonable word lengths and patterns
            reasonable_words = sum(1 for word in words if 1 <= len(word) <= 20)
            quality['word_validity'] = (reasonable_words / len(words)) > 0.8
        
        return quality
    
    def _is_language_switch_appropriate(
        self, 
        prompt: str, 
        generated_text: str, 
        prompt_lang: str, 
        output_lang: str
    ) -> bool:
        """Determine if language switch in generation is appropriate.
        
        Args:
            prompt: Input prompt
            generated_text: Generated text
            prompt_lang: Detected prompt language
            output_lang: Detected output language
            
        Returns:
            True if language switch is appropriate
        """
        # Language switch is appropriate in certain contexts:
        # 1. Mixed language prompt
        # 2. Translation request indicators
        # 3. Code-switching context
        
        if prompt_lang == 'mixed':
            return True
        
        # Check for translation indicators
        translation_indicators = [
            'translate', 'translation', 'በትግርኛ', 'in tigrinya', 'in english',
            'ትርጉም', 'ተርጉም', 'ትርጉሙ'
        ]
        
        prompt_lower = prompt.lower()
        if any(indicator in prompt_lower for indicator in translation_indicators):
            return True
        
        # If the generated text is very short, language detection might be unreliable
        if len(generated_text.split()) < 5:
            return True
        
        return False
    
    def _calculate_quality_metrics(
        self, 
        text: str, 
        model=None, 
        tokenizer=None, 
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics.
        
        Args:
            text: Text to analyze
            model: Optional model for perplexity
            tokenizer: Optional tokenizer for perplexity
            device: Device for computation
            
        Returns:
            Quality metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['length'] = len(text)
        metrics['word_count'] = len(text.split())
        
        # Repetition analysis
        metrics['repetition_score'] = self.quality_metrics.calculate_repetition_score(text)
        
        # Diversity analysis
        metrics['diversity'] = self.quality_metrics.calculate_diversity_score(text)
        
        # Coherence analysis
        metrics['coherence_score'] = self.quality_metrics.calculate_coherence_score(text)
        
        # Perplexity (if model and tokenizer available)
        if model is not None and tokenizer is not None:
            try:
                metrics['perplexity'] = self.quality_metrics.calculate_perplexity(
                    model, tokenizer, text, device
                )
            except Exception as e:
                logger.warning(f"Perplexity calculation failed: {e}")
                metrics['perplexity'] = None
        
        return metrics
    
    def _assess_quality(self, metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Assess overall quality based on metrics.
        
        Args:
            metrics: Quality metrics
            
        Returns:
            Tuple of (quality_score, issues_list)
        """
        score = 1.0
        issues = []
        
        # Length check
        if metrics['word_count'] < self.thresholds['min_length']:
            score -= 0.2
            issues.append(f"Text too short ({metrics['word_count']} words)")
        
        # Repetition check
        if metrics['repetition_score'] > self.thresholds['max_repetition']:
            score -= 0.3
            issues.append(f"High repetition detected ({metrics['repetition_score']:.2f})")
        
        # Diversity check
        diversity_score = metrics['diversity']['ttr']
        if diversity_score < self.thresholds['min_diversity']:
            score -= 0.2
            issues.append(f"Low lexical diversity ({diversity_score:.2f})")
        
        # Coherence check
        if metrics['coherence_score'] < self.thresholds['min_coherence']:
            score -= 0.2
            issues.append(f"Low coherence score ({metrics['coherence_score']:.2f})")
        
        # Perplexity check
        if metrics.get('perplexity') and metrics['perplexity'] > self.thresholds['max_perplexity']:
            score -= 0.1
            issues.append(f"High perplexity ({metrics['perplexity']:.2f})")
        
        return max(0.0, score), issues
    
    def _determine_overall_quality(self, quality_score: float) -> str:
        """Determine overall quality rating.
        
        Args:
            quality_score: Calculated quality score
            
        Returns:
            Quality rating string
        """
        if quality_score >= 0.8:
            return 'excellent'
        elif quality_score >= 0.6:
            return 'good'
        elif quality_score >= 0.4:
            return 'fair'
        elif quality_score >= 0.2:
            return 'poor'
        else:
            return 'very_poor'
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving generation quality.
        
        Args:
            results: Quality validation results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Based on issues
        for issue in results['issues']:
            if 'repetition' in issue.lower():
                recommendations.append("Increase repetition penalty or reduce temperature")
            elif 'diversity' in issue.lower():
                recommendations.append("Increase top-k or top-p sampling parameters")
            elif 'coherence' in issue.lower():
                recommendations.append("Adjust temperature or try different generation parameters")
            elif 'short' in issue.lower():
                recommendations.append("Increase max_length parameter")
            elif 'perplexity' in issue.lower():
                recommendations.append("Model may need more training or different prompt")
        
        # Language-specific recommendations
        lang_analysis = results['language_analysis']
        if not lang_analysis.get('language_consistency', True):
            if not lang_analysis.get('language_switch_appropriate', False):
                recommendations.append("Consider using language-specific prompts or fine-tuning")
        
        # Tigrinya-specific recommendations
        tigrinya_quality = lang_analysis.get('tigrinya_quality', {})
        if not tigrinya_quality.get('character_validity', True):
            recommendations.append("Check Tigrinya text encoding and character validity")
        
        # English-specific recommendations
        english_quality = lang_analysis.get('english_quality', {})
        if not english_quality.get('capitalization', True):
            recommendations.append("Improve capitalization in English text generation")
        
        return recommendations
    
    def batch_validate_quality(
        self, 
        prompts: List[str], 
        generated_texts: List[str],
        model=None,
        tokenizer=None,
        device: str = 'cuda'
    ) -> List[Dict[str, Any]]:
        """Validate quality for batch of generated texts.
        
        Args:
            prompts: List of input prompts
            generated_texts: List of generated texts
            model: Optional model for perplexity calculation
            tokenizer: Optional tokenizer for perplexity calculation
            device: Device for computation
            
        Returns:
            List of quality validation results
        """
        if len(prompts) != len(generated_texts):
            raise ValueError("Number of prompts and generated texts must match")
        
        results = []
        for prompt, generated_text in zip(prompts, generated_texts):
            result = self.validate_generation_quality(
                prompt, generated_text, model, tokenizer, device
            )
            results.append(result)
        
        logger.info(f"Batch quality validation completed for {len(results)} samples")
        return results
    
    def get_quality_summary(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics from batch quality validation.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Quality summary statistics
        """
        if not validation_results:
            return {}
        
        quality_scores = [r['quality_score'] for r in validation_results]
        quality_ratings = [r['overall_quality'] for r in validation_results]
        
        # Calculate statistics
        summary = {
            'total_samples': len(validation_results),
            'average_quality_score': sum(quality_scores) / len(quality_scores),
            'quality_distribution': Counter(quality_ratings),
            'common_issues': [],
            'common_recommendations': []
        }
        
        # Aggregate common issues and recommendations
        all_issues = []
        all_recommendations = []
        
        for result in validation_results:
            all_issues.extend(result['issues'])
            all_recommendations.extend(result['recommendations'])
        
        summary['common_issues'] = [item for item, count in Counter(all_issues).most_common(5)]
        summary['common_recommendations'] = [item for item, count in Counter(all_recommendations).most_common(5)]
        
        return summary


class GenerationParameterOptimizer:
    """Optimizer for generation parameters based on quality feedback."""
    
    def __init__(self):
        """Initialize parameter optimizer."""
        self.quality_validator = BilingualQualityValidator()
        logger.info("GenerationParameterOptimizer initialized")
    
    def optimize_parameters(
        self,
        inference_engine,
        test_prompts: List[str],
        target_quality: float = 0.7,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """Optimize generation parameters for better quality.
        
        Args:
            inference_engine: Bilingual inference engine
            test_prompts: List of test prompts for optimization
            target_quality: Target quality score
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization results with best parameters
        """
        if not test_prompts:
            raise ValueError("Test prompts are required for optimization")
        
        best_params = None
        best_score = 0.0
        optimization_history = []
        
        # Parameter ranges to test
        param_ranges = {
            'temperature': [0.7, 0.8, 0.9, 1.0, 1.1],
            'top_k': [30, 40, 50, 60],
            'top_p': [0.8, 0.85, 0.9, 0.95],
            'repetition_penalty': [1.0, 1.05, 1.1, 1.15]
        }
        
        logger.info(f"Starting parameter optimization with {len(test_prompts)} test prompts")
        
        try:
            for iteration in range(max_iterations):
                # Test different parameter combinations
                for temp in param_ranges['temperature']:
                    for top_k in param_ranges['top_k']:
                        for top_p in param_ranges['top_p']:
                            for rep_penalty in param_ranges['repetition_penalty']:
                                
                                # Configure parameters
                                inference_engine.configure_generation(
                                    temperature=temp,
                                    top_k=top_k,
                                    top_p=top_p,
                                    repetition_penalty=rep_penalty
                                )
                                
                                # Generate test outputs
                                generated_texts = []
                                for prompt in test_prompts:
                                    try:
                                        generated = inference_engine.generate_text(prompt)
                                        generated_texts.append(generated)
                                    except Exception as e:
                                        logger.warning(f"Generation failed for prompt: {e}")
                                        generated_texts.append("")
                                
                                # Validate quality
                                quality_results = self.quality_validator.batch_validate_quality(
                                    test_prompts, generated_texts
                                )
                                
                                # Calculate average quality score
                                avg_score = sum(r['quality_score'] for r in quality_results) / len(quality_results)
                                
                                # Track optimization history
                                optimization_history.append({
                                    'iteration': iteration,
                                    'parameters': {
                                        'temperature': temp,
                                        'top_k': top_k,
                                        'top_p': top_p,
                                        'repetition_penalty': rep_penalty
                                    },
                                    'quality_score': avg_score
                                })
                                
                                # Update best parameters
                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_params = {
                                        'temperature': temp,
                                        'top_k': top_k,
                                        'top_p': top_p,
                                        'repetition_penalty': rep_penalty
                                    }
                                
                                # Early stopping if target quality reached
                                if avg_score >= target_quality:
                                    logger.info(f"Target quality {target_quality} reached")
                                    break
                        
                        if best_score >= target_quality:
                            break
                    if best_score >= target_quality:
                        break
                if best_score >= target_quality:
                    break
            
            # Apply best parameters
            if best_params:
                inference_engine.configure_generation(**best_params)
            
            logger.info(f"Parameter optimization completed. Best score: {best_score:.3f}")
            
            return {
                'best_parameters': best_params,
                'best_quality_score': best_score,
                'optimization_history': optimization_history,
                'target_reached': best_score >= target_quality
            }
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            raise RuntimeError(f"Parameter optimization failed: {str(e)}") from e


def create_quality_validator() -> BilingualQualityValidator:
    """Factory function to create bilingual quality validator.
    
    Returns:
        BilingualQualityValidator instance
    """
    return BilingualQualityValidator()


def create_parameter_optimizer() -> GenerationParameterOptimizer:
    """Factory function to create generation parameter optimizer.
    
    Returns:
        GenerationParameterOptimizer instance
    """
    return GenerationParameterOptimizer()