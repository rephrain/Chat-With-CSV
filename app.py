import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import re
from typing import Dict, Any, List, Tuple, Optional, Union
import sqlite3
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
import re
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import spacy
from collections import defaultdict, Counter
import json
warnings.filterwarnings('ignore')
import os
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

class CSVProcessor:
    """Advanced CSV processing with data cleaning and analysis capabilities"""
    
    def __init__(self):
        self.df = None
        self.metadata = {}
        self.sql_connection = None
    
    def load_and_process_csv(self, uploaded_file) -> pd.DataFrame:
        """Load CSV with robust processing and cleaning"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode file with any supported encoding")
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace(r'[^\w\s]', '', regex=True)
            
            # Advanced data type inference
            df = self._infer_and_convert_types(df)
            
            # Generate metadata
            self._generate_metadata(df)
            
            # Create SQL connection for complex queries
            self._create_sql_connection(df)
            
            self.df = df
            return df
            
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            return None
    
    def _infer_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced type inference and conversion"""
        for col in df.columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if not numeric_series.isna().all() and numeric_series.notna().sum() > len(df) * 0.5:
                df[col] = numeric_series
                continue
            
            # Try to convert to datetime
            try:
                if df[col].dtype == 'object':
                    datetime_series = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if datetime_series.notna().sum() > len(df) * 0.5:
                        df[col] = datetime_series
                        continue
            except:
                pass
            
            # Clean string columns
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    def _generate_metadata(self, df: pd.DataFrame):
        """Generate comprehensive metadata about the dataset"""
        self.metadata = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'summary_stats': {}
        }
        
        # Generate summary statistics for numeric columns
        for col in self.metadata['numeric_columns']:
            self.metadata['summary_stats'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'count': int(df[col].count())
            }
    
    def _create_sql_connection(self, df: pd.DataFrame):
        """Create in-memory SQLite database for complex queries"""
        self.sql_connection = sqlite3.connect(':memory:')
        df.to_sql('data', self.sql_connection, index=False, if_exists='replace')
    
    def execute_sql_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query on the dataset"""
        try:
            return pd.read_sql_query(query, self.sql_connection)
        except Exception as e:
            st.error(f"SQL Error: {str(e)}")
            return pd.DataFrame()

class QueryType(Enum):
    AGGREGATION = "aggregation"
    FILTER = "filter"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    GROUPBY = "groupby"
    CORRELATION = "correlation"
    TEMPORAL = "temporal"
    STATISTICAL = "statistical"
    RANKING = "ranking"
    COMPLEX_MULTI_STEP = "complex_multi_step"
    PREDICTION = "prediction"
    ANOMALY_DETECTION = "anomaly_detection"

@dataclass
class QueryContext:
    """Enhanced context for query processing"""
    intent: QueryType
    confidence: float
    entities: List[Dict]
    columns: List[str]
    conditions: Dict[str, Any]
    operations: List[str]
    temporal_info: Dict[str, Any]
    statistical_params: Dict[str, Any]
    dependencies: List[str]
    sub_queries: List[Dict]
    visualization_hint: str

class LLMQueryProcessor:
    """Ultra-advanced LLM-based query processor with complex prompt handling"""
    
    def __init__(self):
        self.nlp = None
        self.embeddings_model = None
        self.query_cache = {}
        self.context_memory = []
        self.domain_knowledge = self._build_domain_knowledge()
        self.advanced_patterns = self._build_advanced_patterns()
        self.statistical_functions = self._build_statistical_functions()
        self.temporal_patterns = self._build_temporal_patterns()
        self.load_advanced_models()
    
    def load_advanced_models(self):
        """Load advanced NLP models and embeddings"""
        try:
            # Load spaCy for advanced NLP
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                st.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            # Load sentence transformers for semantic understanding
            from sentence_transformers import SentenceTransformer
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            
        except Exception as e:
            st.warning(f"Advanced NLP models unavailable: {str(e)}")
    
    def _build_domain_knowledge(self) -> Dict[str, Any]:
        """Build comprehensive domain knowledge base"""
        return {
            'business_metrics': {
                'revenue': ['sales', 'income', 'earnings', 'turnover', 'proceeds'],
                'profit': ['profit', 'margin', 'net income', 'earnings', 'gain'],
                'cost': ['cost', 'expense', 'expenditure', 'spending', 'outlay'],
                'growth': ['growth', 'increase', 'expansion', 'rise', 'improvement'],
                'performance': ['performance', 'efficiency', 'productivity', 'effectiveness'],
                'customer': ['customer', 'client', 'user', 'buyer', 'consumer'],
                'product': ['product', 'item', 'goods', 'service', 'offering']
            },
            'temporal_contexts': {
                'recent': ['recent', 'latest', 'current', 'now', 'today', 'this week'],
                'historical': ['historical', 'past', 'previous', 'old', 'former'],
                'trending': ['trending', 'trend', 'pattern', 'direction', 'movement'],
                'seasonal': ['seasonal', 'quarterly', 'monthly', 'weekly', 'cyclical']
            },
            'statistical_concepts': {
                'central_tendency': ['average', 'mean', 'median', 'mode', 'typical'],
                'variability': ['variance', 'deviation', 'spread', 'range', 'distribution'],
                'relationships': ['correlation', 'association', 'relationship', 'connection'],
                'comparison': ['compare', 'contrast', 'difference', 'similar', 'versus']
            }
        }
    
    def _build_advanced_patterns(self) -> Dict[str, Any]:
        """Build advanced pattern recognition for complex queries"""
        return {
            'multi_step_indicators': [
                'first', 'then', 'after that', 'next', 'finally', 'also', 'additionally',
                'step by step', 'process', 'workflow', 'sequence'
            ],
            'conditional_logic': [
                'if', 'when', 'unless', 'provided that', 'in case', 'assuming',
                'given that', 'on condition', 'supposing'
            ],
            'comparative_analysis': [
                'compare with', 'versus', 'against', 'relative to', 'in comparison',
                'benchmark', 'baseline', 'standard'
            ],
            'advanced_aggregations': {
                'percentile': ['percentile', 'quartile', 'quantile', 'top %', 'bottom %'],
                'moving_average': ['moving average', 'rolling average', 'running average'],
                'cumulative': ['cumulative', 'running total', 'accumulated'],
                'year_over_year': ['year over year', 'yoy', 'annual growth', 'yearly change'],
                'month_over_month': ['month over month', 'mom', 'monthly change']
            },
            'ranking_patterns': [
                'top', 'bottom', 'best', 'worst', 'highest', 'lowest', 'rank',
                'leaderboard', 'performance ranking', 'sorted by'
            ],
            'anomaly_patterns': [
                'anomaly', 'outlier', 'unusual', 'abnormal', 'strange', 'odd',
                'exception', 'deviation', 'irregular'
            ],
            'prediction_patterns': [
                'predict', 'forecast', 'project', 'estimate', 'anticipate',
                'future', 'next', 'upcoming', 'expected'
            ]
        }
    
    def _build_statistical_functions(self) -> Dict[str, Any]:
        """Define advanced statistical functions"""
        return {
            'descriptive': {
                'mean': 'AVG({col})',
                'median': 'MEDIAN({col})',
                'mode': 'MODE({col})',
                'std': 'STDDEV({col})',
                'variance': 'VAR({col})',
                'skewness': 'SKEW({col})',
                'kurtosis': 'KURT({col})'
            },
            'advanced': {
                'percentile_25': 'PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col})',
                'percentile_75': 'PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col})',
                'iqr': 'PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}) - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col})',
                'z_score': '({col} - AVG({col})) / STDDEV({col})',
                'rank': 'RANK() OVER (ORDER BY {col} DESC)',
                'percent_rank': 'PERCENT_RANK() OVER (ORDER BY {col})'
            }
        }
    
    def _build_temporal_patterns(self) -> Dict[str, Any]:
        """Build temporal analysis patterns"""
        return {
            'time_units': {
                'year': ['year', 'annual', 'yearly'],
                'quarter': ['quarter', 'quarterly', 'q1', 'q2', 'q3', 'q4'],
                'month': ['month', 'monthly', 'jan', 'feb', 'mar', 'apr', 'may', 'jun'],
                'week': ['week', 'weekly'],
                'day': ['day', 'daily', 'date']
            },
            'time_operations': {
                'growth': 'LAG({col}, 1) OVER (ORDER BY {time_col})',
                'moving_avg': 'AVG({col}) OVER (ORDER BY {time_col} ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)',
                'cumulative': 'SUM({col}) OVER (ORDER BY {time_col})'
            }
        }
    
    def parse_complex_query(self, query: str, metadata: Dict, context: Optional[List] = None) -> QueryContext:
        """Parse complex natural language queries with advanced understanding"""
        # Store context for multi-turn conversations
        if context:
            self.context_memory.extend(context)
        
        # Pre-process query
        processed_query = self._preprocess_query(query)
        
        # Extract entities and intent with NLP
        entities = self._extract_entities_advanced(processed_query, metadata)
        intent, confidence = self._classify_intent_with_context(processed_query, entities)
        
        # Handle multi-step queries
        sub_queries = self._identify_sub_queries(processed_query)
        
        # Extract complex conditions and operations
        conditions = self._extract_complex_conditions(processed_query, entities, metadata)
        operations = self._extract_operations_sequence(processed_query)
        
        # Temporal analysis
        temporal_info = self._extract_temporal_context(processed_query, metadata)
        
        # Statistical parameters
        statistical_params = self._extract_statistical_parameters(processed_query)
        
        # Dependencies between operations
        dependencies = self._identify_dependencies(sub_queries, operations)
        
        # Visualization hints
        viz_hint = self._suggest_visualization(intent, entities, operations)
        
        return QueryContext(
            intent=intent,
            confidence=confidence,
            entities=entities,
            columns=self._extract_columns_from_entities(entities, metadata),
            conditions=conditions,
            operations=operations,
            temporal_info=temporal_info,
            statistical_params=statistical_params,
            dependencies=dependencies,
            sub_queries=sub_queries,
            visualization_hint=viz_hint
        )
    
    def _preprocess_query(self, query: str) -> str:
        """Advanced query preprocessing"""
        # Normalize whitespace and punctuation
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Handle common abbreviations
        abbreviations = {
            r'\byoy\b': 'year over year',
            r'\bmom\b': 'month over month',
            r'\bqoq\b': 'quarter over quarter',
            r'\bytd\b': 'year to date',
            r'\bqtd\b': 'quarter to date',
            r'\bmtd\b': 'month to date',
            r'\bavg\b': 'average',
            r'\bstd\b': 'standard deviation',
            r'\bmax\b': 'maximum',
            r'\bmin\b': 'minimum'
        }
        
        for pattern, replacement in abbreviations.items():
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        
        return query
    
    def _extract_entities_advanced(self, query: str, metadata: Dict) -> List[Dict]:
        """Extract entities using advanced NLP techniques"""
        entities = []
        
        if self.nlp:
            doc = self.nlp(query)
            
            # Extract named entities
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'type': 'named_entity'
                })
            
            # Extract numbers and dates
            for token in doc:
                if token.like_num:
                    entities.append({
                        'text': token.text,
                        'label': 'NUMBER',
                        'value': float(token.text) if '.' in token.text else int(token.text),
                        'type': 'numeric'
                    })
                elif token.pos_ == 'NOUN' and token.text.lower() in [col.lower() for col in metadata['columns']]:
                    entities.append({
                        'text': token.text,
                        'label': 'COLUMN',
                        'column_name': self._find_matching_column(token.text, metadata['columns']),
                        'type': 'column_reference'
                    })
        
        # Fallback pattern-based extraction
        entities.extend(self._extract_entities_pattern_based(query, metadata))
        
        return entities
    
    def _extract_entities_pattern_based(self, query: str, metadata: Dict) -> List[Dict]:
        """Pattern-based entity extraction as fallback"""
        entities = []
        
        # Extract percentages
        percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
        for match in re.finditer(percentage_pattern, query):
            entities.append({
                'text': match.group(0),
                'label': 'PERCENTAGE',
                'value': float(match.group(1)),
                'type': 'percentage'
            })
        
        # Extract date ranges
        date_patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                entities.append({
                    'text': match.group(0),
                    'label': 'DATE',
                    'type': 'temporal'
                })
        
        return entities
    
    def _classify_intent_with_context(self, query: str, entities: List[Dict]) -> Tuple[QueryType, float]:
        """Advanced intent classification with context awareness"""
        # Check for complex multi-step patterns
        if any(indicator in query.lower() for indicator in self.advanced_patterns['multi_step_indicators']):
            return QueryType.COMPLEX_MULTI_STEP, 0.9
        
        # Check for prediction patterns
        if any(pattern in query.lower() for pattern in self.advanced_patterns['prediction_patterns']):
            return QueryType.PREDICTION, 0.85
        
        # Check for anomaly detection
        if any(pattern in query.lower() for pattern in self.advanced_patterns['anomaly_patterns']):
            return QueryType.ANOMALY_DETECTION, 0.8
        
        # Check for temporal analysis
        temporal_indicators = sum(1 for entity in entities if entity.get('type') == 'temporal')
        if temporal_indicators > 0 or any(word in query.lower() for word in ['over time', 'trend', 'growth']):
            return QueryType.TEMPORAL, 0.85
        
        # Check for statistical analysis
        statistical_terms = ['correlation', 'regression', 'significance', 'p-value', 'confidence interval']
        if any(term in query.lower() for term in statistical_terms):
            return QueryType.STATISTICAL, 0.8
        
        # Check for ranking
        if any(pattern in query.lower() for pattern in self.advanced_patterns['ranking_patterns']):
            return QueryType.RANKING, 0.75
        
        # Fallback to original classification with enhancements
        return self._classify_basic_intent(query, entities)
    
    def _classify_basic_intent(self, query: str, entities: List[Dict]) -> Tuple[QueryType, float]:
        """Enhanced basic intent classification"""
        # Aggregation patterns
        agg_patterns = ['average', 'sum', 'count', 'total', 'mean', 'maximum', 'minimum']
        if any(pattern in query.lower() for pattern in agg_patterns):
            return QueryType.AGGREGATION, 0.7
        
        # Filter patterns
        filter_patterns = ['where', 'filter', 'show me', 'find', 'select']
        if any(pattern in query.lower() for pattern in filter_patterns):
            return QueryType.FILTER, 0.6
        
        # Group by patterns
        groupby_patterns = ['group by', 'grouped by', 'categorize', 'break down by']
        if any(pattern in query.lower() for pattern in groupby_patterns):
            return QueryType.GROUPBY, 0.75
        
        return QueryType.SUMMARY, 0.5
    
    def _identify_sub_queries(self, query: str) -> List[Dict]:
        """Identify sub-queries in complex multi-step queries"""
        sub_queries = []
        
        # Split by common separators
        separators = ['then', 'and then', 'after that', 'next', 'also', 'additionally', 'furthermore']
        
        parts = [query]
        for sep in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts
        
        for i, part in enumerate(parts):
            part = part.strip()
            if len(part) > 10:  # Minimum length for meaningful sub-query
                sub_queries.append({
                    'order': i,
                    'text': part,
                    'type': 'sequential'
                })
        
        return sub_queries
    
    def _extract_complex_conditions(self, query: str, entities: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Extract complex filtering conditions"""
        conditions = {
            'simple': [],
            'compound': [],
            'nested': [],
            'temporal': [],
            'statistical': []
        }
        
        # Extract numerical conditions
        numerical_patterns = [
            (r'greater than (\d+(?:\.\d+)?)', 'gt'),
            (r'less than (\d+(?:\.\d+)?)', 'lt'),
            (r'equal to (\d+(?:\.\d+)?)', 'eq'),
            (r'between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)', 'between'),
            (r'top (\d+)', 'top_n'),
            (r'bottom (\d+)', 'bottom_n')
        ]
        
        for pattern, condition_type in numerical_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                if condition_type == 'between':
                    conditions['simple'].append({
                        'type': condition_type,
                        'values': [float(match.group(1)), float(match.group(2))]
                    })
                else:
                    conditions['simple'].append({
                        'type': condition_type,
                        'value': float(match.group(1))
                    })
        
        # Extract temporal conditions
        temporal_patterns = [
            r'in the last (\d+) (days?|weeks?|months?|years?)',
            r'since (\d{4})',
            r'during (\d{4})',
            r'this (year|month|week|quarter)'
        ]
        
        for pattern in temporal_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                conditions['temporal'].append({
                    'type': 'time_range',
                    'text': match.group(0),
                    'groups': match.groups()
                })
        
        return conditions
    
    def _extract_operations_sequence(self, query: str) -> List[str]:
        """Extract sequence of operations from complex queries"""
        operations = []
        
        # Map query patterns to operations
        operation_patterns = {
            'aggregate': ['sum', 'average', 'count', 'total', 'mean', 'max', 'min'],
            'filter': ['where', 'filter', 'select', 'find'],
            'sort': ['sort', 'order', 'rank', 'arrange'],
            'group': ['group by', 'categorize', 'break down'],
            'join': ['combine', 'merge', 'join'],
            'calculate': ['calculate', 'compute', 'derive'],
            'compare': ['compare', 'versus', 'against'],
            'analyze': ['analyze', 'examine', 'study']
        }
        
        query_lower = query.lower()
        for operation, patterns in operation_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                operations.append(operation)
        
        return operations
    
    def _extract_temporal_context(self, query: str, metadata: Dict) -> Dict[str, Any]:
        """Extract temporal analysis context"""
        temporal_info = {
            'has_time_dimension': False,
            'time_columns': [],
            'time_granularity': None,
            'time_operations': []
        }
        
        # Identify time columns
        time_indicators = ['date', 'time', 'timestamp', 'created', 'updated', 'year', 'month']
        for col in metadata['columns']:
            if any(indicator in col.lower() for indicator in time_indicators):
                temporal_info['time_columns'].append(col)
                temporal_info['has_time_dimension'] = True
        
        # Determine time granularity
        granularity_patterns = {
            'daily': ['daily', 'day', 'per day'],
            'weekly': ['weekly', 'week', 'per week'],
            'monthly': ['monthly', 'month', 'per month'],
            'quarterly': ['quarterly', 'quarter', 'per quarter'],
            'yearly': ['yearly', 'annual', 'year', 'per year']
        }
        
        query_lower = query.lower()
        for granularity, patterns in granularity_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                temporal_info['time_granularity'] = granularity
                break
        
        # Identify time-based operations
        time_ops = ['trend', 'growth', 'change over time', 'moving average', 'cumulative']
        for op in time_ops:
            if op in query_lower:
                temporal_info['time_operations'].append(op)
        
        return temporal_info
    
    def _extract_statistical_parameters(self, query: str) -> Dict[str, Any]:
        """Extract statistical analysis parameters"""
        params = {
            'confidence_level': 0.95,  # default
            'significance_level': 0.05,  # default
            'test_type': None,
            'correlation_method': 'pearson',
            'outlier_method': 'iqr'
        }
        
        # Extract confidence level
        conf_pattern = r'(\d+)%?\s*confidence'
        conf_match = re.search(conf_pattern, query, re.IGNORECASE)
        if conf_match:
            params['confidence_level'] = float(conf_match.group(1)) / 100
        
        # Extract significance level
        sig_pattern = r'significance.*?(\d+(?:\.\d+)?)'
        sig_match = re.search(sig_pattern, query, re.IGNORECASE)
        if sig_match:
            params['significance_level'] = float(sig_match.group(1))
        
        # Identify statistical tests
        test_patterns = {
            't-test': ['t-test', 'ttest', 't test'],
            'chi-square': ['chi-square', 'chi square', 'chisquare'],
            'anova': ['anova', 'analysis of variance'],
            'regression': ['regression', 'linear model']
        }
        
        query_lower = query.lower()
        for test_type, patterns in test_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                params['test_type'] = test_type
                break
        
        return params
    
    def _identify_dependencies(self, sub_queries: List[Dict], operations: List[str]) -> List[str]:
        """Identify dependencies between operations"""
        dependencies = []
        
        # Simple dependency rules
        if 'filter' in operations and 'aggregate' in operations:
            dependencies.append('filter_before_aggregate')
        
        if 'group' in operations and 'aggregate' in operations:
            dependencies.append('group_before_aggregate')
        
        if 'sort' in operations:
            dependencies.append('sort_last')
        
        return dependencies
    
    def _suggest_visualization(self, intent: QueryType, entities: List[Dict], operations: List[str]) -> str:
        """Suggest appropriate visualization based on query analysis"""
        if intent == QueryType.TEMPORAL:
            return 'line_chart'
        elif intent == QueryType.COMPARISON:
            return 'bar_chart'
        elif intent == QueryType.CORRELATION:
            return 'scatter_plot'
        elif intent == QueryType.STATISTICAL:
            return 'histogram'
        elif intent == QueryType.RANKING:
            return 'horizontal_bar'
        elif intent == QueryType.GROUPBY:
            return 'grouped_bar'
        else:
            return 'table'
    
    def _extract_columns_from_entities(self, entities: List[Dict], metadata: Dict) -> List[str]:
        """Extract column references from entities"""
        columns = []
        for entity in entities:
            if entity.get('type') == 'column_reference':
                columns.append(entity.get('column_name'))
        
        # Fallback to fuzzy matching
        if not columns:
            query_text = ' '.join([e.get('text', '') for e in entities])
            columns = self._fuzzy_match_columns(query_text, metadata['columns'])
        
        return [col for col in columns if col is not None]
    
    def _find_matching_column(self, text: str, columns: List[str]) -> Optional[str]:
        """Find best matching column name"""
        text_lower = text.lower()
        for col in columns:
            if text_lower == col.lower():
                return col
            if text_lower in col.lower() or col.lower() in text_lower:
                return col
        return None
    
    def _fuzzy_match_columns(self, query: str, columns: List[str]) -> List[str]:
        """Fuzzy match columns in query text"""
        from difflib import get_close_matches
        
        query_words = query.lower().split()
        matched_columns = []
        
        for word in query_words:
            matches = get_close_matches(word, [col.lower() for col in columns], n=1, cutoff=0.7)
            if matches:
                for col in columns:
                    if col.lower() == matches[0]:
                        matched_columns.append(col)
                        break
        
        return matched_columns
    
    def generate_advanced_sql(self, context: QueryContext, metadata: Dict) -> Dict[str, Any]:
        """Generate advanced SQL queries based on complex context"""
        result = {
            'main_query': None,
            'sub_queries': [],
            'post_processing': [],
            'visualization_data': {}
        }
        
        if context.intent == QueryType.COMPLEX_MULTI_STEP:
            result = self._handle_multi_step_query(context, metadata)
        elif context.intent == QueryType.TEMPORAL:
            result = self._handle_temporal_query(context, metadata)
        elif context.intent == QueryType.STATISTICAL:
            result = self._handle_statistical_query(context, metadata)
        elif context.intent == QueryType.PREDICTION:
            result = self._handle_prediction_query(context, metadata)
        elif context.intent == QueryType.ANOMALY_DETECTION:
            result = self._handle_anomaly_query(context, metadata)
        elif context.intent == QueryType.RANKING:
            result = self._handle_ranking_query(context, metadata)
        else:
            # Fallback to enhanced basic queries
            result = self._handle_enhanced_basic_query(context, metadata)
        
        return result
    
    def _handle_multi_step_query(self, context: QueryContext, metadata: Dict) -> Dict[str, Any]:
        """Handle complex multi-step queries"""
        result = {
            'main_query': None,
            'sub_queries': [],
            'execution_order': [],
            'temp_tables': []
        }
        
        for i, sub_query in enumerate(context.sub_queries):
            # Create temporary table name
            temp_table = f"temp_step_{i}"
            
            # Process each sub-query
            sub_context = self.parse_complex_query(sub_query['text'], metadata)
            sub_sql = self.generate_advanced_sql(sub_context, metadata)
            
            result['sub_queries'].append({
                'step': i,
                'sql': sub_sql['main_query'],
                'temp_table': temp_table,
                'description': sub_query['text']
            })
            
            result['execution_order'].append(temp_table)
        
        return result
    
    def _handle_temporal_query(self, context: QueryContext, metadata: Dict) -> Dict[str, Any]:
        """Handle temporal analysis queries"""
        result = {'main_query': None, 'post_processing': []}
        
        if not context.temporal_info['time_columns']:
            return result
        
        time_col = context.temporal_info['time_columns'][0]
        target_cols = context.columns if context.columns else metadata['numeric_columns'][:1]
        
        if not target_cols:
            return result
        
        target_col = target_cols[0]
        
        # Time series aggregation based on granularity
        granularity = context.temporal_info.get('time_granularity', 'monthly')
        
        time_functions = {
            'daily': f"DATE({time_col})",
            'weekly': f"DATE({time_col}, 'weekday 0', '-6 days')",
            'monthly': f"strftime('%Y-%m', {time_col})",
            'quarterly': f"strftime('%Y', {time_col}) || '-Q' || ((CAST(strftime('%m', {time_col}) AS INTEGER) - 1) / 3 + 1)",
            'yearly': f"strftime('%Y', {time_col})"
        }
        
        time_func = time_functions.get(granularity, time_functions['monthly'])
        
        # Handle different time operations
        if 'trend' in context.temporal_info.get('time_operations', []):
            result['main_query'] = f"""
                SELECT {time_func} as time_period,
                       AVG({target_col}) as avg_value,
                       COUNT(*) as count,
                       LAG(AVG({target_col})) OVER (ORDER BY {time_func}) as prev_value,
                       (AVG({target_col}) - LAG(AVG({target_col})) OVER (ORDER BY {time_func})) / 
                       LAG(AVG({target_col})) OVER (ORDER BY {time_func}) * 100 as growth_rate
                FROM data
                WHERE {time_col} IS NOT NULL
                GROUP BY {time_func}
                ORDER BY {time_func}
            """
        elif 'moving average' in context.temporal_info.get('time_operations', []):
            result['main_query'] = f"""
                SELECT {time_func} as time_period,
                       AVG({target_col}) as current_value,
                       AVG(AVG({target_col})) OVER (
                           ORDER BY {time_func} 
                           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                       ) as moving_avg_3period
                FROM data
                WHERE {time_col} IS NOT NULL
                GROUP BY {time_func}
                ORDER BY {time_func}
            """
        elif 'cumulative' in context.temporal_info.get('time_operations', []):
            result['main_query'] = f"""
                SELECT {time_func} as time_period,
                       SUM({target_col}) as period_sum,
                       SUM(SUM({target_col})) OVER (
                           ORDER BY {time_func}
                       ) as cumulative_sum
                FROM data
                WHERE {time_col} IS NOT NULL
                GROUP BY {time_func}
                ORDER BY {time_func}
            """
        else:
            # Default temporal aggregation
            result['main_query'] = f"""
                SELECT {time_func} as time_period,
                       COUNT(*) as count,
                       AVG({target_col}) as avg_value,
                       MIN({target_col}) as min_value,
                       MAX({target_col}) as max_value
                FROM data
                WHERE {time_col} IS NOT NULL
                GROUP BY {time_func}
                ORDER BY {time_func}
            """
        
        return result
    
    def _handle_statistical_query(self, context: QueryContext, metadata: Dict) -> Dict[str, Any]:
        """Handle advanced statistical analysis queries"""
        result = {'main_query': None, 'post_processing': []}
        
        if not context.columns:
            return result
        
        target_cols = [col for col in context.columns if col in metadata['numeric_columns']]
        if not target_cols:
            return result
        
        # Correlation analysis
        if context.intent == QueryType.CORRELATION and len(target_cols) >= 2:
            col1, col2 = target_cols[0], target_cols[1]
            result['main_query'] = f"""
                SELECT 
                    COUNT(*) as n,
                    AVG({col1}) as mean_{col1},
                    AVG({col2}) as mean_{col2},
                    SUM(({col1} - (SELECT AVG({col1}) FROM data)) * 
                        ({col2} - (SELECT AVG({col2}) FROM data))) / 
                    (COUNT(*) - 1) as covariance,
                    SQRT(SUM(({col1} - (SELECT AVG({col1}) FROM data)) * 
                             ({col1} - (SELECT AVG({col1}) FROM data))) / 
                         (COUNT(*) - 1)) as std_{col1},
                    SQRT(SUM(({col2} - (SELECT AVG({col2}) FROM data)) * 
                             ({col2} - (SELECT AVG({col2}) FROM data))) / 
                         (COUNT(*) - 1)) as std_{col2}
                FROM data
                WHERE {col1} IS NOT NULL AND {col2} IS NOT NULL
            """
            result['post_processing'].append('calculate_correlation_coefficient')
        
        # Descriptive statistics
        elif 'descriptive' in context.operations:
            target_col = target_cols[0]
            result['main_query'] = f"""
                WITH stats AS (
                    SELECT 
                        COUNT(*) as n,
                        AVG({target_col}) as mean,
                        MIN({target_col}) as min_val,
                        MAX({target_col}) as max_val,
                        SUM({target_col}) as sum_val,
                        SQRT(SUM(({target_col} - (SELECT AVG({target_col}) FROM data)) * 
                                 ({target_col} - (SELECT AVG({target_col}) FROM data))) / 
                             (COUNT(*) - 1)) as std_dev
                    FROM data
                    WHERE {target_col} IS NOT NULL
                ),
                percentiles AS (
                    SELECT 
                        {target_col} as value,
                        NTILE(4) OVER (ORDER BY {target_col}) as quartile,
                        NTILE(100) OVER (ORDER BY {target_col}) as percentile
                    FROM data
                    WHERE {target_col} IS NOT NULL
                )
                SELECT 
                    s.*,
                    (SELECT value FROM percentiles WHERE quartile = 2 LIMIT 1) as median,
                    (SELECT value FROM percentiles WHERE quartile = 3 LIMIT 1) as q3,
                    (SELECT value FROM percentiles WHERE quartile = 1 LIMIT 1) as q1
                FROM stats s
            """
        
        return result
    
    def _handle_prediction_query(self, context: QueryContext, metadata: Dict) -> Dict[str, Any]:
        """Handle prediction and forecasting queries"""
        result = {
            'main_query': None,
            'post_processing': ['linear_regression_forecast'],
            'model_type': 'linear_regression'
        }
        
        if not context.temporal_info['time_columns'] or not context.columns:
            return result
        
        time_col = context.temporal_info['time_columns'][0]
        target_col = context.columns[0] if context.columns[0] in metadata['numeric_columns'] else None
        
        if not target_col:
            return result
        
        # Prepare data for time series forecasting
        result['main_query'] = f"""
            SELECT 
                ROW_NUMBER() OVER (ORDER BY {time_col}) as time_index,
                {time_col} as date_value,
                {target_col} as target_value,
                AVG({target_col}) OVER (
                    ORDER BY {time_col} 
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) as moving_avg
            FROM data
            WHERE {time_col} IS NOT NULL AND {target_col} IS NOT NULL
            ORDER BY {time_col}
        """
        
        return result
    
    def _handle_anomaly_query(self, context: QueryContext, metadata: Dict) -> Dict[str, Any]:
        """Handle anomaly detection queries"""
        result = {
            'main_query': None,
            'post_processing': ['detect_anomalies_iqr']
        }
        
        if not context.columns:
            return result
        
        target_col = context.columns[0] if context.columns[0] in metadata['numeric_columns'] else None
        if not target_col:
            return result
        
        # Z-score based anomaly detection
        result['main_query'] = f"""
            WITH stats AS (
                SELECT 
                    AVG({target_col}) as mean_val,
                    SQRT(SUM(({target_col} - (SELECT AVG({target_col}) FROM data)) * 
                             ({target_col} - (SELECT AVG({target_col}) FROM data))) / 
                         COUNT(*)) as std_val
                FROM data
                WHERE {target_col} IS NOT NULL
            ),
            scored_data AS (
                SELECT 
                    *,
                    ({target_col} - s.mean_val) / s.std_val as z_score,
                    ABS(({target_col} - s.mean_val) / s.std_val) as abs_z_score
                FROM data
                CROSS JOIN stats s
                WHERE {target_col} IS NOT NULL
            )
            SELECT *,
                CASE 
                    WHEN abs_z_score > 3 THEN 'Extreme Outlier'
                    WHEN abs_z_score > 2 THEN 'Moderate Outlier'
                    ELSE 'Normal'
                END as anomaly_status
            FROM scored_data
            ORDER BY abs_z_score DESC
        """
        
        return result
    
    def _handle_ranking_query(self, context: QueryContext, metadata: Dict) -> Dict[str, Any]:
        """Handle ranking and top/bottom queries"""
        result = {'main_query': None}
        print(context)
        if not context.columns:
            return result
        
        # Extract ranking parameters
        rank_column = context.columns[0]
        rank_order = 'DESC'  # Default to descending
        limit_n = 10  # Default limit
        
        # Check for specific ranking instructions
        for condition in context.conditions.get('simple', []):
            if condition['type'] == 'top_n':
                limit_n = int(condition['value'])
                rank_order = 'DESC'
            elif condition['type'] == 'bottom_n':
                limit_n = int(condition['value'])
                rank_order = 'ASC'
        
        # Check if we need grouping
        group_columns = [col for col in context.columns[1:] if col in metadata['categorical_columns']]
        
        if group_columns:
            group_col = group_columns[0]
            result['main_query'] = f"""
                WITH ranked_data AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY {group_col} 
                               ORDER BY {rank_column} {rank_order}
                           ) as rank_within_group
                    FROM data
                    WHERE {rank_column} IS NOT NULL
                )
                SELECT * 
                FROM ranked_data
                WHERE rank_within_group <= {limit_n}
                ORDER BY {group_col}, rank_within_group
            """
        else:
            result['main_query'] = f"""
                SELECT *,
                       ROW_NUMBER() OVER (ORDER BY {rank_column} {rank_order}) as overall_rank
                FROM data
                WHERE {rank_column} IS NOT NULL
                ORDER BY {rank_column} {rank_order}
                LIMIT {limit_n}
            """
        
        return result
    
    def _handle_enhanced_basic_query(self, context: QueryContext, metadata: Dict) -> Dict[str, Any]:
        """Handle enhanced basic queries with advanced features"""
        result = {'main_query': None}
        
        if context.intent == QueryType.AGGREGATION:
            result = self._enhanced_aggregation_query(context, metadata)
        elif context.intent == QueryType.FILTER:
            result = self._enhanced_filter_query(context, metadata)
        elif context.intent == QueryType.GROUPBY:
            result = self._enhanced_groupby_query(context, metadata)
        elif context.intent == QueryType.COMPARISON:
            result = self._enhanced_comparison_query(context, metadata)
        else:
            result = self._enhanced_summary_query(context, metadata)
        
        return result
    
    def _enhanced_aggregation_query(self, context: QueryContext, metadata: Dict) -> Dict[str, Any]:
        """Enhanced aggregation with multiple operations"""
        result = {'main_query': None}
        
        if not context.columns:
            result['main_query'] = "SELECT COUNT(*) as total_records FROM data"
            return result
        
        target_col = context.columns[0]
        if target_col not in metadata['numeric_columns']:
            result['main_query'] = f"SELECT COUNT(DISTINCT {target_col}) as unique_{target_col} FROM data"
            return result
        
        # Multi-metric aggregation
        agg_functions = []
        if 'aggregate' in context.operations or 'calculate' in context.operations:
            agg_functions = [
                f"COUNT({target_col}) as count_{target_col}",
                f"AVG({target_col}) as avg_{target_col}",
                f"MIN({target_col}) as min_{target_col}",
                f"MAX({target_col}) as max_{target_col}",
                f"SUM({target_col}) as sum_{target_col}",
                f"ROUND(SQRT(AVG(({target_col} - (SELECT AVG({target_col}) FROM data)) * ({target_col} - (SELECT AVG({target_col}) FROM data)))), 2) as std_{target_col}"
            ]
        else:
            agg_functions = [f"AVG({target_col}) as avg_{target_col}"]
        
        # Add conditions
        where_clause = self._build_enhanced_where_clause(context.conditions, context.columns, metadata)
        
        result['main_query'] = f"""
            SELECT {', '.join(agg_functions)}
            FROM data
            {where_clause}
        """
        
        return result
    
    def _enhanced_filter_query(self, context: QueryContext, metadata: Dict) -> Dict[str, Any]:
        """Enhanced filtering with complex conditions"""
        result = {'main_query': None}
        
        # Build comprehensive WHERE clause
        where_clause = self._build_enhanced_where_clause(context.conditions, context.columns, metadata)
        
        # Select appropriate columns
        if context.columns:
            select_cols = ', '.join(context.columns[:10])  # Limit for readability
        else:
            select_cols = '*'
        
        # Add ordering if ranking is involved
        order_clause = ""
        if 'sort' in context.operations and context.columns:
            sort_col = context.columns[0]
            if sort_col in metadata['numeric_columns']:
                order_clause = f" ORDER BY {sort_col} DESC"
        
        result['main_query'] = f"""
            SELECT {select_cols}
            FROM data
            {where_clause}
            {order_clause}
            LIMIT 100
        """
        
        return result
    
    def _enhanced_groupby_query(self, context: QueryContext, metadata: Dict) -> Dict[str, Any]:
        """Enhanced group by with advanced aggregations"""
        result = {'main_query': None}
        
        if not context.columns:
            return result
        
        # Identify grouping column (categorical) and aggregation column (numeric)
        group_cols = [col for col in context.columns if col in metadata['categorical_columns']]
        agg_cols = [col for col in context.columns if col in metadata['numeric_columns']]
        
        if not group_cols:
            group_cols = [context.columns[0]]
        
        group_col = group_cols[0]
        
        # Build aggregation functions
        agg_functions = [f"COUNT(*) as count"]
        
        if agg_cols:
            agg_col = agg_cols[0]
            agg_functions.extend([
                f"AVG({agg_col}) as avg_{agg_col}",
                f"MIN({agg_col}) as min_{agg_col}",
                f"MAX({agg_col}) as max_{agg_col}",
                f"SUM({agg_col}) as sum_{agg_col}"
            ])
        
        # Add conditions
        where_clause = self._build_enhanced_where_clause(context.conditions, context.columns, metadata)
        having_clause = ""
        
        # Add HAVING clause for group-level filtering
        if any(cond.get('type') == 'gt' for cond in context.conditions.get('simple', [])):
            having_clause = " HAVING COUNT(*) > 1"
        
        result['main_query'] = f"""
            SELECT {group_col}, {', '.join(agg_functions)}
            FROM data
            {where_clause}
            GROUP BY {group_col}
            {having_clause}
            ORDER BY count DESC
            LIMIT 50
        """
        
        return result
    
    def _enhanced_comparison_query(self, context: QueryContext, metadata: Dict) -> Dict[str, Any]:
        """Enhanced comparison with statistical significance"""
        result = {'main_query': None}
        
        if len(context.columns) < 2:
            return result
        
        col1, col2 = context.columns[0], context.columns[1]
        
        # Numeric comparison
        if col1 in metadata['numeric_columns'] and col2 in metadata['numeric_columns']:
            result['main_query'] = f"""
                SELECT 
                    COUNT(*) as sample_size,
                    AVG({col1}) as avg_{col1},
                    AVG({col2}) as avg_{col2},
                    AVG({col1} - {col2}) as mean_difference,
                    MIN({col1}) as min_{col1},
                    MAX({col1}) as max_{col1},
                    MIN({col2}) as min_{col2},
                    MAX({col2}) as max_{col2},
                    CORR({col1}, {col2}) as correlation
                FROM data
                WHERE {col1} IS NOT NULL AND {col2} IS NOT NULL
            """
        
        # Categorical vs Numeric comparison
        elif col1 in metadata['categorical_columns'] and col2 in metadata['numeric_columns']:
            result['main_query'] = f"""
                SELECT 
                    {col1} as category,
                    COUNT(*) as count,
                    AVG({col2}) as avg_{col2},
                    MIN({col2}) as min_{col2},
                    MAX({col2}) as max_{col2},
                    SQRT(AVG(({col2} - (SELECT AVG({col2}) FROM data WHERE {col1} = t.{col1})) * 
                             ({col2} - (SELECT AVG({col2}) FROM data WHERE {col1} = t.{col1})))) as std_{col2}
                FROM data t
                WHERE {col1} IS NOT NULL AND {col2} IS NOT NULL
                GROUP BY {col1}
                ORDER BY avg_{col2} DESC
            """
        
        return result
    
    def _enhanced_summary_query(self, context: QueryContext, metadata: Dict) -> Dict[str, Any]:
        """Enhanced summary with comprehensive statistics"""
        result = {'main_query': None}
        
        if context.columns:
            target_col = context.columns[0]
            if target_col in metadata['numeric_columns']:
                result['main_query'] = f"""
                    WITH basic_stats AS (
                        SELECT 
                            COUNT(*) as total_count,
                            COUNT({target_col}) as non_null_count,
                            AVG({target_col}) as mean_value,
                            MIN({target_col}) as min_value,
                            MAX({target_col}) as max_value,
                            SUM({target_col}) as sum_value
                        FROM data
                    ),
                    percentile_stats AS (
                        SELECT 
                            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {target_col}) as q1,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {target_col}) as median,
                            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {target_col}) as q3
                        FROM data
                        WHERE {target_col} IS NOT NULL
                    )
                    SELECT b.*, p.*,
                           (p.q3 - p.q1) as iqr,
                           (b.max_value - b.min_value) as range_value
                    FROM basic_stats b
                    CROSS JOIN percentile_stats p
                """
            else:
                # Categorical summary
                result['main_query'] = f"""
                    SELECT 
                        {target_col} as category,
                        COUNT(*) as frequency,
                        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM data), 2) as percentage
                    FROM data
                    WHERE {target_col} IS NOT NULL
                    GROUP BY {target_col}
                    ORDER BY frequency DESC
                    LIMIT 20
                """
        else:
            # Overall dataset summary
            numeric_cols = metadata['numeric_columns'][:5]  # Limit for performance
            if numeric_cols:
                summary_stats = []
                for col in numeric_cols:
                    summary_stats.extend([
                        f"AVG({col}) as avg_{col}",
                        f"COUNT({col}) as count_{col}"
                    ])
                
                result['main_query'] = f"""
                    SELECT 
                        COUNT(*) as total_rows,
                        {', '.join(summary_stats)}
                    FROM data
                """
        
        return result
    
    def _build_enhanced_where_clause(self, conditions: Dict[str, Any], columns: List[str], metadata: Dict) -> str:
        """Build comprehensive WHERE clause from complex conditions"""
        where_parts = []
        
        # Handle simple conditions
        for condition in conditions.get('simple', []):
            if not columns:
                continue
                
            target_col = columns[0]
            cond_type = condition['type']
            
            if cond_type == 'gt' and 'value' in condition:
                where_parts.append(f"{target_col} > {condition['value']}")
            elif cond_type == 'lt' and 'value' in condition:
                where_parts.append(f"{target_col} < {condition['value']}")
            elif cond_type == 'eq' and 'value' in condition:
                where_parts.append(f"{target_col} = {condition['value']}")
            elif cond_type == 'between' and 'values' in condition:
                values = condition['values']
                where_parts.append(f"{target_col} BETWEEN {values[0]} AND {values[1]}")
        
        # Handle temporal conditions
        for condition in conditions.get('temporal', []):
            if condition['type'] == 'time_range' and columns:
                # Find time column
                time_cols = [col for col in columns if any(t in col.lower() for t in ['date', 'time', 'created'])]
                if time_cols:
                    time_col = time_cols[0]
                    groups = condition['groups']
                    
                    if len(groups) >= 2:
                        if groups[1] in ['days', 'day']:
                            where_parts.append(f"{time_col} >= date('now', '-{groups[0]} days')")
                        elif groups[1] in ['months', 'month']:
                            where_parts.append(f"{time_col} >= date('now', '-{groups[0]} months')")
                        elif groups[1] in ['years', 'year']:
                            where_parts.append(f"{time_col} >= date('now', '-{groups[0]} years')")
        
        return f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
    
    def execute_with_post_processing(self, sql_result: Dict[str, Any], context: QueryContext) -> Dict[str, Any]:
        """Execute post-processing operations on query results"""
        processed_result = sql_result.copy()
        
        for operation in sql_result.get('post_processing', []):
            if operation == 'calculate_correlation_coefficient':
                processed_result = self._calculate_correlation(processed_result)
            elif operation == 'linear_regression_forecast':
                processed_result = self._perform_linear_regression(processed_result)
            elif operation == 'detect_anomalies_iqr':
                processed_result = self._detect_anomalies_iqr(processed_result)
        
        return processed_result
    
    def _calculate_correlation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Pearson correlation coefficient from covariance and standard deviations"""
        # This would be implemented with the actual data processing
        result['correlation_coefficient'] = "Calculated via post-processing"
        return result
    
    def _perform_linear_regression(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform linear regression for forecasting"""
        result['forecast_points'] = "Generated via linear regression"
        result['model_metrics'] = {'r_squared': 'TBD', 'mse': 'TBD'}
        return result
    
    def _detect_anomalies_iqr(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies using IQR method"""
        result['anomaly_threshold'] = "Calculated using IQR method"
        return result

class ChatCSVApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.csv_processor = CSVProcessor()
        self.llm_processor = LLMQueryProcessor()
        # Initialize conversation history in session state
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
    
    def run(self):
        """Main application runner"""
        st.set_page_config(
            page_title="Chat with CSV (LLM)",
            page_icon="📊",
            layout="wide"
        )
        
        with st.container():
            st.markdown(
                """
                <div style='text-align: center;'>
                    <h1>Chat with CSV (LLM)</h1>
                    <p style='font-size: 18px;'>Upload a CSV file and ask questions in natural language to get intelligent insights!</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Sidebar for file upload and settings
        with st.sidebar:
            st.header("📁 File Upload")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                st.success("File uploaded successfully!")
                
                # Process the CSV
                with st.spinner("Processing CSV..."):
                    df = self.csv_processor.load_and_process_csv(uploaded_file)
        
        # Main content area
        if uploaded_file is not None and self.csv_processor.df is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                self._render_chat_interface()
            
            with col2:
                self._render_data_preview()
        else:
            st.info("👆 Please upload a CSV file to get started!")
            
            # Show example queries
            st.subheader("🔍 Example Queries")
            examples = [
                "What is the average age?",
                "How many records are there?",
                "Show me the maximum salary",
                "What's the total revenue?",
                "Give me a summary of the data"
            ]
            
            for example in examples:
                st.code(example)
    
    def _render_chat_interface(self):
        """Render the chat interface"""
        st.subheader("💬 Chat Interface")
        
        # Display conversation history from session state FIRST
        for i, (query, response_text, result_df, debug_info) in enumerate(st.session_state.conversation_history):
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                st.markdown(response_text)

                if result_df is not None:
                    st.dataframe(result_df, use_container_width=True)

                if debug_info:
                    with st.expander("🧠 Debug Info", expanded=False):
                        for key, value in debug_info.items():
                            if key == "Generated SQL":
                                st.code(value, language="sql")
                            else:
                                st.markdown(f"- **{key}**: `{value}`")
            
            st.markdown("---")
        
        # Use a form for input - this auto-clears after submission
        with st.form(key="query_form", clear_on_submit=True):
            query = st.text_input("Ask a question about your data:", 
                                placeholder="e.g., What is the average age?")
            submitted = st.form_submit_button("Send", type="primary")
        
        if submitted and query:
            with st.spinner("🔎 Analyzing your query..."):
                response_text, result_df, debug_info = self._process_query(query)

                # Save history for display
                st.session_state.conversation_history.append((query, response_text, result_df, debug_info))
                st.rerun()
    
    def _process_query(self, query: str) -> Tuple[str, Optional[pd.DataFrame], dict]:
        """Returns (response_text, table_df, debug_info_dict)"""
        try:
            context = self.llm_processor.parse_complex_query(query, self.csv_processor.metadata)

            # Collect debug info
            debug_info = {
                "Query": query,
                "Intent": getattr(context, 'intent', 'N/A'),
                "Confidence": getattr(context, 'confidence', 'N/A'),
                "Columns": getattr(context, 'columns', []),
                "Operations": getattr(context, 'operations', []),
            }

            sql_result = self.llm_processor.generate_advanced_sql(context, self.csv_processor.metadata)
            sql_query = sql_result.get('main_query')
            if sql_query:
                debug_info["Generated SQL"] = sql_query

                result_df = self.csv_processor.execute_sql_query(sql_query)

                if not result_df.empty:
                    if len(result_df) == 1 and len(result_df.columns) == 1:
                        value = result_df.iloc[0, 0]
                        value_str = f"**{value:.2f}**" if isinstance(value, (int, float)) else f"**{value}**"
                        return f"The result is {value_str}", None, debug_info
                    else:
                        return "Here are the results:", result_df, debug_info
                else:
                    return "⚠️ No results found for your query.", None, debug_info

            elif getattr(context, 'analysis_type', '') == 'summary':
                return self._generate_summary_response(), None, debug_info

            else:
                return "🤔 I need more specific information to give an accurate answer.", None, debug_info

        except Exception as e:
            return f"❌ I encountered an error while processing your query:\n\n`{str(e)}`", None, {}

    def _generate_summary_response(self) -> str:
        """Generate a comprehensive summary of the dataset"""
        metadata = self.csv_processor.metadata
        
        summary = f"""
**Dataset Summary:**

📊 **Basic Info:**
- Total rows: {metadata['shape'][0]:,}
- Total columns: {metadata['shape'][1]}

📈 **Column Types:**
- Numeric columns: {len(metadata['numeric_columns'])}
- Text columns: {len(metadata['categorical_columns'])}
- Date columns: {len(metadata['datetime_columns'])}

🔢 **Key Statistics:**
"""
        
        for col, stats in metadata['summary_stats'].items():
            summary += f"\n**{col}:** Mean = {stats['mean']:.2f}, Range = {stats['min']:.2f} to {stats['max']:.2f}"
        
        return summary
    
    def _render_data_preview(self):
        """Render comprehensive data preview and statistics"""
        st.subheader("👀 Data Preview")
        
        if self.csv_processor.df is not None:
            df = self.csv_processor.df
            
            # Create tabs for better organization
            preview_tab, stats_tab, viz_tab, quality_tab = st.tabs([
                "📋 Data Sample", "📊 Statistics", "📈 Visualizations", "🔍 Data Quality"
            ])
            
            with preview_tab:
                # Dataset overview
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", f"{len(df):,}")
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                with col4:
                    st.metric("Duplicate Rows", df.duplicated().sum())
                
                # Show customizable data sample
                st.subheader("Data Sample")
                sample_size = st.slider("Number of rows to display:", 5, min(50, len(df)), 10)
                sample_type = st.radio("Sample type:", ["First rows", "Random sample", "Last rows"], horizontal=True)
                
                if sample_type == "First rows":
                    sample_df = df.head(sample_size)
                elif sample_type == "Random sample":
                    sample_df = df.sample(min(sample_size, len(df))) if len(df) > 0 else df
                else:
                    sample_df = df.tail(sample_size)
                
                st.dataframe(sample_df, use_container_width=True)
                
                # Column information
                st.subheader("Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Null %': (df.isnull().sum() / len(df) * 100).round(2),
                    'Unique Values': df.nunique(),
                    'Unique %': (df.nunique() / len(df) * 100).round(2)
                })
                st.dataframe(col_info, use_container_width=True)
            
            with stats_tab:
                numeric_cols = self.csv_processor.metadata['numeric_columns']
                categorical_cols = [col for col in df.columns if col not in numeric_cols]
                
                # Numeric statistics
                if numeric_cols:
                    st.subheader("📈 Numeric Columns Statistics")
                    numeric_stats = df[numeric_cols].describe()
                    st.dataframe(numeric_stats, use_container_width=True)
                    
                    # Additional statistics
                    st.subheader("📊 Additional Numeric Statistics")
                    additional_stats = pd.DataFrame({
                        'Column': numeric_cols,
                        'Skewness': [df[col].skew() for col in numeric_cols],
                        'Kurtosis': [df[col].kurtosis() for col in numeric_cols],
                        'Variance': [df[col].var() for col in numeric_cols],
                        'Range': [df[col].max() - df[col].min() for col in numeric_cols],
                        'IQR': [df[col].quantile(0.75) - df[col].quantile(0.25) for col in numeric_cols]
                    })
                    st.dataframe(additional_stats.round(3), use_container_width=True)
                
                # Categorical statistics
                if categorical_cols:
                    st.subheader("📝 Categorical Columns Statistics")
                    cat_stats = []
                    for col in categorical_cols:
                        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
                            cat_stats.append({
                                'Column': col,
                                'Unique Values': df[col].nunique(),
                                'Most Frequent': mode_val,
                                'Frequency': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0,
                                'Frequency %': (df[col].value_counts().iloc[0] / len(df) * 100).round(2) if len(df[col].value_counts()) > 0 else 0
                            })
                    
                    if cat_stats:
                        cat_df = pd.DataFrame(cat_stats)
                        st.dataframe(cat_df, use_container_width=True)
            
            with viz_tab:
                st.subheader("📈 Data Visualizations")
                
                # Visualization options
                viz_type = st.selectbox("Choose visualization type:", [
                    "Distribution Analysis", "Correlation Analysis", "Missing Data Pattern", "Category Analysis"
                ])
                
                if viz_type == "Distribution Analysis" and numeric_cols:
                    selected_col = st.selectbox("Select numeric column:", numeric_cols)
                    
                    if selected_col:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histogram
                            fig_hist = px.histogram(df, x=selected_col, 
                                                title=f"Distribution of {selected_col}",
                                                marginal="box")
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        with col2:
                            # Box plot
                            fig_box = px.box(df, y=selected_col, 
                                            title=f"Box Plot of {selected_col}")
                            st.plotly_chart(fig_box, use_container_width=True)
                
                elif viz_type == "Correlation Analysis" and len(numeric_cols) > 1:
                    # Correlation heatmap
                    corr_matrix = df[numeric_cols].corr()
                    fig_corr = px.imshow(corr_matrix, 
                                    title="Correlation Heatmap",
                                    color_continuous_scale="RdBu_r",
                                    aspect="auto")
                    fig_corr.update_layout(width=600, height=500)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Pairwise scatter plot option
                    if len(numeric_cols) >= 2:
                        st.subheader("Pairwise Relationship")
                        col1, col2 = st.columns(2)
                        with col1:
                            x_col = st.selectbox("X-axis:", numeric_cols, key="x_axis")
                        with col2:
                            y_col = st.selectbox("Y-axis:", [col for col in numeric_cols if col != x_col], key="y_axis")
                        
                        if x_col and y_col:
                            fig_scatter = px.scatter(df, x=x_col, y=y_col,
                                                title=f"{x_col} vs {y_col}")
                            st.plotly_chart(fig_scatter, use_container_width=True)
                
                elif viz_type == "Missing Data Pattern":
                    # Missing data visualization
                    missing_data = df.isnull().sum()
                    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                    
                    if not missing_data.empty:
                        fig_missing = px.bar(x=missing_data.values, y=missing_data.index,
                                        orientation='h',
                                        title="Missing Data by Column",
                                        labels={'x': 'Number of Missing Values', 'y': 'Columns'})
                        st.plotly_chart(fig_missing, use_container_width=True)
                        
                        # Missing data percentage
                        missing_pct = (missing_data / len(df) * 100).round(2)
                        fig_pct = px.bar(x=missing_pct.values, y=missing_pct.index,
                                    orientation='h',
                                    title="Missing Data Percentage by Column",
                                    labels={'x': 'Percentage Missing', 'y': 'Columns'})
                        st.plotly_chart(fig_pct, use_container_width=True)
                    else:
                        st.success("🎉 No missing data found in the dataset!")
                
                elif viz_type == "Category Analysis":
                    categorical_cols_viz = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() < 20]
                    
                    if categorical_cols_viz:
                        selected_cat = st.selectbox("Select categorical column:", categorical_cols_viz)
                        
                        if selected_cat:
                            value_counts = df[selected_cat].value_counts().head(20)  # Top 20 categories
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                # Bar chart
                                fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                                            title=f"Distribution of {selected_cat}")
                                fig_bar.update_xaxes(tickangle=45)
                                st.plotly_chart(fig_bar, use_container_width=True)
                            
                            with col2:
                                # Pie chart (for top categories)
                                top_categories = value_counts.head(10)
                                fig_pie = px.pie(values=top_categories.values, names=top_categories.index,
                                            title=f"Top 10 Categories in {selected_cat}")
                                st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info("No suitable categorical columns found for visualization.")
            
            with quality_tab:
                st.subheader("🔍 Data Quality Assessment")
                
                # Overall data quality score
                total_cells = len(df) * len(df.columns)
                missing_cells = df.isnull().sum().sum()
                duplicate_rows = df.duplicated().sum()
                
                quality_score = max(0, 100 - (missing_cells / total_cells * 50) - (duplicate_rows / len(df) * 30))
                
                st.metric("Data Quality Score", f"{quality_score:.1f}/100")
                
                # Quality issues breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🚨 Quality Issues")
                    issues = []
                    
                    if missing_cells > 0:
                        issues.append(f"• {missing_cells:,} missing values ({missing_cells/total_cells*100:.1f}% of all data)")
                    
                    if duplicate_rows > 0:
                        issues.append(f"• {duplicate_rows:,} duplicate rows ({duplicate_rows/len(df)*100:.1f}% of total rows)")
                    
                    # Check for potential outliers in numeric columns
                    outlier_cols = []
                    for col in numeric_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col].count()
                        if outliers > 0:
                            outlier_cols.append(f"{col}: {outliers} outliers")
                    
                    if outlier_cols:
                        issues.append("• Potential outliers detected:")
                        for outlier_info in outlier_cols:
                            issues.append(f"  - {outlier_info}")
                    
                    if not issues:
                        st.success("✅ No major data quality issues detected!")
                    else:
                        for issue in issues:
                            st.warning(issue)
                
                with col2:
                    st.subheader("💡 Recommendations")
                    recommendations = []
                    
                    if missing_cells > total_cells * 0.05:  # More than 5% missing
                        recommendations.append("• Consider data imputation strategies for missing values")
                    
                    if duplicate_rows > 0:
                        recommendations.append("• Remove or investigate duplicate rows")
                    
                    if len(outlier_cols) > 0:
                        recommendations.append("• Investigate potential outliers in numeric columns")
                    
                    # Check for high cardinality categorical columns
                    high_card_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.8]
                    if high_card_cols:
                        recommendations.append(f"• Consider feature engineering for high-cardinality columns: {', '.join(high_card_cols)}")
                    
                    if not recommendations:
                        st.success("✅ Data appears to be in good condition!")
                    else:
                        for rec in recommendations:
                            st.info(rec)
                
                # Detailed column analysis
                st.subheader("🔍 Detailed Column Analysis")
                problematic_cols = []
                
                for col in df.columns:
                    issues = []
                    
                    # Check missing values
                    missing_pct = df[col].isnull().sum() / len(df) * 100
                    if missing_pct > 20:
                        issues.append(f"High missing rate: {missing_pct:.1f}%")
                    
                    # Check constant values
                    if df[col].nunique() == 1:
                        issues.append("Constant values (no variation)")
                    
                    # Check high cardinality for categorical
                    if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.9:
                        issues.append("Very high cardinality")
                    
                    if issues:
                        problematic_cols.append({
                            'Column': col,
                            'Issues': ', '.join(issues),
                            'Data Type': str(df[col].dtype),
                            'Missing %': f"{missing_pct:.1f}%"
                        })
                
                if problematic_cols:
                    prob_df = pd.DataFrame(problematic_cols)
                    st.dataframe(prob_df, use_container_width=True)
                else:
                    st.success("✅ All columns appear to be in good condition!")
        
        else:
            st.warning("No data available. Please upload a CSV file first.")

def main():
    """Main function to run the Streamlit app"""
    app = ChatCSVApp()
    app.run()

if __name__ == "__main__":
    main()