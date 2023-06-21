# Natural-Language-Processing

## Acknowledgements
The notes and code samples (some of them) have been referred from **Practical Natural Language Processing** by **Sowmya Vajjala**, **Bodhisattwa Majumder**, **Anuj Gupta**, and **Harshit Surana**.

## Table of Contents
1. <a href = "https://github.com/naik24/Natural-Language-Processing/edit/master/README.md#1-introduction">Introduction

## 1. Introduction
Natural Language Processing (NLP) is an area of computer science that deals with methods to analyze, model, and understand human language. 

**Core Tasks**: Text Classification, Information Extraction, Conversational Agent, Information Retrieval, Question Answering Systems

**General Applications**: Spam Classification, Calendar Event Extraction, Personal Assistants, Search Engines, Jeopardy

**Industry Specific**: Social Media Analysis, Retail Catalog Extraction, Health Records Analysis, Financial Analysis, Legal Entity Extraction

**1.1 NLP Tasks**

- Language Modelling: Task of predicting what the next word in a sentence will be based on the history of previous words. Goal of this task is to learn the probability of a sequence of words appearing in a given language. 
- Text Classification: Task of bucketing the text into a known set of categories based on its content. 
- Information Extraction: Task of extracting relevant information from text, such as calendar events from emails or the names of people mentioned in a social media post
- Information Retrieval: Task of finding documents relevant to a user query from a large collection.
- Conversational Agent: Task of building dialogue systems that can converse in human languages.
- Text Summarization: Task of creating short summaries of longer documents while retaining the core content and preserving the overall meaning of the text
- Question Answering: Task of building a system that can automatically answer questions posed in natural language
- Machine Translation: Task of converting a piece of text from one language to another.
- Topic Modelling: Task of uncovering the topical structure of a large collection of documents.

**1.2 Linguistics**

Human language is composed of four major building blocks: phonemes, morphemes and lexemes, syntax, and context. NLP applications need knowledge of different levels of these building blocks, starting from the basic sounds of language i.e. phonemes to texts with some meaningful expressions i.e. context.

- **Phonemes**: Smallest units of sound in a language. Standard English has 44 phonemes, which are either single letters or a combination of letters. 
- **Morphemes and Lexemes**: Smallest unit of language that has a meaning. It is formed by a combination of phonemes. Not all morphemes are words, but all prefixes and suffixes are morphemes. Lexemes are the structural variations of morphemes related to one another by meaning. For example, "sit" and "sitting" belong to the same lexeme form. Morphological analysis, which analyzes the structure of words by studying its morphemes and lexemes, is a foundational block for many NLP tasks, such as tokenization, stemming, learning word embeddings, and POS tagging.
- **Syntax**: Set of rules to construct grammatically correct sentences out of words and phrases in a language. A common approach to representing syntactic structure in linguistics is a <a href = "https://en.wikipedia.org/wiki/Parse_tree"> parse tree </a>. A parse tree has a hierarchical structure of language, with words at the lowest level, followed by POS tags, followed by phrases, and ending with a sentence at the highest level. NOTE: Syntax of one language can be very different from that of another language, and the language-processing approaches needed for that language will change accordingly.
- **Context**: Context is coming together of various parts in a language to convey a particular meaning. Context includes long-term references, world knowledge, and a common sense along with the literal meaning of words and phrases. Context is composed from semantics and pragmatics. Semantics is the direct meaning of the words and sentences without external context. Pragmatics adds world knowledge and external context of the conversation to enable us to infer implied meaning. 
