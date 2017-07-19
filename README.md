### Harry Potter and the Sentiment Scores  

A Python-based pipeline to track how sentiments vary in the Harry Potter book series as character interactions and plots change. The pipeline includes natural language processing, topic modeling, unsupervised clustering and time series analysis techniques.  

**Base Directory:** `HarryPotter/`    

**Sub-directories:**

* `Book_Scripts/`
The raw (unicode) texts from the Harry Potter books.

*  `NLP/SupportModules/`
Contains helper functions and classes.
  * `data_transform_module.py`: functions for un/pickle-ing
   files.  
  * `contraction_expressions.ipynb`: creates a dictionary to map common contraction expressions to their expansions such as I'll to I will.  
  * `book_module.py`: helper functions and classes for breaking book chapter texts into user-specified text segments, determining sentiment scores of each segment and plotting the sentiment scores.
  * `theme_module.py`: helper functions and classes for finding segments in the book chapter contained certain user-specified text-strings. sentiment scores of the segments are plotted sequentially. applies time series smoothing (rolling average) to the plot.


* `NLP/`  
Codes for sentiment analysis, topic modeling, etc.:
 * `hp_book_chapters.ipynb`: reads text files for each book
cleans and formats each book - expand contraction words according to specification of "contractions_dictionary.p"
splits books in chapters
saves each book as a dictionary (key=chapter_number, value = chapter as a string)
 * `book_sentiment_analysis.ipynb`: Loads Harry Potter book texts. Cleans texts for NLP processing (e.g. remove punctuations, expand contractions such as I'll to I will). Breaks book texts in to segments (defalt: 400 words). Compute sentiment scores (polarity and subjectivity) of the segments. Polarity and subjectivity for each book are plotted separately. Different colors in the plot represents different chapter. The dots represent segments.
NOTE: the supporting functions and classes are at ~/NLP/SupportModules/
 * `book_sentiment_interactive_plots.ipynb`: makes interactive polarity plot. Lets a user place cursor on a dot. The chapter and segment number pops-up. The user can enter the chapter and segment number in the cell below the plot to view the text segment.  
 * `theme_sentiment_analysis.ipynb`: This code takes a list of keywords and text segment size as inputs and extracts text segments containing those keywords (theme). Sentiment scores are computed for the extracted segments.
Plots sentiment variation of the theme (keyword presence) across the books. A (time series) smoothing average is applied.
  * `book_topic_modeling.ipynb`: Finds key topics from all documents extracted from the books. Documents are text segments. Segment size is defined by users.
  Using topic modeling techniques such as LDA, LSI, key topics are extracted from the documents. Text segments/documents are clustered (using k-means) into topics.  
