from textblob import TextBlob
from nltk.tokenize import word_tokenize

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import mpld3

class Book():  
    """
    segments book chapter texts into segments of n words.

    input: one book text as dictionary.
    each dictionary value is a chapter text in string format.
    
    output: number of chapter, 
    		text segments of chapters, 
    		text segment index 
    
    """
    def __init__(self, book, chunk_size):
        
        self.book = book
        self.chunk_size = chunk_size
        self.n_chaps = self.get_numChapters()
        self.chunk_info = self.get_chapterChunkInfo()
  
    def get_numChapters(self):
        n_chapters = len(self.book)
        return n_chapters

    def get_topics(self):
        pass
     
    def get_chapterChunkInfo(self):
        
        book_chap_chunk_info = []
        
        for key in self.book:
            chapter = self.book[key].lower()
            chapter_tokens = word_tokenize(chapter) # decide save or not save chapter tokens
            
            if len(chapter_tokens) >= self.chunk_size: 
                chapter_dict = {} 
                start_ind = list(range(0,len(chapter_tokens), self.chunk_size))
                end_ind = start_ind[1:] + [len(chapter_tokens)]
                chapter_dict["n_chunks"] = len(start_ind) # number of chunks per chapter

                chapter_chunk_index = []  # tuples (start, end) index for each chunk in a chapter as 
                chapter_chunk_text = []
                chapter_chunk_sentiment = [] # tuples(polarity, subjectivity) for each chunk in a chapter            
                for ind in range(0, len(start_ind)):
                    chunk = ' '.join(x for x in chapter_tokens[start_ind[ind]:end_ind[ind]])
                    pol = TextBlob(chunk).sentiment.polarity
                    sub = TextBlob(chunk).sentiment.subjectivity
                    chapter_chunk_index.append((start_ind[ind], end_ind[ind]))
                    chapter_chunk_text.append(chunk)
                    chapter_chunk_sentiment.append((pol, sub))
                    
                chapter_dict["chunk_index"] = chapter_chunk_index
                chapter_dict["chunk_text"] = chapter_chunk_text
                chapter_dict["chunk_sentiment"] = chapter_chunk_sentiment
                
            book_chap_chunk_info.append(chapter_dict)
        return book_chap_chunk_info






def prepData_plotSentiment(book_info):
    """
    convert data from book class into lists 
    for plotting
    """ 
    idx0 = 0  
    book_polarity = []
    book_subjectivity = []
    book_chunk_seq = []
    for chap_dict in book_info:
        n_chunks = chap_dict['n_chunks'] 
        sentiment = chap_dict['chunk_sentiment']
        pol = [x[0] for x in sentiment]
        sub = [x[1] for x in sentiment]
        chunk_seq = list(range(idx0, idx0+n_chunks))
        idx0 = idx0+n_chunks

        book_polarity.append(pol)
        book_subjectivity.append(sub)
        book_chunk_seq.append(chunk_seq)
    return book_polarity, book_subjectivity, book_chunk_seq





def plotSentiment(pol, sub, x_vals, seq):
    """
    plot polarity and subjectivity 
    """
    
    # define color space
    cs  = cm.Set1(np.arange(9)/9)
    while cs.shape[0] <= len(x_vals):
        cs = np.concatenate((cs, cs), axis=0)
    cs = cs[0: len(x_vals)]
    
    # define fig size and windows
    fig, axs = plt.subplots(2,1,figsize=(16,12))
      
    # polarity
    for i in range(0, len(x_vals)):
        axs[0].plot(x_vals[i], pol[i], color=cs[i])
        axs[0].scatter(x_vals[i], pol[i], color=cs[i])
        axs[0].axvspan(min(x_vals[i]), max(x_vals[i]), color=cs[i], alpha=0.1, lw=0)
    #axs[0].set_ylim(-1, 1)
    axs[0].set_title('Polarity of Book {}'.format(seq), fontsize=20, fontname="Times New Roman")
    axs[0].set_ylabel('Polarity', fontsize=16, fontname="Times New Roman")
    axs[0].set_xlabel('Chapters', fontsize=16, fontname="Times New Roman")
    axs[0].tick_params(axis='x',          # changes apply to the x-axis
                       which='both',      # both major and minor ticks are affected
                       bottom='off',      # ticks along the bottom edge are off
                       top='off',
                       labelbottom='off') # labels along the bottom edge are off

    # subjectivity
    for i in range(0, len(x_vals)):
        axs[1].plot(x_vals[i], sub[i], color=cs[i])
        axs[1].scatter(x_vals[i], sub[i], color=cs[i])
        axs[1].axvspan(min(x_vals[i]), max(x_vals[i]), color=cs[i], alpha=0.1, lw=0)
    #axs[1].set_ylim(0, 1)
    axs[1].set_title('Subjectivity of Book {}'.format(seq), fontsize=20, fontname="Times New Roman")
    axs[1].set_ylabel('Subjectivity', fontsize=16, fontname="Times New Roman")
    axs[1].set_xlabel('Chapters', fontsize=16, fontname="Times New Roman")
    axs[1].tick_params(axis='x',          # changes apply to the x-axis
                       which='both',      # both major and minor ticks are affected
                       bottom='off',      # ticks along the bottom edge are off
                       top='off',
                       labelbottom='off') # labels along the bottom edge are off 
    plt.show()   



def interactive_plotPolarity(pol, sub, x_vals, seq):    
    
    # define color space
    cs  = cm.Set1(np.arange(9)/9)
    while cs.shape[0] <= len(x_vals):
        cs = np.concatenate((cs, cs), axis=0)
    cs = cs[0: len(x_vals)]
    
    fig, axs = plt.subplots(figsize=(14,6))
    

    # polarity
    for i in range(0, len(x_vals)):
        # create a function to get a list of pop-up information for each segment
        pop_up_vals = list(range(len(x_vals[i])))
        pop_up_chunk_id = list(range(len(x_vals[i])))
        pop_up_chap_number = ["Ch .{}".format(i+1)]*len(x_vals[i])
        for t in range(len(pop_up_chunk_id)):
            id_val = pop_up_chunk_id[t]+1
            pop_up_chap_number[t] = pop_up_chap_number[t]+', Seg .{}'.format(id_val)
            
        lines = axs.plot(x_vals[i], pol[i], color=cs[i], marker='s')
        mpld3.plugins.connect(fig, 
                              mpld3.plugins.PointLabelTooltip(lines[0],
                                                              labels=pop_up_chap_number) ) 
    axs.set_title('Polarity of Book {}'.format(seq), fontsize=20, fontname="Times New Roman")
    axs.set_ylabel('Polarity', fontsize=16, fontname="Times New Roman")
    axs.set_xlabel('Chapters', fontsize=16, fontname="Times New Roman")
    axs.set_xticks([])
    
    mpld3.enable_notebook()