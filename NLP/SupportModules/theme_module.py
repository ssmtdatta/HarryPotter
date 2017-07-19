from textblob import TextBlob
from nltk.tokenize import word_tokenize



class Theme():  
    """
    input: book texts in dictionary format and 
           keywords as list e.g. ["harry", 'dumbledore']
    output: text segments containing key words

    """
    def __init__(self, book, chunk_size, match_words_list):
        
        self.book = book
        self.chunk_size = chunk_size
        self.match_words_list = match_words_list
        self.theme_info = self.get_chapterChunkTheme()
  
 
    def get_themeMatch(self, chunk_tokens):
        is_present = 0
        for i in range(0, len(self.match_words_list)):
            if self.match_words_list[i] in chunk_tokens:
                is_present = is_present + 1
        if len(self.match_words_list) == is_present:
            return True
        else:
            return False
                             
    def get_chapterChunkTheme(self):    
        book_matched_text = []
        
        for key in self.book:
            
            chapter = self.book[key].lower()
            chapter_tokens = word_tokenize(chapter) # decide save or not save chapter tokens
            
            if len(chapter_tokens) >= self.chunk_size:
                
                chap_dict = {} # create a dictionary of a chapter is gt length
                matched_text = []
                matched_sentiment = []
                
                # define chunk indices
                start_ind = list(range(0,len(chapter_tokens), self.chunk_size))
                end_ind = start_ind[1:] + [len(chapter_tokens)]
                
                for ind in range(0, len(start_ind)):                    
                    chunk_tokens = chapter_tokens[start_ind[ind]:end_ind[ind]]
                    if self.get_themeMatch(chunk_tokens):
                        chunk = ' '.join(x for x in chunk_tokens)
                        pol = TextBlob(chunk).sentiment.polarity
                        sub = TextBlob(chunk).sentiment.subjectivity
                        matched_text.append(chunk)
                        matched_sentiment.append((pol, sub))
                
                chap_dict['text'] = matched_text
                chap_dict['sentiment'] = matched_sentiment
            
            book_matched_text.append(chap_dict)         
                        
        return book_matched_text


def plotTheme_timeSeries(df_books):
    x_vals_start = []
    book_seq = df_books['book'].tolist()
    for s in range(1, 8):
        ind = book_seq.index(s)
        x_vals_start.append(ind)

    x_vals_end = x_vals_start[1:] + [len(book_seq)]
    
    cs  = cm.Dark2(np.arange(7)/7)
    fig, axs = plt.subplots(figsize=(18,6))
    axs.plot(range(df_books.shape[0]), df_books['rolling'], linewidth = 3)
    for i in range(len(x_vals_end)):
        axs.axvspan(x_vals_start[i], x_vals_end[i], color=cs[i], alpha=0.15, lw=0)
    axs.tick_params(axis='x',          # changes apply to the x-axis
                           which='both',      # both major and minor ticks are affected
                           bottom='off',      # ticks along the bottom edge are off
                           top='off',
                           labelbottom='off') # labels along the bottom edge are off
    plt.show()
    

def prepData_timeSeries(theme_keywords):

    df_books = pd.DataFrame(columns=['book', 'score', "rolling"])

    for seq in range(1, 8):
        book_dict = dtmod.unpickleSomething(BOOK_PATH, "book{}_chapters.p".format(seq))
        book = Theme(book_dict, 50, theme_keywords)
        theme_info = book.theme_info

        df = pd.DataFrame(columns=['book', 'score', "rolling"])
        chunk_sentiment = []
        for chunk_dict in theme_info:
            if not chunk_dict['sentiment']:
                chunk_sentiment.append(0)
            else:
                chunk_sentiment.append(chunk_dict['sentiment'][0][0])

        df['book'] = [seq]*len(chunk_sentiment)
        df['score'] = chunk_sentiment
        df['rolling'] = df['score'].rolling(window=7,min_periods=1,center=True).mean() 

        df_books = pd.concat([df_books, df], axis=0)
        
    return df_books


def prepData_plotSentiment(theme_info):
    """
    returns a list of lists
    outer list = for a book
    each inner list is for a chapter
    entries in a chapter-list are chunks
    """
    theme_polarity = []
    theme_subjectivity = []
    theme_text = []
    for chap_dict in theme_info:
        
        sentiment = chap_dict['sentiment']
        pol = [x[0] for x in sentiment]
        sub = [x[1] for x in sentiment]
        if pol:
            theme_polarity.append(pol)
        if sub:
            theme_subjectivity.append(sub)
        
        text = chap_dict['text']
        if text:
            theme_text.append(text)
        
    return theme_polarity, theme_subjectivity, theme_text



    
def plotSentiment(pol, sub, text, seq):

    # define x-axis values for plotting 
    x_vals = []
    init = 0
    for inner_list in pol:
        start_at = init
        end_at = start_at + len(inner_list)
        vals = list(range(start_at, end_at))
        x_vals.append(vals)
        init = init + len(inner_list)
        
        
    # define color space
    cs  = cm.Set1(np.arange(9)/9)
    while cs.shape[0] <= len(x_vals):
        cs = np.concatenate((cs, cs), axis=0)
    cs = cs[0: len(x_vals)]

    
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