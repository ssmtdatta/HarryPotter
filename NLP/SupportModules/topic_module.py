class Document():  
    """
    input is one book as dictionary.
    each list in the dictionary is a chapter.
    """
    def __init__(self, book, chunk_size):
        self.book = book
        self.chunk_size = chunk_size
        self.docs = self.get_bookDocuments()
    
    def get_bookDocuments(self):
        book_documents = []
        for key in self.book:
            chapter = self.book[key].lower()
            chapter_tokens = word_tokenize(chapter)         
            
            if len(chapter_tokens) >= self.chunk_size: 
                start_ind = list(range(0,len(chapter_tokens), self.chunk_size))
                end_ind = start_ind[1:] + [len(chapter_tokens)]
                
                for ind in range(0, len(start_ind)):
                    tokenized_chunk = chapter_tokens[start_ind[ind]:end_ind[ind]]
                    chunk = ' '.join(x for x in tokenized_chunk)
                    book_documents.append(chunk)
        return(book_documents)





def get_LDA_topicModel(all_documents):    
    # Create a CountVectorizer 
    count_vectorizer = CountVectorizer(ngram_range=(1, 2),  stop_words='english')
    # Fit it on the document data (trining?)
    count_vectorizer.fit(all_documents)
    # Create the term-document matrix (Transpose places terms on the rows)
    counts = count_vectorizer.transform(all_documents).transpose()
    # Convert sparse matrix of counts to a gensim corpus
    corpus = matutils.Sparse2Corpus(counts)
    # Map matrix rows to words (tokens)
    id2word = dict((v, k) for k, v in count_vectorizer.vocabulary_.items())
    
    # Create lda model
    lda = models.LdaModel(corpus=corpus, num_topics=10, id2word=id2word, passes=10)
    #lda topics
    lda_print_topics = lda.print_topics()
    # Transform the docs from the word space to the topic space
    lda_corpus = lda[corpus]
    
    return lda_corpus, lda_print_topics




def get_LSI_topicModel(all_documents):
    
    # Create a CountVectorizer 
    count_vectorizer = CountVectorizer(ngram_range=(1, 2),  stop_words='english')
    # Fit it on the document data (trining?)
    count_vectorizer.fit(all_documents)
    
    # Create the term-document matrix (Transpose places terms on the rows)
    counts = count_vectorizer.transform(all_documents).transpose()
    print("shape of counts", counts.shape)
    # Convert sparse matrix of counts to a gensim corpus
    corpus = matutils.Sparse2Corpus(counts)
    # Map matrix rows to words (tokens)
    id2word = dict((v, k) for k, v in count_vectorizer.vocabulary_.items())
    
    # Create lda model
    lsi = models.LsiModel(corpus=corpus, id2word=id2word, num_topics=50)
    #lda topics
    lsi_print_topics = lsi.print_topics()
    length = len(lsi_print_topics)
    # Transform the docs from the word space to the topic space
    lsi_corpus = lsi[corpus]
    
    return lsi_corpus, lsi_print_topics, length



def convert_LSIdocs2matrix(lsi_docs):
    topic_coord_array = []
    for doc in lsi_docs:
        doc_coords = []
        for tup in doc:
            doc_coords.append(tup[1])
        topic_coord_array.append(doc_coords)
    
    return np.array(topic_coord_array)

