class WordVec:
    def __init__(self, args):
        print('processing corpus')
        if args.restore is None:
            sentences =  read_data(args.corpus)
            print('training')
            self.wvec_model = Word2Vec(sentences=sentences, size=args.dimension, window=args.window,
                                       workers=args.workers,
                                       sg=args.sg,
                                       batch_words=args.batch_size, min_count=1, max_vocab_size=args.vocab_size)
            self.wvec_model.save('wordvec_model_train_' + str(args.dimension) + '.pkl')
        else:
            #self.wvec_model = KeyedVectors.load_word2vec_format(args.restore, binary=True)
            print('loading model')
            self.wvec_model = Word2Vec.load(args.restore)
        self.rand_model = RandomVec(args.dimension)

    def __getitem__(self, word):
        #word = word.lower()
        try:
            return self.wvec_model[word]
        except KeyError:
            #print("Don't found!")
            return self.rand_model[word]
