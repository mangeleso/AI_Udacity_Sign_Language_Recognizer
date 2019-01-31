import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        #from office-hours:
        # d = features 
        # n = HMM states
        # free parameters = n**2 + 2*d*n-1

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        bestModel = None
        bestBIC = float("inf")
        best_num_components = self.min_n_components

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n)
                logL = hmm_model.score(self.X, self.lengths)
                free_params = n**2 + 2 * self.X.shape[1] * n - 1
                BIC = -2 * logL + free_params * np.log(sum(self.lengths))
                if BIC < bestBIC:
                    bestModel = hmm_model
                    bestBIC = BIC
            except:
                pass

        return bestModel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        bestModel = None
        bestDIC = float("-inf")

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n)
                this_word_logL = hmm_model.score(self.X, self.lengths)

                sum_all_other_logL = 0
                for all_other_word in self.words:
                    if all_other_word != self.this_word:
                        wordX, wordLengths = self.hwords[all_other_word]
                        sum_all_other_logL += hmm_model.score(wordX, wordLengths)
                    
                DIC = this_word_logL - (sum_all_other_logL / (len(self.words) - 1))
                if DIC > bestDIC:
                    bestDIC = DIC
                    bestModel = hmm_model
            
            except:
                pass

        return bestModel


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        max_logL = float("-inf")
        n_splits = min(len(self.lengths), 3)
        best_num_components = self.min_n_components


        #print("N OF FOLDS>", n_splits)

        for n in range(self.min_n_components, self.max_n_components + 1):

          
            try:
                total_logL = 0

                if n_splits > 1:
                    split_method = KFold(n_splits)
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                        # Create train and test data
                        X_train , X_lenghts_train = asarray(combine_sequences(cv_train_idx, self.sequences))
                        X_test  , X_lenghts_test  = asarray(combine_sequences(cv_test_idx, self.sequences))

                        #Train HMM
                        hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, X_lenghts_train)

                        #Get LogL on the test data
                        logL_test = hmm_model.score(X_test, X_lenghts_test)
                        total_logL += logL_test

                    total_logL = total_logL / n_splits

                #pass kfolds
                else:
                    hmm_model = self.base_model(n)
                    total_logL = hmm_model.score(self.X, self.lengths)

                if total_logL > max_logL:
                        max_logL = total_logL
                        best_num_components = n 

            except: 
                pass


        return GaussianHMM(n_components=best_num_components, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)


        

