import pandas as pd
from backend.settings import DATA_FOLDER
from nltk.corpus import stopwords
from lemminflect import getAllLemmas
from backend.lib.contractions import contractions
from gensim.models import Word2Vec
import subprocess
import gzip
import argparse


def main(file, name):
    stop_words = set(stopwords.words('english'))
    df = pd.read_csv(file)
    df['tokenized'] = df['text'].str.split(' ')
    # replace contractions with full words
    df['tokenized'] = df['tokenized'].apply(
        lambda x: [contractions[word] if word in contractions else word for word in x])
    # remove stopwords
    df['tokenized'] = df['tokenized'].apply(lambda x: [word for word in x if word not in stop_words])
    # remove words with a length below 3 characters
    df['tokenized'] = df['tokenized'].apply(lambda x: [word for word in x if len(word) >= 3])
    # lemmatize words
    df['tokenzied'] = df['tokenized'].apply(lambda x: [y for y in [getLemma(word) for word in x] if y is not None])
    # remove words with a length below 3 characters (again, as for example 'doing' will now be 'do' after step 4)
    df['tokenized'] = df['tokenized'].apply(lambda x: [word for word in x if len(word) >= 3])
    # remove commas
    df['tokenized'] = df['tokenized'].apply(lambda x: [word.replace(',', '') for word in x])

    model = Word2Vec(df['tokenized'],
                     min_count=1,
                     workers=4,  # Number of processors (parallelisation)
                     window=5,  # Context window for words during training
                     iter=100)  # Number of epochs training over corpus
    # paths
    model_name = '{}_model'.format(name)
    model_gzip_path = '/'.join([DATA_FOLDER, ''.join([model_name, '.txt.gz'])])
    model_txt_path = ''.join([DATA_FOLDER, '/', model_name, '.txt'.format(name)])
    # spacy_model_path = ''.join([DATA_FOLDER, '/spacy.{}.model'.format(name)])

    # save the model as a txt file
    model.wv.save_word2vec_format(model_txt_path, binary=False)

    # gzip the txt file
    f_in = open(model_txt_path, 'rb')
    f_out = gzip.open(model_gzip_path, 'w')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()

    # init the spacy model
    # subprocess.run(['python', '-m', 'spacy', 'init-model', 'en', spacy_model_path, '--vectors-loc', model_gzip_path])
    return


def getLemma(word):
    lemma = None
    try:
        lemma = getAllLemmas(word)[0]
    except KeyError:
        pass
    return lemma


def setup():
    parser = argparse.ArgumentParser(description='Convert a csv file of sentences into a word vector model')
    parser.add_argument('file', type=str, help='CSV file to be converted into word vector')
    parser.add_argument('name', type=str, help='Name of model')
    args = parser.parse_args()
    main(file=args.file, name=args.name)


if __name__ == "__main__":
    setup()
