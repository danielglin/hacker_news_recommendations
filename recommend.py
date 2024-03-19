from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import nltk
import polars as pl
import requests


def preprocess_bio(bio, stopwords):
    """
    Pre-processes a bio by POS-tagging, removing stopwords, and extracting just the nouns
    
    PARAMETERS:
        - bio (str): string to POS-tag, remove stopwords from, and extract
            nouns from
        - stopwords (list of str): stopwords to remove from bio
    
    RETURNS:
        - s_nouns (list of str): list of non-stopword nouns in bio
    """
    NOUN_POS_TAGS = ('NN', 'NNS', 'NNP', 'NNPS')

    tagged_bio = pos_tag(word_tokenize(bio.lower()))

    # keep only the nouns
    s_nouns = [t.lower() for (t, pos) in tagged_bio if pos in NOUN_POS_TAGS]
    s_nouns = [t for t in s_nouns if t not in stopwords]
    return s_nouns


def _preprocess_headline(headline):
    """
    Pre-processes a headline by lower-casing it and tokenizing it
    
    PARAMETERS:
        - headline (str): the headline to pre-process
        
    RETURNS:
        - l_headline_tokens (list of str): list of lower-cased
            tokens in headline
    """
    headline_lower = headline.lower()
    l_headline_tokens = word_tokenize(headline_lower)
    return l_headline_tokens
    
    
def count_overlap(l_bio_nouns, headline):
    """
    Counts the number of nouns in common between the list of nouns from 
    a bio and a headline
    
    PARAMETERS:
        - l_bio_nouns (list of str): list of the nouns in the bio
        - headline (str): the headline to pre-process
        
    RETURNS:
        - num_overlap (int): how many nouns are in both, ignoring repeated nouns
    """
    l_headline_tokens = _preprocess_headline(headline)
    s_headline_tokens = set(l_headline_tokens)
    s_bio_nouns = set(l_bio_nouns)
    overlapping_tokens = s_headline_tokens.intersection(s_bio_nouns)
    return len(overlapping_tokens)  


def get_top_headlines():
    """
    Returns the headlines of the top 500 articles on Hacker News
    
    PARAMETERS:
        - None
        
    RETURNS:
        - top_headlines (list of str): headlines of the top 500 articles
    """
    TOP_STORIES_URL = 'https://hacker-news.firebaseio.com/v0/topstories.json'
    
    # pulling top 500 stories
    top_stories = requests.get(TOP_STORIES_URL)

    # go through the 500 top stories' ids to pull the headlines
    top_headlines = []

    for item_id in top_stories.json():
        story_req = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{item_id}.json')
        top_headlines.append(story_req.json()['title'])

    return top_headlines


def rank_headlines(bio, headlines):
    """
    Ranks headlines for a bio
    
    PARAMETERS:
        - bio (str): user bio to base rankings off of
        - headlines (list of str): headlines to rank
        
    RETURNS:
        - df_rank (polars.DataFrame): DataFrame with headlines in the
            'headlines' column and ranking in the 'rank' column
    """
    df_rank = pl.DataFrame({
        'headlines': headlines
    })
    
    # pre-process the bio
    stopwords = nltk.corpus.stopwords.words('english')
    l_bio_nouns = preprocess_bio(bio, stopwords)

    # calculate each headlines' score
    df_rank = df_rank.with_columns(
        pl.col('headlines').map_elements(lambda h: count_overlap(l_bio_nouns, h)).alias('scores')
    )    
    
    # rank based off the score
    df_rank = df_rank.with_columns(
        pl.col('scores').rank(method='min', descending=True).alias('rank')
    )
    df_rank = df_rank.drop('scores')
    df_rank = df_rank.sort(by='rank')
    return df_rank    