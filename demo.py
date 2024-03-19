import gradio as gr
import nltk
from recommend import get_top_headlines, rank_headlines

stopwords = nltk.corpus.stopwords.words('english')
top_headlines = get_top_headlines()


def rank(bio):
    """
    Wrapper function for ranking the top headlines

    PARAMETERS:
        - bio (str): user bio to base rankings off of

    RETURNS:
        - df_rank (polars.DataFrame): DataFrame with headlines in the
            'headlines' column and ranking in the 'rank' column
    """
    return rank_headlines(bio, top_headlines)
    
    
if __name__ == '__main__':
    demo = gr.Interface(
        fn=rank,
        inputs=[gr.Textbox(label='Provide a bio describing your interests')],
        outputs=[gr.Dataframe(label='Recommended Hacker News Articles')]
    )

    demo.launch()