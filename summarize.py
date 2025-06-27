from transformers import pipeline

def summarize_text(input_text, max_length=130, min_length=30):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    article = """
    Natural language processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and human language.
    It involves developing algorithms and models that enable computers to understand, interpret, and generate human language in a way that is meaningful.
    NLP has a wide range of applications, including machine translation, sentiment analysis, text summarization, question answering, and chatbots.
    With the advent of large language models like BERT, GPT, and BART, the accuracy and capabilities of NLP systems have significantly improved.
    Summarization is one of the key tasks in NLP, where the goal is to produce a concise and coherent version of a longer text while preserving its meaning.
    """
    print("Original Article:")
    print(article)

    summary = summarize_text(article)
    print("\nSummary:")
    print(summary)
