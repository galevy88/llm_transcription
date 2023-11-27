from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from dotenv import load_dotenv
from pprint import pprint

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain



load_dotenv()

embeddings = OpenAIEmbeddings()

def fetch_and_prepare_transcript_for_llm(youtube_url: str):
    video_id = youtube_url.split("watch?v=")[-1]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # Check if there is an English transcript
    transcript = transcript_list.find_transcript(['en'])

    # Fetch the actual transcript data
    transcript_data = transcript.fetch()

    # Formatting the transcript data as SRT
    formatter = SRTFormatter()
    srt_transcript = formatter.format_transcript(transcript_data)

    # Process the SRT transcript into Document objects
    processed_documents = []
    for entry in transcript_data:
        text = entry['text']
        start_time = entry['start']
        doc = Document(page_content=text, metadata={'timestamp': start_time})
        processed_documents.append(doc)

    # Use the text splitter to split the processed transcript
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(processed_documents)

    # Create a FAISS index from the processed documents
    db = FAISS.from_documents(docs, embeddings)
    pprint(docs)
    return db, srt_transcript


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    pprint(docs)
    # Constructing a detailed description of each document
    docs_description = "\n\n".join([f"Document Segment: '{d.page_content}' (Timestamp: {d.metadata['timestamp']} seconds)" for d in docs])

    llm = OpenAI(model_name="gpt-3.5-turbo-0301")
    prompt = PromptTemplate(
        input_variables=["question", "docs_description"],
        template=f"""
        As a highly capable AI language model, you are presented with a task involving the analysis of video transcript segments. Each segment is part of a larger transcript from a video, and it comes with a specific timestamp indicating when in the video the segment occurs.

        Here are the details of the documents you will analyze:
        {docs_description}

        Your task is two-fold:

        1. **Summarization**: Review the content of these documents thoroughly. Understand the context and the subject matter being discussed in each segment. Then, create a concise summary that captures the key points, themes, and important information from the entire set of documents. This summary should reflect the essence of the video's content.

        2. **Timestamp Identification**: After creating the summary, your next task is to identify the most critical points or moments in the summary. For each of these key points or moments, you need to specify the corresponding timestamp from the original video. These timestamps are crucial as they will be used to extract images from the video that represent these key moments.

        The timestamps should be listed at the end of your response in a structured format for easy extraction and processing. Use the following format:
        ###TIMESTAMPS###

        "timestamp1": "value_timestamp1",
        "timestamp2": "value_timestamp2",
        "timestamp3": "value_timestamp3",
        ...
        

        Please proceed with the summarization first, followed by the timestamp identification based on the summary. Keep in mind the clarity and accuracy of both the summary and the timestamps are of utmost importance.


        Begin your summarization and timestamp identification here:
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs_description=docs_description)
    response = response.replace("\n", " ")
    return response, docs
