import langchain_helper as lch
import os

# YouTube video URL
youtube_url = "https://www.youtube.com/watch?v=Gjnup-PuquQ&t=5s"
query="Can you summarize for me this transcripte?"
# Fetch and prepare the transcript for LLM
db, formatted_transcript = lch.fetch_and_prepare_transcript_for_llm(youtube_url)

response, x = lch.get_response_from_query(db, query, k=4)

print(response)
