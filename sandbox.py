from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter

def save_transcript_with_timestamps(youtube_url: str, output_file: str):
    video_id = youtube_url.split("watch?v=")[-1]
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Check if there is an English transcript
        transcript = transcript_list.find_transcript(['en'])

        # Fetch the actual transcript data
        transcript_data = transcript.fetch()

        # Formatting the transcript data as SRT
        formatter = SRTFormatter()
        srt_transcript = formatter.format_transcript(transcript_data)

        # Save the transcript with timestamps to a file
        with open(output_file, 'w') as file:
            file.write(srt_transcript)

        print(f"Transcript with timestamps saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
youtube_url = "https://www.youtube.com/watch?v=Gjnup-PuquQ&t=5s"
output_file = "transcript_with_timestamps.txt"
save_transcript_with_timestamps(youtube_url, output_file)
