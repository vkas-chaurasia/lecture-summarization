import os
import time
import json
from fpdf import FPDF
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import PydanticOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel
from typing import List, Dict

# --- Models ---
class TopicExtraction(BaseModel):
    topics: List[str]

class TopicSummary(BaseModel):
    topic: str
    summary: str

# --- PDF Class ---
class HTMLPDF(FPDF):
    def __init__(self, title=None):
        super().__init__()
        self.custom_title = title

    def header(self):
        if self.custom_title:
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(0, 0, 255)  # Blue color
            self.set_y(10)
            self.set_x(10)
            self.cell(0, 10, self.custom_title, align="C")
            self.ln(10)
            self.set_text_color(0, 0, 0)  # Reset to black
        else:
            self.set_y(15)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

# --- Prompts ---
def generate_topic_structured_prompt(chunk_text, parser):
    return f"""
You are a transcript analyst.

Below is a chunk of a lecture or conversation. Your task is to identify logical transitions and divide the text into **sections** based on:

- Topic changes
- Speaker changes
- Q&A transitions
- Clear shifts in focus or structure

For **each section**, do the following:
- Provide a short **heading or title** (1 sentence max)
- Do NOT include summaries or explanations — only section titles
- **Ignore** irrelevant or non-substantive content such as:
 - Breaks (e.g., "Short break", "Let's resume")
 - Technical issues (e.g., "Mic not working")
 - Administrative notes (e.g., "Assignment deadline")
 - Filler or small talk

Transcript chunk:
\"\"\"
{chunk_text}
\"\"\"

Respond using this JSON format:

{parser.get_format_instructions()}

Return only the JSON response.
"""

def generate_topic_summary_prompt(topic, chunk_text, parser, previous_summary=None, other_topics=None):
    summary_part = f"\\nPrevious summary:\\n{previous_summary}" if previous_summary else ""
    others = "\\n".join(f"- {t}" for t in other_topics if t != topic) if other_topics else ""
    format_instructions = parser.get_format_instructions()

    prompt = (
        "You are an expert in summarizing educational transcripts.\\n\\n"
        "Your task is to write a clear and detailed summary **for the following topic**:\\n\\n"
        f"- Focus ONLY on the topic: \"{topic}\"\\n"
        "- Try to **preserve the speaker's structure and flow**\\n"
        "- Include relevant **examples, laws, or Q&A** if explicitly mentioned\\n"
        "- Do NOT summarize other unrelated topics\\n"
    )

    if others:
        prompt += f"- Other topics in this chunk to ignore:\\n{others}\\n"

    prompt += "- If a previous summary exists, enrich it with new details (no repetition)\\n\\n"
    prompt += f'Transcript chunk:\\n\"\"\"\\n{chunk_text}\\n\"\"\"\\n\\n'
    prompt += summary_part + "\\n\\n"
    prompt += (
        "Instructions:\\n"
        "- If a previous summary is provided, consolidate the new information into it without duplicating content.\\n"
        "- If examples, case studies, or real-world illustrations are mentioned in the chunk, "
        "**include them in the summary** as well.\\n\\n"
    )
    prompt += format_instructions

    return prompt

def generate_html_prompt(topic, raw_summary):
    return f"""
You are formatting an educational transcript summary for PDF export.

Use clean, simple HTML formatting:
- <p> for paragraphs
- <ul><li>...</li></ul> for bullet points if present
- Do NOT include any introductory phrases like "Here is the formatted HTML".

Do not include CSS or extra styling.
Format the following content as structured HTML:

\"\"\"
{raw_summary}
\"\"\"
"""

# --- Main Logic ---

def load_transcript(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def generate_summary_pipeline(transcript_path, output_dir, api_key):
    """
    Runs the full summarization pipeline:
    Transcript -> Chunks -> Topic Extraction -> Summarization -> PDF
    """
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found in environment.")
    
    os.environ["NVIDIA_API_KEY"] = api_key
    
    # Init LLMs - using standard instruct model
    llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
    
    # 1. Load and Chunk
    print(f"Loading transcript: {transcript_path}")
    full_text = load_transcript(transcript_path)
    chunks = chunk_text(full_text)
    print(f"Created {len(chunks)} chunks.")

    # 2. Extract Topics
    print("Extracting topics...")
    topic_map = []
    known_topics = []
    
    # Create a structured LLM for topics
    topic_llm = llm.with_structured_output(TopicExtraction)

    for i, chunk in enumerate(tqdm(chunks, desc="Topic Extraction")):
        # We don't need the elaborate parser prompt anymore, just a simple instruction
        prompt = f"""
        Analyze this transcript chunk and act as a Topic Extractor.
        Identify logical transitions and divide the text into sections based on topic changes, speaker changes, or clear shifts.
        Ignore filler/small talk.
        
        Transcript chunk:
        {chunk}
        """
        try:
            parsed = topic_llm.invoke(prompt)
            topics = parsed.topics
            
            topic_map.append({
                "chunk_index": i,
                "topics": topics
            })
            
            for topic in topics:
                if topic not in known_topics:
                    known_topics.append(topic)
        except Exception as e:
            print(f"Chunk {i} failed to extract topics: {e}")

    # 3. Generate Summaries per Topic
    print("Generating summaries...")
    topic_summaries: Dict[str, str] = {}
    
    # Create structured LLM for summary
    summary_llm = llm.with_structured_output(TopicSummary)
    
    for item in tqdm(topic_map, desc="Summarization"):
        chunk_index = item["chunk_index"]
        chunk_content = chunks[chunk_index]
        topics = item["topics"]
        
        for topic in topics:
            previous = topic_summaries.get(topic)
            others = [t for t in topics if t != topic]
            
            prompt = f"""
            You are an expert in summarizing educational transcripts.
            Summarize the following transcript chunk specifically for the topic: "{topic}".
            
            Context:
            - Other topics in this chunk (ignore these): {others}
            - Previous summary content (if any): {previous}
            
            Instructions:
            - Focus ONLY on "{topic}"
            - Preserve examples and technical details.
            - If we have a previous summary, merge new info into it smoothly.
            
            Transcript chunk:
            {chunk_content}
            """
            
            try:
                parsed = summary_llm.invoke(prompt)
                
                if parsed is None:
                    print(f"  > Warning: Could not summarize '{topic}' in this chunk (Model returned None). Skipping.")
                    continue

                if topic in topic_summaries:
                    topic_summaries[topic] += "\n" + parsed.summary
                else:
                    topic_summaries[topic] = parsed.summary
            except Exception as e:
                print(f"Failed to summarize topic '{topic}': {e}")

    # Save Intermediate JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "topic_summaries.json")
    with open(json_path, "w") as f:
        json.dump(topic_summaries, f, indent=2)
    print(f"Saved topic summaries to {json_path}")

    # 4. Generate PDF
    print("Generating PDF...")
    html_summaries = {}
    
    # Formatting to HTML
    for topic, raw_summary in tqdm(topic_summaries.items(), desc="HTML Formatting"):
        time.sleep(1.0) # Rate limit protection
        prompt = generate_html_prompt(topic, raw_summary)
        try:
            response = llm.invoke(prompt)
            html = response.content.strip()
            html_summaries[topic] = html
        except Exception as e:
            print(f"HTML format failed for {topic}: {e}")
            html_summaries[topic] = f"<p>{raw_summary}</p>"

    # Writing PDF
    filename = os.path.splitext(os.path.basename(transcript_path))[0]
    pdf_path = os.path.join(output_dir, f"{filename}_summary.pdf")
    
    pdf = HTMLPDF(title=filename.replace("_", " "))
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    for topic, html in html_summaries.items():
        # Title
        pdf.set_font("Helvetica", "B", 13)
        # Using write_html for the title too to keep it consistent if we wanted, 
        # but write_html handles basic tags. For the header we can just use cell or write_html.
        # FPDF write_html is limited. Let's try to mimic simple HTML processing manually or rely on basic write_html
        # Since we generated HTML, we should use write_html.
        
        pdf.write_html(f"<h2>{topic}</h2>")
        
        pdf.set_font("Helvetica", size=12)
        pdf.write_html(html)
        pdf.ln(5)

    pdf.output(pdf_path)
    print(f"PDF saved to: {pdf_path}")
    return pdf_path
