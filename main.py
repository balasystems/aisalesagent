import functions_framework
from flask import Request, jsonify
from vertexai import init
from vertexai.generative_models import GenerativeModel
from bs4 import BeautifulSoup
import traceback
import datetime

PROJECT_ID = "sandbox"
LOCATION = "global"
MODEL_ID = "gemini-3-pro-preview"


@functions_framework.http
def summarize_email(request: Request):
    try:
        request_json = request.get_json(silent=True)
        email_body_html = request_json.get("email_body", "")
        subject = request_json.get("subject", "No Subject")
        sender = request_json.get("from", "Unknown")
        receivedDateTime = request_json.get("receivedDateTime", "")
        toRecipients = request_json.get("toRecipients", "Unknown")

        if not email_body_html:
            return jsonify({"error": "Missing email_body"}), 400

        # Plain-text version just for Gemini analysis
        email_body_text = BeautifulSoup(
            email_body_html or "", "html.parser"
        ).get_text(separator="\n", strip=False)

        # Format receivedDateTime -> "DD-MM-YYYY, HH:MM AM/PM"
        formatted_received = receivedDateTime
        try:
            if receivedDateTime:
                dt = datetime.datetime.fromisoformat(
                    receivedDateTime.replace("Z", "+00:00")
                )
                formatted_received = dt.strftime("%d-%m-%Y, %I:%M %p")
        except Exception:
            pass

        # Init Vertex AI & model
        init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_ID)

        # SUMMARY-ONLY PROMPT (from above)
        prompt = f"""
You are an Sales Assistant. Analyze the following email thread and provide a professional SUMMARY ONLY with the following structure, inside HTML (no Markdown), add the below info in a div with style font-family:Segoe UI, sans-serif; font-size:14px;:

<h4 style="margin: 4px 0 4px 0;">Context Summary:</h4>
Briefly summarize the overall topic or project discussed.

<h4 style="margin: 4px 0 4px 0;">Needs Summary:</h4>
Clearly identify who needs what from whom. Only include outstanding tasks.
- Use this format: <b> Sender : </b> – Outstanding task.

<h4 style="margin: 4px 0 4px 0;">Detailed Email-by-Email Summary:</h4>
Extract all individual emails or message segments in the thread. This includes both standard email messages and any forwarded comments or annotations added by people before forwarding (even if informal). 

For each segment:
- First, extract the date and time from the email metadata or visible header (e.g., "Date: March 17, 2025 at 7:26 PM").
- If no timestamp is available, insert the current timestamp in the format "March 17, 2025 at 7:26 PM".
- For the most recent email in the thread, if no explicit timestamp is visible, use this as its timestamp: "{formatted_received}" in the format "Month DD, YYYY at HH:MM AM/PM".
- Identify the sender name.
- Include all relevant details or actions mentioned in the email body.
- List each message in reverse chronological order (most recent email first).

- IMPORTANT FORMATTING FOR THIS SECTION:
  • Do NOT use <ul>, <ol>, or <li>.
  • For EACH email, output a SINGLE paragraph in exactly this HTML pattern:

    <p style="margin:0; line-height:1.1;">
      ▪ <b>(Email Timestamp) – Sender Name :</b> Summary of key points and any requests/information.
    </p>

  • The bullet "▪ " must appear inside the <p> as shown.
  • Timestamp + sender name + colon remain bold together.
  • The summary text must stay on the same line, inside the same <p>.
  • Do NOT add blank <p> elements or extra <br> lines between entries.

    - Collapse multiple blank lines and extra <br> spacing

<h4 style="margin: 4px 0 4px 0;">Condensed Recap:</h4>
A bulleted quick summary of key points across emails. For the recap, wrap the bullets in:
<div style="background-color:#f5f8fa; 
            border-radius:8px; 
            font-family:Segoe UI, sans-serif; 
            font-size:14px; 
            color:#333; 
            padding:2px 2px; 
            margin-top:2px; 
            margin-bottom:2px; 
            line-height:0.5;">
Use a list:
<ul style="list-style-type: square; padding-left: 15px; margin: 0;">
    <li style="margin-bottom: 4px; line-height:0.3;"> 
    </li>
</ul>
Add a single <hr> after the recap.

VERY IMPORTANT:
- Do NOT include any "Original Email" section.
- Do NOT reconstruct or restate the original email body.
- Do NOT use labels like "Undated", "No date", or similar.
- Do NOT restate or copy these instructions.
- Do NOT wrap any part of the output in Markdown code block syntax like ```html or ``` — just return raw HTML.

Here is the plain-text version of the email thread for your analysis:

{email_body_text}
"""

        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 4096,
                "temperature": 0.2
            },
            safety_settings=[]
        )

        usage = getattr(response, "usage_metadata", None)
        print(f"[summarize-email] USING_MODEL={MODEL_ID}, usage={usage}")

        gemini_summary_html = response.text or ""


        full_html = f"""
<div style="font-family:Segoe UI, sans-serif; font-size:14px;">
  {gemini_summary_html}
  <hr>
  <h4 style="margin: 4px 0 4px 0;">Original Email:</h4>
  <p style="margin:0 0 6px 0;">
    <b>From: </b>{sender}<br>
    <b>Sent : </b>{formatted_received}<br>
    <b>To : </b>{toRecipients}<br>
    <b>Subject: </b>{subject}<br>
  </p>
  <hr>
  {email_body_html}
</div>
"""

        return jsonify({
            "subject": "Re:Summary: " + subject,
            "from": sender,
            "email_body": email_body_html,
            "summary": full_html,   # <- PA should use THIS as body
            "model_used": MODEL_ID,
            "usage": str(usage)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
