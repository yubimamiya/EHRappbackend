# This is a new chat.py file that uses asyncio.gather to parallelize the processing of PDF pages.
# This tried to complete the full payment processing flow
# switch to more concise prompts and less schema sending to reduce token usage
# this now adds many functions to try and save the results to the database

import json
import os

# import additional packages
from quart import Blueprint, request, jsonify
import fitz  # PyMuPDF
from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance
from io import BytesIO
import base64
import re
import asyncio
import tiktoken
import pandas as pd
import pytz
import uuid
from typing import List
from collections import defaultdict

from asyncio import to_thread  # safer than manually using loop.run_in_executor
# from .connect_to_db import connect_to_db  # import your function

# import relevant packages
# YUBI: I remember datetime giving me some problems before, so we will see if this works
# Python has a built in datetime module
from datetime import datetime, date, time, timedelta
import asyncpg


import azure.identity.aio
import openai
from quart import (
    Blueprint,
    Response,
    current_app,
    render_template,
    request,
    stream_with_context,
)

# Change line to avoid trying to load templates or serve static assets that aren't needed in the back end
# bp = Blueprint("chat", __name__, template_folder="templates", static_folder="static")
bp = Blueprint("chat", __name__)

# establish connection to database
async def connect_to_db():
    try:
        '''
        # Fetch DB connection details from environment with exact casing and dashes
        curr_host = os.environ["DB_HOST"]
        curr_port = os.environ.get("DB_PORT", "5432")
        curr_dbname = os.environ["DB_NAME"]
        curr_user = os.environ["DB_USER"]
        curr_password = os.environ["DB_PASSWORD"]
        '''
        # YUBI: revert back to this for now because adding more environment variables is becoming a headache 
        # NOTE: these must be edited by the user if the connection ever changes
        curr_host = "c-rec-ex-app-db-cluster.h2jfiotjltxxps.postgres.cosmos.azure.com"
        curr_port = "5432"
        curr_dbname = "citus"
        curr_user = "citus"
        curr_password = "xenon2025!"


        # Create asyncpg connection pool (recommended)
        pool = await asyncpg.create_pool(
            host=curr_host,
            port=curr_port,
            database=curr_dbname,
            user=curr_user,
            password=curr_password,
            ssl="require",  # Azure Cosmos DB requires SSL
            min_size=1,
            max_size=5,
        )

        print("Connection to Azure Cosmos DB for PostgreSQL successful.")
        return pool
    except Exception as e:
        print("Failed to connect to the database.")
        print("Error:", e)
        return None
    

# helper function to load google spreadsheet from google drive
async def load_google_sheet(sheet_id, sheet_name):
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}&headers=1'
    df = pd.read_csv(url)
    # df['shipto'] = df['shipto'].astype(str)
    # convert shipto number into an Int64 value
    df['shipto'] = pd.to_numeric(df['shipto'], errors='coerce').astype('Int64')
    return df[['shipto', 'Location']]


@bp.before_app_serving
async def configure_openai():
    openai_host = os.getenv("OPENAI_HOST", "github")
    # YUBI: this is gpt-4o
    bp.model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    if openai_host == "local":
        # Use a local endpoint like llamafile server
        current_app.logger.info("Using model %s from local OpenAI-compatible API with no key", bp.model_name)
        bp.openai_client = openai.AsyncOpenAI(api_key="no-key-required", base_url=os.getenv("LOCAL_OPENAI_ENDPOINT"))
    elif openai_host == "github":
        current_app.logger.info("Using model %s from GitHub models with GITHUB_TOKEN as key", bp.model_name)
        bp.openai_client = openai.AsyncOpenAI(
            api_key=os.environ["GITHUB_TOKEN"],
            base_url="https://models.inference.ai.azure.com",
        )
    else:
        client_args = {}
        # Use an Azure OpenAI endpoint instead,
        # either with a key or with keyless authentication
        # YUBI: remove key authentication step to make sure it doesn't go there
        '''
        if os.getenv("AZURE_OPENAI_KEY_FOR_CHATVISION"):
            # Authenticate using an Azure OpenAI API key
            # This is generally discouraged, but is provided for developers
            # that want to develop locally inside the Docker container.
            current_app.logger.info("Using model %s from Azure OpenAI with key", bp.model_name)
            client_args["api_key"] = os.getenv("AZURE_OPENAI_KEY_FOR_CHATVISION")
        '''
        # else:
        if os.getenv("RUNNING_IN_PRODUCTION"):
            client_id = os.getenv("AZURE_CLIENT_ID")
            current_app.logger.info(
                "Using model %s from Azure OpenAI with managed identity credential for client ID %s",
                bp.model_name,
                client_id,
            )
            azure_credential = azure.identity.aio.ManagedIdentityCredential(client_id=client_id)
        else:
            # should run this block
            tenant_id = os.environ["AZURE_TENANT_ID"]
            current_app.logger.info(
                "Using model %s from Azure OpenAI with Azure Developer CLI credential for tenant ID: %s",
                bp.model_name,
                tenant_id,
            )
            azure_credential = azure.identity.aio.AzureDeveloperCliCredential(tenant_id=tenant_id)
        client_args["azure_ad_token_provider"] = azure.identity.aio.get_bearer_token_provider(
            azure_credential, "https://cognitiveservices.azure.com/.default"
        )
        bp.openai_client = openai.AsyncAzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2025-01-01-preview",
            **client_args,
        )

    # establish connection to PostgreSQL DB
    try:
        bp.db_conn = await connect_to_db()
        if bp.db_conn:
            current_app.logger.info("Successfully connected to PostgreSQL.")
        else:
            raise ValueError("Database pool returned None.")
    except Exception as e:
        current_app.logger.error(f"Database connection failed: {e}")
        await asyncio.sleep(10)  # throttle retry storm
        bp.db_conn = None

    # load the patient schema json template from data folder
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'patient_schema.json')
    try:
        with open(file_path, 'r') as f:
            bp.patient_schema = json.load(f)
        current_app.logger.info("Loaded patient schema from %s", file_path)
    except Exception as e:
        current_app.logger.error("Failed to load patient schema: %s", e)
        bp.patient_schema = {}  # Fallback or raise if critical
        
    # Load shipto spreadsheet from Google Sheets
    try:
        sheet_id = '1_V2vuOLrDyYB6Ksqq5j9X8OSTIfSZUjqL4Rh0P4Puuk'
        sheet_name = 'ShiptoLocation'
        bp.shipto_df = await load_google_sheet(sheet_id, sheet_name)
        current_app.logger.info("Successfully loaded ShiptoLocation Google Sheet.")
    except Exception as e:
        current_app.logger.error(f"Failed to load Google Sheet: {e}")
        bp.shipto_df = pd.DataFrame(columns=['shipto', 'Location'])  # fallback empty DataFrame



@bp.after_app_serving
async def shutdown_openai():
    await bp.openai_client.close()

    # Close the PostgreSQL DB Connection if open
    if hasattr(bp, "db_conn") and bp.db_conn:
        try:
            await bp.db_conn.close()
            current_app.logger.info("PostgreSQL connection pool closed.")
        except Exception as e:
            current_app.logger.warning(f"Failed to close DB connection pool: {e}")


# YUBI: Do not need to render HTML template
# Test debugging function
@bp.get("/")
async def index():
    return await render_template("index.html")

# Test debugging function
@bp.route("/test", methods=["GET", "POST", "OPTIONS"])
async def test():
    if request.method == "GET":
        return "GET method works!"
    elif request.method == "POST":
        return "POST method works!"
    elif request.method == "OPTIONS":
        # For OPTIONS, you can also set CORS headers if needed
        return "OPTIONS method works!"


# Convert a PyMuPDF page to a binarized, enhanced PIL image.
async def convert_pdf_page_to_image(page, dpi_threshold):
    pix = page.get_pixmap(dpi=dpi_threshold)
    img_bytes = pix.tobytes("png")

    # Convert to grayscale
    pil_image = Image.open(BytesIO(img_bytes)).convert("L")

    # Enhance contrast while still grayscale
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_image = enhancer.enhance(2.0)  # You can tweak this value if needed

    # Binarize
    # binarized_image = enhanced_image.point(lambda x: 0 if x < 128 else 255, mode="1")
    binarized_image = enhanced_image.convert("1")

    return binarized_image

# Convert a PIL image to a base64 string.
async def image_to_base64(img: Image.Image):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


async def filter_pdf_dual_path(doc, inclusion_keywords=None, exclusion_keywords=None, max_images=8, ocr_threshold=100, case_sensitive=False, dpi_threshold=200):
    """
    Filters a PyMuPDF PDF document to determine which pages to include for processing.

    Parameters:
    - doc: fitz.Document object.
    - inclusion_keywords: List of keywords to include pages (optional).
    - exclusion_keywords: List of keywords to exclude pages (optional).
    - max_images: Maximum allowed images per page before excluding.
    - ocr_threshold: Minimum characters needed to consider text extraction successful.
    - case_sensitive: Whether keyword matching is case-sensitive.

    Returns:
    - List of page numbers (0-indexed) to include.
    """

    inclusion_keywords = inclusion_keywords or []
    exclusion_keywords = exclusion_keywords or []
    pages_to_include = []

    for i, page in enumerate(doc):

        # Count images on page
        if len(page.get_images(full=True)) > max_images:
            continue  # too many images, skip

        # Step 1: Try direct text extraction
        text = page.get_text()

        # If it can't extract any text, there is no need to check anything
        if text == "":
            # add page
            pages_to_include.append(i)

        extracted_text = text if case_sensitive else text.lower()

        # Step 3: Exclude if matches any exclusion keywords
        if exclusion_keywords:
            for kw in exclusion_keywords:
                kw_check = kw if case_sensitive else kw.lower()
                if kw_check in extracted_text:
                    continue  # if it has the keyword, skip this page

        # Step 4: Check inclusion criteria
        # this shouldn't be used
        if inclusion_keywords:
            if not any((kw if case_sensitive else kw.lower()) in extracted_text for kw in inclusion_keywords):
                continue  # no match, skip this page


        # If it passed all filters, include the page
        pages_to_include.append(i)

    return pages_to_include



# Call the AI model with the image and user message
async def call_model_on_image(image_base64, user_message, processing_mode, NUM_PAGES=None):
    # YUBI: only use NUM_PAGES for payment processing mode
    section_prompt = ""
    user_content = []
    # YUBI: I am setting this right now, but we want this to be dynamic based on the number of pages in the PDF
    detail_level = "auto"  # Default detail level for images


    if processing_mode == "payment":
        section_prompt += ("This image is a section of a scanned document that may contain Explanation of Benefits (EOB) or payments for medical services."
	        "Do not examine the image any further if it does not contain any monetary numbers, if it does contain the scanned front of an envelope, or if it does contain any of the following phrases: \'U.S. Postage\', \'This is just a notification\', \'Business Reply Mail\', \'Why was this decision made\', \'Frequently Asked Questions\', \'Overpayment\', \'Claim Remittance\'."
            "An EOB is typically titled \'Explanation of Benefits\' or \'Claim Details\' and includes a table of medical costs. The amount of money paid by the health insurance company (Amount Paid) is written in the bottom-right entry of the EOB table. Represent an EOB as a JSON object that contains the following fields: Patient Name (string) and Amount Paid (number). Discard and do not return EOB objects where Amount Paid is 0."
            "There are 2 types of Payment: Check and Virtual Card. A check typically appears as a wide, bordered, rectangular area enclosing a printed check number, a \'Pay to the Order of\' line, a payment amount written in numbers and words, a signature line, and a MICR number sequence. It is not a check if the rectangular area encloses a dense table."
            "A virtual card typically appears as an outlined rectangle with rounded corners enclosing a 16-digit card number, a CVV code, an expiration date, and a credit card company logo. The card may appear with the text \'Mastercard Express ClaimsCard\' or \'Virtual Card\'. The card is displayed next to a payment amount. It is not a card if there are no numbers enclosed by the outlined rectangle. Otherwise, if it looks similar to a payment, it is a payment."
            f"For every Payment found, extract the page number it is on. The page number is written as \'Page # of {NUM_PAGES}\' on every page, where # represents the page number. Represent a Payment as a JSON object that contains the following fields: Payer Name (string), Payee Name (string), Amount Paid (number), Payment Type (string), Payment Page Number (integer), Card Number (number), CVV Code (number), Expiration Date (string), Check Number (number). Missing fields can be set to \'null\'."
            "Return two arrays in a JSON object with the following keys: page_array and objects_array. page_array is an array of all the page numbers containing a Payment. objects_array is an array of all EOB and Payment JSON objects found. Format each array and your full response as raw JSON only with no extra text or formatting like markdown backticks. Empty arrays are allowed.")
        # user_content.append({"type": "text", "text": json.dumps(bp.payment_schema)})
        # user_content.append({"type": "text", "text": json.dumps(bp.EOB_schema)})
    elif processing_mode == "inventory":
        # inventory invoice processing mode
        section_prompt += ("This image is an invoice for medical supplies."
            "There are two types of Payees for the invoices: Airgas or McKesson. The Payee is written in the top left of every image. If the Payee is Airgas, find the PO BOX Number for Airgas. If the Payee is McKesson, there is no PO BOX Number. Represent the invoice as a JSON object that contains the following fields in this order: Ship To Number, Payee, Amount, Invoice Date, Due Date, Invoice Number, PO BOX Number."
            "The Ship To Number is likely written near the text \'Shipped To:\' or \'SHIP TO:\'. The PO BOX Number is likely written in the upper right section of an Airgas image beneath the text \'PLEASE MAKE CHECKS PAYABLE AND REMIT TO:\' and \'Airgas USA, LLC\'. If a field is not in the image, the returned value is \'null\'."
            "Format all dates as MM/DD/YYYY as a string."
            "Return the JSON object inside of a single array. Format each array and your full response as raw JSON only with no extra text or formatting like markdown backticks.")
    elif processing_mode == "provider":
        # provider invoice processing mode
        section_prompt += ("This image is a section of an employee invoice form completed by an anesthesiologist. It may include a Provider Invoice table or a Case Log table."
            "There are 2 Table Categories of Provider Invoice tables: Hours and Expenses. Hours tables show pay from hours worked. Expenses tables show pay from mileage or hotel expenses. For every Provider Invoice table present, represent every non-empty row in the table as a distinct JSON Object with the following fields: Object Type, Table Category, Provider Name, Provider Title, Facility Name, Date, Hourly Pay Rate, Total Hours Worked, Total Pay. Examples of Provider Title are CRNA and MD. Represent Total Pay, Total Hours Worked, and Hourly Pay Rate as a number with no symbols. Object Type is always \'Provider Invoice\'. Missing values can be set to \'null\'."
            "A Case Log table includes a list of anesthesia procedures. If a Case Log table is present, represent every non-empty row in the table as a distinct JSON Object with the following fields: Object Type, Date, Patient Initials, Performed Procedure, Type of Anesthesia Used, Presence of Complications (Yes or No). Object Type is always \'Case Log\'."
            "Format all Date values as MM/DD/YYYY. Return an array of all JSON objects found in the image. Format your response as raw JSON only with no extra text or formatting like markdown backticks. An empty array is allowed if relevant tables are not found in the image.")
    else:
        # default processing mode is billing
        section_prompt += (
            "The file contains scanned documents of medical patients. This image contains 1 to 2 pages from the documents. Do not look any further at pages that contain any of the following titles: Consent Form, Schedule Report, Anesthesia Record, Consent for Anesthesia Services, Referral Details, Discharge Instructions, Medication Reconciliation Form, EGD Report."
            "Extract the following for each patient if present: Full Name, Date of Birth, Sex, Address, Email, Phone, Primary and Secondary Insurance Name, Type, Member ID, Group ID, CPT Code, ICD Code."
            "Insurance Type is either Medicare or Commercial (which includes all others). If both are present, Medicare is the primary."
            "Format each field as a separate key-value pair: the key is the patient\'s full name, the value is \'Field Name Field Value\'. Repeat the patient name as the key for each field."
            "Do not include commas within values. Omit any fields not found."
            "Return all pairs as a single comma-separated list in raw JSON only. Do not include any explanations and do not wrap the response in markdown formatting such as triple backticks or \'```json\'."
        )

    # This sends all messages, so API request may exceed token limits
    all_messages = [{"role": "system", "content": "You are an information extraction system."}]
    if image_base64:
        user_content.append({"text": user_message, "type": "text"})
        user_content.append({"text": section_prompt, "type": "text"})
        user_content.append({"image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": detail_level}, "type": "image_url"})
        all_messages.append({"role": "user", "content": user_content})

    # YUBI: I'm going to use same AI model for both processing modes for now
    chat_coroutine = await bp.openai_client.chat.completions.create(
        # Azure Open AI takes the deployment name as the model name
        model=bp.model_name,
        messages=all_messages,
        stream=True,
        temperature=0.2,
    )

    # save answers
    response_text = ""
    async for chunk in chat_coroutine:
        if chunk and chunk.choices:
            delta = chunk.choices[0].delta
            if delta and hasattr(delta, "content") and delta.content:
                response_text += delta.content

    # Strip markdown backticks and clean output
    cleaned_response = response_text.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[len("```json"):].strip()
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[len("```"):].strip()
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3].strip()

    if processing_mode == "payment":
        try:
            data = json.loads(cleaned_response)
            page_array = data.get('page_array', [])
            objects_array = data.get('objects_array', [])
        except json.JSONDecodeError:
            print("Failed to decode response as JSON.")
            page_array, objects_array = [], []

        # return an array of the pages with payments and an array of the JSON data instances
        return page_array, objects_array
    
    return cleaned_response  # concatenated string for all other modes
        

# Call the AI model for follow-up questions
async def call_model_followup(prompt):
    
    # This sends all messages, so API request may exceed token limits
    all_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    user_content = []
    user_content.append({"text": prompt, "type": "text"})
    all_messages.append({"role": "user", "content": user_content})

    # send to model
    chat_coroutine = await bp.openai_client.chat.completions.create(
        # Azure Open AI takes the deployment name as the model name
        model=bp.model_name,
        messages=all_messages,
        stream=True,
        temperature=0.2,
    )

    # save answers
    response_text = ""
    async for chunk in chat_coroutine:
        if chunk and chunk.choices:
            delta = chunk.choices[0].delta
            if delta and hasattr(delta, "content") and delta.content:
                response_text += delta.content

    return response_text


async def summarize_pages(partials):
    """Concatenate all arrays of pages into one array."""
    # assumes that input is a nested array of numbers
    merged = set()
    for pages in partials:
        merged.update(pages)
    # returns a single, flat array of numbers that are sorted in ascending order
    return sorted(merged)



# summarize answers function that batches the partial answers for batched calls to AI model
# returns a list of JSON data instances
# change token limit from 6000 to 12000
async def summarize_matches(partials, batch_token_limit=12000):
    """Aggregate partial answers into a single list of JSON objects by batching."""
        
    def count_tokens(text):
        try:
            if not isinstance(text, str):
                text = json.dumps(text)
            enc = tiktoken.get_encoding(bp.model_name)
            return len(enc.encode(text))
        except Exception:
            if not isinstance(text, str):
                text = str(text)
            return len(text.split())

    # match_schema_file = bp.match_schema

    # YUBI: testing simpler prompt    
    summary_prompt = (
        "This is a JSON array containing two types of objects: Explanation of Benefits (EOB) objects and Payment objects."
        "EOB objects include the fields \'Patient Name\' and \'Amount Paid\'."
        "Payment objects include fields such as \'Payer Name\', \'Receiver Name\', \'Amount Paid\', \'Payment Type\', and other payment-specific fields."
        "Match each EOB object to a Payment object only if the \'Amount Paid\' values are equal."
        "Return a new array of JSON objects, each representing one matched pair, with the following fields: Payer Name, Payee Name, Patient Name, Amount Paid, Payment Type, Payment Page Number, Card Number, CVV Code, Expiration Date, and Check Number. Fields can be \'null\' if they do not exist."
        "If no matches are found, return an empty array. Output raw JSON only with no extra text or formatting like markdown backticks. Do not include unmatched objects."
    )

    # Chunk partials to respect token limit per batch
    batches = []
    current_batch = []
    current_tokens = 0

    for part in partials:
        part_str = json.dumps(part)
        tokens = count_tokens(part_str)
        if current_tokens + tokens > batch_token_limit and current_batch:
            batches.append(current_batch)
            current_batch = [part_str]
            current_tokens = tokens
        else:
            current_batch.append(part_str)
            current_tokens += tokens


    if current_batch:
        batches.append(current_batch)

    all_json_objects = []

    for batch in batches:
        # partials_connected = "\n".join(batch)
        # YUBI: debugging statement, read in partials as a valid JSON array
        partials_connected = json.dumps([json.loads(p) for p in batch])
        all_messages = [
            {"role": "system", "content": "You are an information extraction system."},
            {
                "role": "user",
                "content": [
                    {"text": partials_connected, "type": "text"},
                    {"text": summary_prompt, "type": "text"},
                    # {"type": "text", "text": json.dumps(match_schema_file)},
                ]
            }
        ]

        chat_coroutine = await bp.openai_client.chat.completions.create(
            model=bp.model_name,
            messages=all_messages,
            stream=True,
            temperature=0.2,
        )

        response_text = ""
        async for chunk in chat_coroutine:
            if chunk and chunk.choices:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    response_text += delta.content

        # --- Cleaning Markdown Wrappers ---
        cleaned_response = response_text.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[len("```json"):].strip()
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[len("```"):].strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3].strip()
        
        try:
            parsed_batch = json.loads(cleaned_response)
            if isinstance(parsed_batch, list):
                all_json_objects.extend(parsed_batch)
            else:
                raise ValueError("Expected a list of JSON objects")
        except Exception as e:
            raise RuntimeError(f"Failed to parse model output as JSON: {str(e)}\nRaw response: {response_text}")

    return all_json_objects



# summarize answers function that batches the partial answers for batched calls to AI model
# returns a list of JSON data instances
# YUBI: try increasing batch token limit to see if this fixes issue
async def summarize_answers(partials, processing_mode, batch_token_limit=15000):
    """Aggregate partial answers into a single list of JSON objects by batching."""

    def count_tokens(text):
        try:
            if not isinstance(text, str):
                text = json.dumps(text)
            enc = tiktoken.encoding_for_model(bp.model_name)
            return len(enc.encode(text))
        except Exception:
            if not isinstance(text, str):
                text = str(text)
            return len(text.split())  # Fallback: approx 1 token per word

    if processing_mode == "provider":
        # YUBI: add prompt for provider here
        summary_prompt = (
            "This is an array of JSON objects. The Object Type field of each object specifies whether it is a \'Provider Invoice\' or a \'Case Log\'. Each object has a Date field. Convert every Provider Invoice object to a new Service JSON object in the following steps."
            "Step 1: For every Provider Invoice where the Table Category field is equal to \'Hours\', find all matching Case Log objects with the same Date value. Combine the information from these objects into one new Service JSON object with these fields: Provider Name, Provider Title, Date, Table Category, Total Pay, Facility Name, Total Hours Worked, Hourly Pay Rate, Case Log Number (number of matching Case Logs), Patient Initials (list of initials from matching Case Logs), Full Case Log (array of the matching Case Logs, but remove Object Type, Anesthesiologist Name, and Date from each one). If there are no matching Case Log objects, the last 3 fields of the Service object should be \'null\'."
            "Step 2: For every Provider Invoice where the Table Category field is equal to \'Expenses\', convert it into a Service object with only the first 6 fields filled. The rest of the fields should be \'null\'."
            "Return an array of Service JSON objects. Only return raw JSON with no extra text, formatting, or markdown."
        )


    else:
        # default processing mode is billing
        # YUBI: testing simpler prompt
        summary_prompt = (
            "This is a comma-separated list of key-value pairs about medical patients. "
            "Each key is a patient's full name; each value is a labeled field (e.g., 'Date of Birth 01/01/1980'). "
            "Some names may refer to the same person despite differences (e.g., middle names, initials, or capitalization). "
            "Group similar names and use the longest full name in each group. "
            "Aggregate fields for each patient into a single JSON object. Each patient object must contain the following fields: Patient Name, Date of Birth, Sex, Address, Email, Phone, Primary Insurance Name, Primary Insurance Type, Primary Insurance Member ID, Primary Insurance Group ID, Secondary Insurance Name, Secondary Insurance Type, Secondary Insurance Member ID, Secondary Insurance Group ID, CPT Codes, and ICD Codes. All fields are strings, except CPT Codes and ICD Codes, which are arrays of strings that include all CPT and ICD codes found."
            "For other fields with conflicting values, choose the most likely one. Missing fields should be \'null\'."
            "Return an array of patient JSON objects. Output raw JSON only. Do not include any explanations and do not wrap the response in markdown formatting such as triple backticks or \'```json\'."
        )

    all_json_objects = []

    if processing_mode == "provider":
        # Group partials by Date
        grouped_by_date = defaultdict(list)

        for part in partials:
            try:
                # Ensure we can parse each part
                parsed = json.loads(part) if isinstance(part, str) else part

                # It should be a list of JSON objects
                if isinstance(parsed, list):
                    for obj in parsed:
                        if isinstance(obj, dict):
                            date = obj.get("Date", "__no_date__")
                            grouped_by_date[date].append(obj)
                elif isinstance(parsed, dict):
                    date = parsed.get("Date", "__no_date__")
                    grouped_by_date[date].append(parsed)
                else:
                    print(f"[WARN] Skipping unexpected format in part: {type(parsed)}")
            except Exception as e:
                print(f"[ERROR] Failed to parse part: {part[:200]}... Error: {e}")

        # Each group becomes its own batch
        batches = list(grouped_by_date.values())

    else:
        # Original batching by token limit
        batches = []
        current_batch = []
        current_tokens = 0

        for part in partials:
            tokens = count_tokens(part)
            if current_tokens + tokens > batch_token_limit and current_batch:
                batches.append(current_batch)
                current_batch = [part]
                current_tokens = tokens
            else:
                current_batch.append(part)
                current_tokens += tokens

        if current_batch:
            batches.append(current_batch)

    # all_json_objects = []

    for batch in batches:
        # YUBI: currently DEBUGGING so reverting back to this
        # partials_connected = "\n".join(batch)
        # YUBI: edit this to be more flexible for provider processing mode
        partials_connected = "\n".join(part if isinstance(part, str) else json.dumps(part) for part in batch)

        all_messages = [
            {"role": "system", "content": "You are an information extraction system."},
            {
                "role": "user",
                "content": [
                    {"text": partials_connected, "type": "text"},
                    {"text": summary_prompt, "type": "text"},
                ]
            }
        ]

        chat_coroutine = await bp.openai_client.chat.completions.create(
            model=bp.model_name,
            messages=all_messages,
            stream=True,
            temperature=0.2,
        )

        response_text = ""
        async for chunk in chat_coroutine:
            if chunk and chunk.choices:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    response_text += delta.content

        try:
            # Strip common markdown formatting artifacts before parsing
            # YUBI: does this work with raw JSON outputs from gpt-4o?
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[len("```json"):].strip()
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[len("```"):].strip()
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3].strip()

            parsed_batch = json.loads(cleaned_response)
            if isinstance(parsed_batch, list):
                all_json_objects.extend(parsed_batch)
            else:
                raise ValueError("Expected a list of JSON objects")
        except Exception as e:
            raise RuntimeError(f"Failed to parse model output as JSON: {str(e)}\nRaw response: {response_text}")

    return all_json_objects


# function to connect summaries of all JSON objects into a final answer
# for billing processing mode only
async def connect_summaries(all_json_objects, processing_mode):
    """Aggregate summarized chunks into a answer ."""

    json_input_str = json.dumps(all_json_objects)
    
    # call model with final message prompt
    all_messages = [{"role": "system", "content": "You are an information extraction system."}]

    # YUBI EDIT: add schema file for the payment information
    if processing_mode == "payment":
        schema_file = ""
    else:
        # default processing mode is billing
        schema_file = bp.patient_schema

    final_prompt = ""
    if processing_mode == "payment":
        # YUBI: add payment prompt here
        final_prompt += "add prompt here"
    else:
        # default processing mode is billing
        # YUBI: testing simpler prompt
        final_prompt += ("This is a list of JSON data instances that each represent a patient. Review the list and combine any data instances that refer to the same patient. Data instances refer to the same patient if they have a similar Full Name (e.g., middle names, initials, or capitalization). "
            "Aggregate fields for each patient into a single JSON object. Each patient object must contain the following fields: Patient Name, Date of Birth, Sex, Address, Email, Phone, Primary Insurance Name, Primary Insurance Type, Primary Insurance Member ID, Primary Insurance Group ID, Secondary Insurance Name, Secondary Insurance Type, Secondary Insurance Member ID, Secondary Insurance Group ID, CPT Codes, and ICD Codes. All fields are strings, except CPT Codes and ICD Codes, which are arrays of strings that include all CPT and ICD codes found."
            "For other fields with conflicting values, choose the most likely one. "
            "Missing fields should be \'null\'. Return an array of patient JSON objects. Output raw JSON only; no extra text or formatting like markdown backticks.")


    # IDK if this check is necessary
    user_content = []
    user_content.append({"text": json_input_str, "type": "text"})
    user_content.append({"text": final_prompt, "type": "text"})
    # add schema file to the user content
    # user_content.append({"type": "text", "text": json.dumps(schema_file)})
    all_messages.append({"role": "user", "content": user_content})
        

    # send to model
    chat_coroutine = await bp.openai_client.chat.completions.create(
        # Azure Open AI takes the deployment name as the model name
        model=bp.model_name,
        messages=all_messages,
        stream=True,
        temperature=0.2,
    )

    # save answers
    response_text = ""
    async for chunk in chat_coroutine:
        if chunk and chunk.choices:
            delta = chunk.choices[0].delta
            if delta and hasattr(delta, "content") and delta.content:
                response_text += delta.content

    # --- Clean markdown wrappers like ```json ---
    cleaned_response = response_text.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[len("```json"):].strip()
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[len("```"):].strip()
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3].strip()

    return cleaned_response


def validate_patient_fields(patients):

    annotated = []
    for patient in patients:
        entry = {}
        for key, value in patient.items():
            if isinstance(value, str) and value in ["null", "None", "", "N/A", "not provided", " "]:
                # do not highlight empty cells because they are already empty
                valid = True
            elif value == None:
                valid = True
            elif key == "Date of Birth":
                # value can be any arrangement of numbers and dashes or slashes, but can't have any alphabetic characters
                valid = bool(re.match(r"^\d{1,4}[-/]\d{1,4}[-/]\d{1,4}$", str(value)))
                reason = None if valid else "Date has invalid characters or format"
            elif key == "Sex":
                valid = value in {"M", "F", "Male", "Female"}
                reason = None if valid else "Must be 'M' or 'F'"
            elif key == "Phone Number":
                valid = bool(re.match(r"^\d{10}$", str(value)))
                reason = None if valid else "Must be 10 digits with no dashes, parentheses, or spaces"
            elif key in {"Primary Insurance Type", "Secondary Insurance Type"}:
                valid = value in {"Medicare", "Commercial"}
                reason = None if valid else "Must be 'Medicare' or 'Commercial'"
            elif key in {"Primary Insurance Member ID", "Primary Insurance Group ID",
                         "Secondary Insurance Member ID", "Secondary Insurance Group ID"}:
                valid = bool(re.match(r"^[A-Z0-9]+$", str(value)))
                reason = None if valid else "Must be alphanumeric with no spaces"
            elif key == "CPT Codes":
                # value must be an array of strings where each string is a 5-digit number and there are no more than 5 strings in the array
                valid = isinstance(value, list) and all(
                    isinstance(code, str) and re.match(r"^\d{5}$", code) for code in value
                ) and len(value) <= 5
                reason = None if valid else "Each CPT code must be a 5-digit number and there can be no more than 5 codes"
            elif key == "ICD Codes":
                # value must be an array of strings where each string is alphanumeric with 3 to 7 characters and there are no more than 5 strings in the array
                valid = isinstance(value, list) and all(
                    isinstance(code, str) and re.match(r"^[A-Z0-9]{3,7}$", code) for code in value
                ) and len(value) <= 5
                reason = None if valid else "Each ICD code must be alphanumeric with 3 to 7 characters and there can be no more than 5 codes"
            else:
                valid = True
                reason = None
            entry[key] = {"value": value, "valid": valid}
            
            if not valid:
                entry[key]["reason"] = reason
        annotated.append(entry)
    return annotated


def validate_payment_fields(payments):
    # set everything to valid by default
    annotated = []
    for payment in payments:
        entry = {}
        for key, value in payment.items():
            if isinstance(value, str) and value in ["null", "None", "", "N/A", "not provided", " "]:
                # do not highlight empty cells because they are already empty
                valid = True
            elif value == None:
                valid = True
            elif key == "Amount Paid":
                # Amount Paid should be a valid monetary value (e.g., 123.45)
                valid = bool(re.match(r"^\d+(\.\d{1,2})?$", str(value)))
                reason = None if valid else "Must be a valid monetary value"
            elif key in {"Payer Name", "Payee Name", "Patient Name"}:
                # Payer name should be a string with only alphabetic characters and spaces
                valid = bool(re.match(r"^[A-Za-z\s]+$", str(value)))
                reason = None if valid else "Must contain only alphabetic characters and spaces"
            elif key == "Payment Type":
                # Payment Type should be either "Check" or "Virtual Card"
                valid = value in {"Check", "Virtual Card"}
                reason = None if valid else "Must be 'Check' or 'Virtual Card'"
            elif key == "Payment Page Number":
                # Page number should be an integer
                valid = isinstance(value, int) and value > 0
                reason = None if valid else "Must be a positive integer"
            elif key == "Card Number":
                # Card Number should be a 16-digit number
                valid = bool(re.match(r"^\d{16}$", str(value)))
                reason = None if valid else "Must be a 16-digit number"
            elif key == "CVV":
                # CVV should be a 3 to 4 digit number
                valid = bool(re.match(r"^\d{3,4}$", str(value)))
                reason = None if valid else "Must be a 3 or 4 digit number"
            elif key == "Expiration Date":
                # Expiration Date should be in MM/YY format
                valid = bool(re.match(r"^(0[1-9]|1[0-2])/\d{2}$", str(value)))
                reason = None if valid else "Must be in MM/YY format"
            elif key == "Check Number":
                # Check Number should be a number
                valid = bool(re.match(r"^\d+$", str(value)))
                reason = None if valid else "Must be a number"
            else:
                valid = True  # Other fields are considered valid by default
                reason = None
            entry[key] = {"value": value, "valid": valid}

            if not valid:
                entry[key]["reason"] = reason

        annotated.append(entry)
    return annotated

def validate_inventory_fields(inventory):
    # set everything to valid by default
    annotated = []
    for instance in inventory:
        entry = {}
        for key, value in instance.items():
            if isinstance(value, str) and value in ["null", "None", "", "N/A", "not provided", " "]:
                # do not highlight empty cells because they are already empty
                valid = True
            elif value == None:
                valid = True
            elif key == "Ship To Number" or key == "Invoice Number":
                valid = bool(re.match(r"^\d+$", str(value)))
                reason = None if valid else "Must be a number"
            elif key == "Payee":
                valid = value in {"Airgas", "McKesson"}
                reason = None if valid else "Payee must be Airgas or McKesson"
            elif key == "Invoice Date" or key == "Due Date":
                # update to be MM/DD/YYYY format
                valid = bool(re.match(r"^(0[1-9]|1[0-2])/([0][1-9]|[12][0-9]|3[01])/\d{4}$", str(value)))
                reason = None if valid else "Must be in MM/DD/YYYY format"
            else:
                valid = True
                reason = None
            entry[key] = {"value": value, "valid": valid}

            if not valid:
                entry[key]["reason"] = reason

        annotated.append(entry)
    return annotated

def validate_provider_fields(provider):
    # set everything to valid by default
    annotated = []
    for instance in provider:
        entry = {}
        for key, value in instance.items():
            if isinstance(value, str) and value in ["null", "None", "", "N/A", "not provided", " "]:
                # do not highlight empty cells because they are already empty
                valid = True
            elif value == None:
                valid = True
            elif key in ["Total Pay", "Total Hours Worked", "Hourly Pay Rate", "Case Log Number"]:
                valid = bool(re.match(r"^\d+(\.\d+)?$", str(value).strip()))
                reason = None if valid else "Must be a number"
            elif key == "Table Category":
                valid = value in {"Hours", "Expenses"}
                reason = None if valid else "Category must be Hours or Expenses"
            else:
                valid = True
            reason = None
            entry[key] = {"value": value, "valid": valid}

            if not valid:
                entry[key]["reason"] = reason

        annotated.append(entry)
    return annotated

# add new values for inventory
def inventory_processing(obj_list):
    # Define sorting logic: Urban facilities go last, then sort by Due Date
    def sort_key(obj):
        facility_name = obj.get("Location", "")
        has_urban = "urban" in facility_name.lower()
        date_str = obj.get("Due Date", "")
        try:
            # sort by due date
            due_date = datetime.strptime(date_str, "%m/%d/%Y")
        except (ValueError, TypeError):
            due_date = datetime.max  # Invalid/missing dates go last
        return (has_urban, due_date)

    sorted_list = sorted(obj_list, key=sort_key)

    enriched = []
    for obj in sorted_list:
        # get the Location name from the ShipTo number
        shipToNum = obj.get("Ship To Number", None)
        locationName = ""
        if shipToNum is not None:
            try:
                # make sure it is an integer
                try:
                    shipToNum = int(shipToNum)
                except (ValueError, TypeError):
                    shipToNum = None
                matches = bp.shipto_df.loc[bp.shipto_df['shipto'] == shipToNum, 'Location']
                locationName = matches.iloc[0] if not matches.empty else ""
            except (ValueError, TypeError):
                locationName = ""

        # get the POBOX name from the POBOX number
        poboxNum = obj.get("PO BOX Number", None)
        try:
            poboxNum = int(poboxNum)
        except (ValueError, TypeError):
            poboxNum = None

        if poboxNum == 734671:
            poboxName = "Central-Dallas1"
        elif poboxNum == 734445:
            poboxName = "North-Chicago"
        elif poboxNum == 734672:
            poboxName = "South-Dallas2"
        elif poboxNum == 102289:
            poboxName = "West-Pasadena"
        else:
            poboxName = ""

        # Convert "Amount" to a float without any symbols
        amount = obj.get("Amount", "")
        new_amount = None
        if isinstance(amount, str):
            try:
                # Remove $ and commas, then convert to float
                cleaned = amount.replace("$", "").replace(",", "").strip()
                new_amount = float(cleaned)
            except ValueError:
                new_amount = None
        elif isinstance(amount, (int, float)):
            new_amount = amount
        else:
            new_amount = None

        new_obj = obj.copy()
        new_obj["Location"] = locationName
        new_obj["PO Box Name"] = poboxName
        new_obj["Category"] = "Medical inventory"
        new_obj['Amount'] = new_amount

        enriched.append(new_obj)

    return enriched


# add new values for provider
def provider_processing(obj_list):
    enriched = []
    for obj in obj_list:
        # Example logic: add fields based on existing ones
        # get the total pay
        pay = obj.get("Total Pay", None)
        category = obj.get("Table Category", None)

        # check data type of Total Pay
        if isinstance(pay, str):
            try:
                # Remove any non-digit characters except '.' and '-'
                cleaned = re.sub(r"[^\d\.\-]", "", pay)
                pay = float(cleaned)
            except ValueError:
                pay = None  # if conversion fails, treat as missing

        # YUBI: hard code days worked = 1
        if category == "Hours":
            days = 1
            if pay is not None:
                avg_cost_per_day = pay / days
            else:
                avg_cost_per_day = 0
        else:
            # for Expenses, do not include days or average cost per day
            days = None
            avg_cost_per_day = None

        new_obj = obj.copy() 
        new_obj["Working Days"] = days
        new_obj["Average Cost per Day"] = avg_cost_per_day

        enriched.append(new_obj)
    return enriched


# Updated code to handle PDF processing in parallel
@bp.route('/process_pdf', methods=['POST'])
async def process_pdf():
    # Retrieve the uploaded PDF file from the request
    uploaded_file = (await request.files)['file']
    if not uploaded_file:
        return jsonify({"error": "Missing file"}), 400

    # Retrieve the optional user message sent along with the PDF
    user_message = (await request.form).get('message', '')
    processing_mode = (await request.form).get('processing_mode', 'billing')

    # Attempt to read and open the PDF using PyMuPDF
    try:
        pdf_data = uploaded_file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        NUM_PAGES = len(doc)

        # remove code to extract only first page
        '''
        # only keep first page for inventory mode 
        if processing_mode == "inventory":
            first_page_doc = fitz.open()
            first_page_doc.insert_pdf(doc, from_page=0, to_page=0)
            doc.close()
            doc = first_page_doc
        '''
        
    except Exception as e:
        # Return 500 error if PDF cannot be opened
        return jsonify({"error": f"Failed to open PDF: {str(e)}"}), 500

    exclusion_keywords = []
    if processing_mode == "payment":
        batch_size=1
        dpi_threshold = 200
        MAX_CONCURRENT_BATCHES = 2
        exclusion_keywords = ["U.S. Postage Paid"]
    elif processing_mode == "inventory":
        batch_size = 1
        dpi_threshold = 200 
        MAX_CONCURRENT_BATCHES = 2
        # YUBI: no exclusions for now
        exclusion_keywords = []
    elif processing_mode == "provider":
        batch_size = 1
        dpi_threshold = 200
        MAX_CONCURRENT_BATCHES = 2
        # YUBI: no exclusions for now
        exclusion_keywords = []
    else:
        # default processing mode is billing
        batch_size = 2
        dpi_threshold = 100
        MAX_CONCURRENT_BATCHES = 4
        exclusion_keywords = ["Consent Form", "Schedule Report", "Anesthesia Record", "Consent for Anesthesia Services", "Referral Details", "Discharge Instructions", "Medication Reconciliation Form", "EGD Report"]

    # filter for meaningful pages only in billing and payment PDF files
    # Determine which pages to include
    # YUBI: DEBUGGING only to see if we are excluding important pages
    if processing_mode == "inventory" or processing_mode == "provider" or processing_mode == "billing":
        # For inventory mode, skip filtering and include all pages
        pages_include = list(range(len(doc)))
    else:
        # YUBI: EDIT so that we can skip the provider mode as well, right now it is being included here
        try:
            pages_include = await filter_pdf_dual_path(
                doc,
                inclusion_keywords=None,
                exclusion_keywords=exclusion_keywords,
                max_images=8,
                ocr_threshold=100,
                case_sensitive=False,
                dpi_threshold=dpi_threshold
            )
        except RuntimeError as e:
            return jsonify({"error with filtering pdf": str(e)}), 504
        except Exception as e:
            return jsonify({"error": f"filtering pdf failed: {str(e)}"}), 500

        if not pages_include:
            return jsonify({"error": "No valid pages found for processing."}), 400

    num_pages = len(pages_include)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)  # Controls concurrency limit

    # Helper function to stack multiple images vertically into one tall image
    def stack_images_vertically(images):
        widths, heights = zip(*(img.size for img in images))
        total_height = sum(heights)
        max_width = max(widths)
        combined = Image.new('RGB', (max_width, total_height), (255, 255, 255))  # White background
        y_offset = 0
        for img in images:
            combined.paste(ImageOps.expand(img, border=0, fill='white'), (0, y_offset))
            y_offset += img.height
        return combined

    # Async function to process a batch of pages:
    # - converts pages to images,
    # - stacks images vertically if multiple,
    # - encodes image as base64,
    # - calls AI model with the image and user message.

    # YUBI: from typing import List
    async def process_page_batch(page_indices: List[int]):
        async with semaphore:  # Acquire semaphore before starting to limit concurrency
            images = []
            # Loop through pages in the batch
            # for page_idx in range(start_idx, min(start_idx + batch_size, num_pages)):
            for page_idx in page_indices:
                # Extract one page as a separate PDF document
                subdoc = fitz.open()
                subdoc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
                page = subdoc[0]
                # Convert PDF page to PIL image asynchronously

                # adjust dpi based on processing mode
                if processing_mode == "payment":
                    dpi_threshold = 200  # Higher DPI for payment processing
                elif processing_mode == "inventory":
                    dpi_threshold = 200
                elif processing_mode == "provider":
                    dpi_threshold = 200
                else:
                    dpi_threshold = 100  # Default DPI for billing processing

                try:
                    pil_image = await convert_pdf_page_to_image(page, dpi_threshold)
                except Exception as e:
                    return None  # Skip this page if conversion fails
                    # return jsonify({"error": f"Failed to convert page {page_idx} to image: {e}"})
                
                images.append(pil_image)

            if not images:
                return None  # No pages to process in this batch

            # Stack images vertically if multiple pages, else use single image
            merged_image = images[0] if len(images) == 1 else stack_images_vertically(images)
            
            # Convert merged image to base64 string for model input
            try:
                img_base64 = await image_to_base64(merged_image)
            except RuntimeError as e:
                return jsonify({"error with image to base64 conversion": str(e)}), 504
            except Exception as e:
                # Return 500 for any other errors during batch processing
                return jsonify({"error": f"image to base64 conversion failed: {str(e)}"}), 500

            try:
                # Call the AI model with a timeout to avoid hanging
                # EDIT HERE: enable parameters to be passed to this function
                if processing_mode == "payment":
                    page_array, objects_array = await asyncio.wait_for(call_model_on_image(img_base64, user_message, processing_mode, NUM_PAGES), timeout=150)
                    return page_array, objects_array                    
                else:
                    # this model call is for inventory, provider, and billing processing modes
                    result = await asyncio.wait_for(call_model_on_image(img_base64, user_message, processing_mode), timeout=150)
                    return result
            # continue processing more pages if one page fails or times out
            except asyncio.TimeoutError:
                # I won't be able to see these print statements
                print(f"[TIMEOUT] Skipping pages: {page_indices}")
                return None
            except Exception as e:
                print(f"[ERROR] Failed processing pages {page_indices}: {e}")
                return None
            

    # Create async tasks for each batch of pages
    # This worked before, but it wasn't using pages_include
    # tasks = [asyncio.create_task(process_page_batch(i)) for i in range(0, num_pages, batch_size)]

    # Create page index batches based on pages_include
    # YUBI: test this
    batches = [pages_include[i:i+batch_size] for i in range(0, len(pages_include), batch_size)]
    tasks = [asyncio.create_task(process_page_batch(batch)) for batch in batches]


    partial_answers = []
    # Run all batch tasks concurrently (limited by semaphore)
    try:
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out any exceptions and collect valid results only
        valid_results = []
        for r in batch_results:
            if isinstance(r, Exception):
                print(f"[BATCH ERROR] Skipping batch due to error: {r}")
                continue
            valid_results.append(r)


        if processing_mode == "payment":

            partial_pages = [r[0] for r in valid_results if r is not None]
            partial_objects = [obj for r in valid_results if r is not None for obj in r[1]]
        else:
            # Filter out any None results (empty batches)
            # works for all other modes
            partial_answers = [r for r in valid_results if r is not None]

    except RuntimeError as e:
        # Return 504 Gateway Timeout if any batch timed out
        return jsonify({"error": str(e)}), 504
    except Exception as e:
        # Return 500 for any other errors during batch processing
        return jsonify({"error": f"Batch processing failed: {str(e)}"}), 500

    # After all batches processed, aggregate partial answers into a final answer
    # EDIT HERE: enable parameters to be passed to this function
    if processing_mode == "payment":
        # have a different summarizing process for payment processing mode using partial_pages and partial_objects
        try:
            all_pages = await summarize_pages(partial_pages)
        except Exception as e:
            return jsonify({"error": f"Failed during summarization of pages: {str(e)}"}), 500
        
        try:
            all_matches = await summarize_matches(partial_objects)
        except Exception as e:
            return jsonify({"error": f"Failed during summarization of pages: {str(e)}"}), 500
        
        matches_json = all_matches

    elif processing_mode == "billing":  
        try:
            summarized_answer = await summarize_answers(partial_answers, processing_mode)
        except Exception as e:
            return jsonify({"error": f"Failed during summarization of partial answers: {str(e)}"}), 500
    
        try:
            final_answer = await connect_summaries(summarized_answer, processing_mode)
        except Exception as e:
            return jsonify({"error": f"Failed during summarization of summarized answers: {str(e)}"}), 500

        # Parse the final aggregated model output as JSON
        try:
            answer_json = json.loads(final_answer)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Failed to parse model output as JSON: {str(e)}", "raw_output": final_answer}), 500
    elif processing_mode == "inventory":
        # Parse the final aggregated model output as JSON
        try:
            
            # we are concatenating all of the string together in partial answers bc it is a list of strings
            parsed_lists = [json.loads(part) for part in partial_answers]
            combined_output = [item for sublist in parsed_lists for item in sublist]  # flatten list of lists

            answer_json = combined_output # already a list of dictionaries so I don't lead to load json again
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Failed to parse model output as JSON: {str(e)}", "raw_output": partial_answers}), 500
        
        # manually adding the remaining fields to every inventory invoice object
        try:
            complete_answer = inventory_processing(answer_json)
        except Exception as e:
            return jsonify({"error": f"Inventory enrichment failed: {str(e)}"}), 500
    else:
        # provider processing mode is here
        try:
            summarized_answer = await summarize_answers(partial_answers, processing_mode)
        except Exception as e:
            return jsonify({"error": f"Failed during summarization of partial answers: {str(e)}"}), 500

        try:
            # summarized answer is already a list of dicts
            answer_json = summarized_answer
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Failed to parse model output as JSON: {str(e)}", "raw_output": summarized_answer}), 500
        
        # manually adding the remaining fields to every provider invoice object
        try:
            complete_answer = provider_processing(answer_json)
        except Exception as e:
            return jsonify({"error": f"Provider enrichment failed: {str(e)}"}), 500

    # Validate the parsed data and annotate invalid fields

    if processing_mode == "payment":
        try:
            # YUBI: create a new function to validate payment fields
            annotated_payments = validate_payment_fields(matches_json)
        except Exception as e:
            current_app.logger.error("Validation failed: %s", e)
            return {"error": "Validation error", "details": str(e)}, 500

        # YUBI: EDIT json to ensure that the key is "payments" instead of "patients"
        # return two results to front end, one is annotated_payments and the other is all_pages formatted into a string
        # YUBI: does this work? what does 200 mean?
        return jsonify({"payments": annotated_payments, "pages": all_pages}), 200
    elif processing_mode == "inventory":
        try:
            annotated_inventory = validate_inventory_fields(complete_answer)
        except Exception as e:
            current_app.logger.error("Validation failed: %s", e)
            return {"error": "Validation error", "details": str(e)}, 500

        # Return the validated and annotated patient data as JSON response
        return jsonify({"inventory": annotated_inventory})
    elif processing_mode == "provider":
        try:
            annotated_provider = validate_provider_fields(complete_answer)
        except Exception as e:
            current_app.logger.error("Validation failed: %s", e)
            return {"error": "Validation error", "details": str(e)}, 500

        # Return the validated and annotated patient data as JSON response
        return jsonify({"provider": annotated_provider})
    else:
        # default processing mode is billing
        try:
            annotated_patients = validate_patient_fields(answer_json)
        except Exception as e:
            current_app.logger.error("Validation failed: %s", e)
            return {"error": "Validation error", "details": str(e)}, 500

        # Return the validated and annotated patient data as JSON response
        return jsonify({"patients": annotated_patients})


# New route for follow-up
@bp.route("/followup", methods=["POST"])
async def followup():
    try:
        form = await request.form
        message = form["message"]
        # YUBI: Edit this function based on processing mode as well
        # change from previous_patients to previous_answer so that it is more general
        previous_answer_raw = form.get("previous_answer")


        # Parse the JSON string into an object
        try:
            previous_answer_json = json.loads(previous_answer_raw)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format for previous_patients"}), 400

        # Pretty-print the JSON for readability
        previous_answer_pretty = json.dumps(previous_answer_json, indent=2)

        # Build model message with context
        followup_prompt = (
            "The user has previously asked you to extract information from a scanned document. "
            "They now have a follow-up question. Below is the structured data from your previous response, "
            "and the user's follow-up question. Use this context to answer clearly and directly.\n\n"
            f"Previous extracted data:\n{previous_answer_pretty}\n\n"
            f"Follow-up question:\n{message}"
        )

        # Call model
        model_response = await call_model_followup(followup_prompt)

        return jsonify({"answer": model_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# save results to PostgreSQL database
@bp.route("/save_results", methods=["POST"])
async def save_results():

    # check to see if there is a connection
    try:
        if bp.db_conn is None:
            return jsonify({"error": "Database connection not initialized"}), 500
    except Exception as e:
        return jsonify({"error": f"DB connection error: {str(e)}"}), 500
    
    try:
        # Get JSON data
        form = await request.form
        files = (await request.files).getlist("files")

        # Required fields from form
        title = form.get("title")
        mode = form.get("mode")

        # table_data = form.get("rows")
        table_data = json.loads(form.get("rows"))  # Converts back to Python list of dicts by parsing JSON


        if not (title and mode and table_data):
            return jsonify({"error": "Missing title, mode, or table data"}), 400

        if not files:
            return jsonify({"error": "No PDF files uploaded"}), 400

        if len(files) > 20:
            return jsonify({"error": "You may upload up to 20 PDF files"}), 400

        # Compose record object

        # current date and time in ET
        curr_timezone = pytz.timezone('America/New_York')
        now = datetime.now(curr_timezone)

        # timestamp in ISO format (datetime object) for DB
        timestamp = now

        # human-readable title
        title = now.strftime("%Y-%m-%d %H:%M:%S %Z")


        table_data_json = json.dumps(table_data)

        # Insert results into database
        result_id = str(uuid.uuid1()) # create random unique ID based on timestamp and MAC address
        async with bp.db_conn.acquire() as conn:
            await conn.execute("""
                INSERT INTO result_objects (result_id, timestamp, title, mode, table_data)
                VALUES ($1, $2, $3, $4, $5)
            """, result_id, timestamp, title, mode, table_data_json)

            for file in files:
                file_id = str(uuid.uuid1())
                content = file.read() # returns file in bytes
                await conn.execute("""
                    INSERT INTO result_files (file_id, result_id, filename, content)
                    VALUES ($1, $2, $3, $4)
                """, file_id, result_id, file.filename, content)

            # file.filename is the name of the file when I first uploaded it. 
            # TODO: I can change it to what Michelle wanted by creating a variable before this line

        return jsonify({"message": "Results saved successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@bp.route("/get_result_by_timestamp", methods=["GET"])
async def get_result_by_timestamp():
    try:
        raw_ts = request.args.get("timestamp")
        if not raw_ts:
            return jsonify({"error": "Missing timestamp parameter"}), 400

        # Parse input timestamp
        try:
            input_dt = datetime.fromisoformat(raw_ts)
        except ValueError:
            return jsonify({"error": "Invalid timestamp format. Use ISO 8601."}), 400

        # If timestamp is naive (no tz), assume ET and convert to UTC
        if input_dt.tzinfo is None:
            eastern = pytz.timezone("America/New_York")
            input_dt = eastern.localize(input_dt).astimezone(pytz.UTC)
        else:
            input_dt = input_dt.astimezone(pytz.UTC)

        # Construct fuzzy window for match
        lower_bound = input_dt - timedelta(seconds=5)
        upper_bound = input_dt + timedelta(seconds=5)

        # This query takes advantage of the timestamp index
        # query from result_objects to get results table and metadata
        async with bp.db_conn.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT result_id, title, mode, table_data, timestamp
                FROM result_objects
                WHERE timestamp >= $1 AND timestamp <= $2
                ORDER BY timestamp
                LIMIT 1
            """, lower_bound, upper_bound)

            if not result:
                return jsonify({"error": "No result found near the provided timestamp"}), 404

            result_id = result["result_id"]

            files = await conn.fetch("""
                SELECT filename, content
                FROM result_files
                WHERE result_id = $1
                ORDER BY filename
            """, result_id)

        # Extract base64 content and filenames
        pdf_files_base64 = []
        pdf_filenames = []
        for row in files:
            if row["content"]:
                pdf_files_base64.append(base64.b64encode(row["content"]).decode("utf-8"))
                pdf_filenames.append(row["filename"])

        return jsonify({
            "title": result["title"],
            "mode": result["mode"],
            "table_data": json.loads(result["table_data"]),
            "pdf_files": pdf_files_base64,
            "pdf_filenames": pdf_filenames,  # <- add filenames here
            "matched_timestamp": result["timestamp"].isoformat()
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# list all of the potential timestamps of past results
@bp.route("/list_result_timestamps", methods=["GET"])
async def list_result_timestamps():
    try:
        # return the 20 most recent timestamps
        async with bp.db_conn.acquire() as conn:
            rows = await conn.fetch("""
                SELECT timestamp, title
                FROM result_objects
                ORDER BY timestamp DESC
                LIMIT 20
            """)

        results = [
            {
                "timestamp": row["timestamp"].isoformat(),
                "title": row["title"]
            }
            for row in rows
        ]
        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

