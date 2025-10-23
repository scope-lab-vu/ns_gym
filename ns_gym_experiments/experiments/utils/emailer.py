import os
from os import getenv
from dotenv import load_dotenv
from pathlib import Path
import argparse

load_dotenv(".env")
# Import smtplib for the actual sending function.
import smtplib
from datetime import datetime

# Here are the email package modules we'll need.
from email.message import EmailMessage
import re
from config import *


def main(log_type):
    # Create the container email message.
    msg = EmailMessage()
    msg["Subject"] = "APC Pipeline finished - VM Backend"
    me = os.getenv("EMAIL")
    recipients = [WEGO_MAL]
    msg["From"] = me
    msg["To"] = ", ".join(recipients)
    msg.preamble = "You will not see this in a MIME-aware mail reader.\n"

    # Open the plain text file whose name is in textfile for reading.
    with open(f"{LIVE_BUCKET}/APC_Wego/wego-site/{DATES_JSON}") as fp:
        # Create a text/plain message
        msg.set_content(fp.read())

    # Open the files in binary mode.  You can also omit the subtype
    # if you want MIMEImage to guess it.

    now = datetime.now()
    log_name = now.strftime("%Y-%m-%d")

    files = [f"{LOG_DIR}/{file}" for file in os.listdir(LOG_DIR) if (file.lower().endswith(".gz"))]
    files.sort(key=os.path.getmtime)
    logfiles = [files[-1]]
    for file in logfiles:
        filename = os.path.basename(os.path.normpath(file))
        with open(file, "rb") as fp:
            img_data = fp.read()
        msg.add_attachment(img_data, maintype="application", subtype="tar+gzip", filename=filename)

    # Send the email via our own SMTP server.
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        EMAIL_PASSWORD = getenv("EMAIL_PASSWORD")
        s.login(WEGO_MAIL, EMAIL_PASSWORD)
        s.send_message(msg)
        s.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description here")

    # Define command-line arguments
    parser.add_argument("-t", "--log_type", help="Type of log to send", default="PROCESS_SCRIPT")

    args = parser.parse_args()

    main(args.log_type)
