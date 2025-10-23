import yagmail
from dotenv import load_dotenv
from datetime import datetime
import os

def send_mail(experiment_name, status, message):
    try:
        load_dotenv()
        user = os.getenv("EMAIL")
        # app_password = os.getenv("PASSWORD")
        to = os.getenv("EMAIL")  # Consider allowing this to be a parameter
        now = datetime.now() 
        log_name = now.strftime("%Y-%m-%d")
        subject = f"Experiment: {experiment_name}, Status: {status}, Date: {log_name}"
        contents = [message]
        try: 
            yag = yagmail.SMTP(user,os.getenv("PASS2"))
            yag.close()
        except Exception as e:
            print(f"Failed to login: {str(e)}")
            
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

# Example usage if __name__ == "__main__": send_mail("Test", "Success?", "This is a test email")

