import smtplib
import smtplib, ssl
import configparser
import src.config_log as cf
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

config_Url = configparser.ConfigParser()
config_Url.read('config/config.ini')
sender_mail = config_Url["Mail Related"]["sender_mail"] 
sender_password = config_Url["Mail Related"]["sender_password"] 
path = config_Url["Mail Related"]["path"] 

smtp_server = 'smtp-mail.outlook.com'
cf.success_log(200, "Smtp Server Connected", "smtp_server", path)
smtp_port = 587
gmail = sender_mail
password = sender_password
secure = False



def send_mail(employeeName, to_email, cc_email,timesheetPeriod,projectDescription,projectTaskDescription,timsheetEffort, attachment_path=None):
    try:
        df_data = {'Timesheet Period': timesheetPeriod, 'Project Description': projectDescription, 'Project Task Description': projectTaskDescription, 'Timesheet Efforts': timsheetEffort }
        # Generate the HTML table
        table = '<table border="1" width="80%"><tr><th>Category</th><th>Value</th></tr>'

        for key, value in df_data.items():
            table += f'<tr><td>{key}</td><td>{value}</td></tr>'

        table += '</table>'

    except Exception as e:
        print("Error",e)

    try:
        message = MIMEMultipart()
        message['From'] = 'Contact <{sender}>'.format(sender = gmail)
        message['To'] = ", ".join(to_email)
        message['Cc'] = ", ".join(cc_email) if cc_email else ''
        print(type(timsheetEffort))
        message['Subject'] =  projectDescription + '-' + timsheetEffort
        cf.success_log(200, "Generating Body", "smtp_server", path)
        body = f'''

                Dear {employeeName},<br><br>
                
                <p>Your recent timesheet submission requires additional information to help us ascertain R&D hours for your activity. For Canada it would need to be SR&ED and the rest of the Globe it would be R&D</p>
                <p>To help us meet SR&ED tax requirements, please reply to this email and share:</p>
                <ul>
                <li>Brief description elaborating on the specific technical aspects of the activity</li>
                <li>Any uncertainties or challenges encountered. Explain how problem-solving, experimentation, or research was necessary to address the identified issues.</li>
                </ul>
                
                <b>Information Required For</b><br><br>
                {table}<br><br>

                <b>Please copy the table below and reply</b><br><br>
                <table border="1" width="80%">
                <tr><th>Type of Work</th><th>Allocate Hours</th><th>Description of work done</th></tr>
                <tr><td>R&D</td><td></td><td></td></tr>
                <tr><td>Routine</td><td></td><td></td></tr>
                </table>
                <br><br>
                Thank you for your attention to this matter.<br><br>

                Regards,<br>
                System Administrator<br><br>
                '''

        message.attach(MIMEText(body, 'html'))
        cf.success_log(200, "Attach to MIMEText", "smtp_server", path)
        try:
            context = ssl.create_default_context()
            cf.success_log(200, "Create the context using ss.create_default_context", "smtp_server", path)
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(gmail, password)
                cf.success_log(200, "Login to server", "smtp_server", path)
                server.sendmail(gmail, to_email + (cc_email if cc_email else []),
                            message.as_string())
                cf.success_log(200, "Sending Mail", "smtp_server", path)
                server.quit()
            return True
        except Exception as e:
            print("Error",e)
            return False
    except Exception as e:
        print(e)

# dt = datetime.now()
# violation_id = "202302071367"
# date_time = dt
# location = "BG-2 Front"
# violation_type = "no_helmet"
# to = ["mukesh.kumar@algo8.ai"]
# cc = ["mukesh.kumar@algo8.ai","bramhesh.srivastav@algo8.ai","pramod.jangid@algo8.ai","shivansh.kumar@algo8.ai"] #"yadavak3@indianoil.in"]

# result = send_mail('Mukesh kumar', to, cc,'April 1st - April 30th', 'Rogers XYZ', 'Attendance','18 hours')
# print(result)