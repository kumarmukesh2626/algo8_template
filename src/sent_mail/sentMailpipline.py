import requests
import time
import requests
import json
from io import BytesIO
import json
import os
import pandas as pd
import time
import pandas as pd
from PIL import Image
from database.ConnectDb import DatabaseHandler
import sys
from src.sent_mail.sendMail import send_mail
# alert_url ='http://localhost:5005/alert'
from datetime import datetime

db = DatabaseHandler()
    
import requests

def certinaityAI(projectDescription,month=6):
    try:
        query1 = """ SELECT
            employeeId,
            employeeName,
            spocEmailId,
            spocName,
            projectId,
            timesheetDate,
            timesheetEfforts,
            projectDescription,
            projectTaskDescription,
            taskType
        FROM
            timesheetdata
        WHERE
            timesheetEfforts IS NOT NULL
            AND projectDescription IS NOT NULL
            AND projectTaskDescription IS NOT NULL
            AND taskType = 'Pending'
            AND (projectDescription LIKE '%{project_description}%')
            AND timesheetDate >= DATE_SUB(CURRENT_DATE, INTERVAL {months} MONTH)
            AND timesheetDate <= CURRENT_DATE;
        """.format(project_description=projectDescription, months=month)

        current_timestamp = datetime.now()
        data = db.fetch_data(query1)
        print(data)
        # data['timesheetDate'] = pd.to_datetime(data['timesheetDate'])
        # print(data['timesheetDate'])
        # Fill NULL (nan) values in 'spocName' with 'Unknown' or any other placeholder value
        data['spocName'].fillna('Unknown', inplace=True)

        # Convert 'timesheetDate' to datetime
        data['timesheetDate'] = pd.to_datetime(data['timesheetDate'],format='%y-%m-%d')

        # Define a custom aggregation function to keep the first projectDescription
        def first_project_description(x):
            return x.iloc[0]

        # Group data by 'spocName', 'projectDescription', and aggregate columns
        grouped_data = data.groupby(['spocName', 'projectDescription']).agg({
            'timesheetDate': [('min_date', 'min'), ('max_date', 'max')],
            'timesheetEfforts': 'sum',
            'projectTaskDescription': lambda x: ', '.join(x.unique()),
            'spocEmailId': [('spoc_email', 'first')],
        }).reset_index()
        grouped_data.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped_data.columns]

        # Merge duplicate rows based on projectDescription by summing timesheetEfforts
        grouped_data = grouped_data.groupby(['projectDescription', 'spocName']).agg({
            'timesheetDate_min_date': 'min',
            'timesheetDate_max_date': 'max',
            'timesheetEfforts_sum': 'sum',
            'projectTaskDescription_<lambda>': lambda x: ', '.join(x.unique()),
            'spocEmailId_spoc_email': 'first'
        }).reset_index()

        # # Reset the placeholder value back to NULL
        # grouped_data['spocName'] = grouped_data['spocName'].replace('Unknown', pd.NA)

        print(grouped_data)
        columns = ['Timestamp','spocName','spocMailid','projectDescription','projectTaskDescription','timeSheetEffort','Response','taskType','toRecipients','ccRecipients','SenderEmailID','Subject']
        

        excel_file_path = 'grouped_data_latest_per_month.xlsx'

        # Save the DataFrame to the Excel file
        grouped_data.to_excel(excel_file_path, index=False)

        # print(f'Data saved to {excel_file_path}')

        for index, row in grouped_data.iterrows():
            spoc_name= row[('spocName')]
            timesheet_start_date_min = row['timesheetDate_min_date']
            timesheet_end_date_max = row[('timesheetDate_max_date')]
            project_description = row[('projectDescription')]
            project_task = row[('projectTaskDescription_<lambda>')]
            timesheet_effort = row[('timesheetEfforts_sum')]
            spoc_email_id = row[('spocEmailId_spoc_email')]
            to_email_id = ['mukesh.kumar@algo8.ai']
            to_cc_id =  [] #[spoc_email_id]
            timesheet_effort = str(timesheet_effort)
            # Format the start and end dates as "Month Dayst - Month Dayth"
            timesheet_date_range = f"{timesheet_start_date_min.strftime('%B')} {timesheet_start_date_min.day}st - {timesheet_end_date_max.strftime('%B')} {timesheet_end_date_max.day}th"
            print(timesheet_date_range)
            values = [current_timestamp,spoc_name,to_email_id,project_description,project_task,timesheet_effort,'Null','Null','Null','Null',spoc_email_id,'Null']
            # payload = {
            #             'employeeName': spoc_name,
            #             'email': to_email_id,
            #             'cc_mail': to_cc_id,
            #             'timesheetPeriod': timesheet_date_range,
            #             'projectDescription': project_description,
            #             'projectTaskDescription': project_task,
            #             'timsheetEffort': timesheet_effort,
            #         }
            # # Only include 'projectTaskDescription' and 'timesheetEffort' if they are not null
            # if not pd.isna(project_task):
            #     payload['projectTaskDescription'] = project_task
            # if not pd.isna(timesheet_effort):
            #     payload['timsheetEffort'] = timesheet_effort

            # response = requests.post(alert_url, json=payload)
            # if response.status_code == 200:
            try:
                send_mail(spoc_name, to_email_id, to_cc_id, timesheet_date_range, project_description, project_task, timesheet_effort)
                print('Alert sent successfully',project_description)
                db.run(columns,values,'naggingDetails')
            except Exception as e:
                print(e)
               
        return True
    except Exception as e:
        print(e)
