import html2text
from bs4 import BeautifulSoup
import re

'''
function cleans body by removing signature and previous mail content in the body and in the end remove 
html tags from giving us final string.
'''
arr_from = ["**from:**", "**von:**","**de:**","**de:**","**de :**","**de:**"]
arr_sent = ["**sent:**", "**gesendet:**","**enviada:**","**enviada el:**","**envoyé :**","**enviado el:**"]
arr_to = ["**to:**", "**an:**","**para:**","**Para:**","**à :**","**Para:**"]
arr_date = ["**date:**","asd","asd","asd","asd","asd"]

# **de:** anand pratap singh <anandpratapsingh@kanplas.com>  
# **enviado el:** jueves, 26 de mayo de 2022 6:42  
#  **para:** clara vergés - satucesa europa <clara@satucesa.es>  
#  **cc:** sa
# count = 0
def clean_body_mail(body_email):
    count = 0
    try:
        count+=1
        soup = BeautifulSoup(body_email, 'html.parser')
        body_email_store = str(soup)
        # print(body_email_store)
        array_test = []
        # find all div with signature init
        # div_bs4 = soup.find_all('div', class_ = re.compile('signature$'))
        div_bs4 = soup.find_all('div', {"class":re.compile('signature$', re.IGNORECASE)})
        if len(div_bs4) == 0:
            div_bs4 = soup.find_all('div', {"id":re.compile('signature$', re.IGNORECASE)})
        print("------number of signature---------->"+str(len(div_bs4)))
        # 
        # print(re.search("signature$", body_email))
        for div in div_bs4: # check if sequence is maintaiend
            # print(str(div))
            array_test.append((str(div),""))
        for k, v in array_test:
            body_email_store = body_email_store.replace(k, v)
        print("signature removed")

        # removing div appendonsend so that we can get clean mails
        soup2 = BeautifulSoup(body_email_store, 'html.parser')
        body_email_store = str(soup2)
        array_test2 = []
        div_bs4_2 = soup2.find_all('div', id="appendonsend")
        # print(len(div_bs4_2))
        if len(div_bs4_2) == 0:
            print( " no reply found")
        else:
            num = body_email_store.find(str(div_bs4_2[0]))
            body_email_store =  body_email_store[0:num]
            
        body_email_store = html2text.html2text(body_email_store)
        clean_body = body_email_store.lower()
        # new flow where i check exitance of from, to and cc and if combination is found we remove the text below it
        arr_split_n = clean_body.split("\n")
        try:
            list_num_exist = []
            for ich, val in enumerate(arr_split_n):
                for k in arr_from:
                    try:
                        temp = val.index(k)
                        list_num_exist.append(ich)
                    except:
                        pass
            min_num_exist = min(list_num_exist)
            for k in range(len(arr_sent)):
                if ((arr_sent[k] in arr_split_n[min_num_exist+1] or arr_sent[k] in arr_split_n[min_num_exist+2]) or (arr_date[k] in arr_split_n[min_num_exist+1] or arr_date[k] in arr_split_n[min_num_exist+2])) and (arr_to[k] in arr_split_n[min_num_exist+2] or arr_to[k] in arr_split_n[min_num_exist+3]):
                    index_del = clean_body.index(arr_split_n[min_num_exist])
                    print(index_del)
                    clean_body = clean_body[:index_del]
                    break
                    # print(clean_body)
                else:
                    pass
            # print("in try")
        except: 
            clean_body = body_email_store
        clean_body = clean_body.lower()
        # clean_body = re.sub('[!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~\n\d\r]+?',' ',body_email_store)
        #  remove signature of specific client : "pema-verpackung.de"
        test_email_check_del = clean_body.find("mit freundlichem gruß aus syke / kind regards")
        if test_email_check_del != -1:
            clean_body = clean_body[:test_email_check_del]
        test_clean_again = clean_body.find("**confidential & privileged information**")
        if test_clean_again != -1:
            clean_body = clean_body[:test_clean_again]
        test_clean_again_2 = clean_body.find("**atlantic packaging**")
        if test_clean_again_2 != -1:
            clean_body = clean_body[:test_clean_again_2]
        print("body is now clean.")
        return(True,clean_body)
    except:
        body_email_store = html2text.html2text(body_email)
        clean_body = body_email_store.lower()
        return (False,clean_body)
