from selenium import webdriver
from time import sleep
from selenium.webdriver.common.keys import Keys
from parsel import Selector

driver = webdriver.Chrome()
driver.get('https://sanskritdocuments.org/')
sleep(0.5)

file = open('data_preparation/synthetic/input_links.txt','r')
for line in file.readlines():
    line = str(line)
    driver.get(line)
    sleep(0.5)
    
    xpaths = driver.find_elements_by_xpath('//ul[@style="list-style-type:none"]/li/a[3]')
    Links = [xpath.get_attribute("href") for xpath in xpaths ]
    try:
        for Link in Links:
            
            driver.get(Link)
            sleep(0.5)
            article_name =driver.find_element_by_xpath('//body/pre[@itemprop="text"]/h2').get_attribute('innerHTML')
            inner_text =driver.find_element_by_xpath('//div[@id="article"]/pre').get_attribute('innerHTML')
            out_file = open('data_preparation/synthetic/sanskritdoc.txt','w')
            out_file.write(inner_text)
            sleep(0.5)
    except:
        pass
driver.quit()

out_file.close()


with open('data_preparation/synthetic/sanskritdoc.txt', 'r') as f:
    lines = f.read()
y = list(',.0123456789-_|')
CHARMAP = [chr(i) for i in range(2304,2432)] + [chr(i) for i in range(65,90)] + y

f = open('data_preparation/synthetic/sanskritdoc.txt', 'w')
""" Removing whitespace,empty lines and replacing english character with #"""
for line in lines:
    if len(line.strip())!=0:
        annot_text = line.strip()
        annot_text = annot_text.upper()
        annot_text = ''.join([c for c in annot_text if c in CHARMAP])
        for c in annot_text:
            if ord(c)>=65 and ord(c)<=90:
                annot_text = annot_text.replace(c, '#')
        f.write(annot_text)
        f.write('\n')
f.close()







