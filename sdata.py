#!/usr/bin/env python3
import requests

from bs4 import BeautifulSoup
from functools import reduce

project_count = 0

def getPageHTMLString(page_num):
    url = "https://devpost.com/software/trending?page={}".format(page_num)
    r = requests.get(url)
    return r.text

def savePageDescriptions(page_html):
    soup = BeautifulSoup(page_html, "html.parser")
    atags = soup.find_all("a", "link-to-software")

    for atag in atags:
        url = atag["href"]
        #print(atag)
        winner = False
        if atag.find("img", "winner"):
            winner = True
        #print("winner =", winner, "url =", url)
        #continue
        saveDescriptionForProject(winner, url)

def saveDescriptionForProject(winner, project_url):
    global project_count

    folder = "winners" if winner else "losers"

    #print("project =", project_url, " | folder =", folder)
    #return

    r = requests.get(project_url)
    project_html = BeautifulSoup(r.text, 'html.parser')
    desc = project_html.find(id="app-details-left")
    paragraphs = desc.find_all('p')
    text = "\n".join(map(lambda p: p.string, filter(lambda p: p.string, paragraphs)))
    #text = reduce(lambda x, y: x + "\n" + y, map(lambda p: p.string, paragraphs[1:]))
    with open("{}/{}.txt".format(folder, project_count), "w") as f:
        f.write(text)
    project_count += 1

def main():
    for i in range(1500, 2000):
        page_html = getPageHTMLString(i)
        savePageDescriptions(page_html)
        print("Completed parsing of page {}".format(i))

if __name__ == '__main__':
    main()