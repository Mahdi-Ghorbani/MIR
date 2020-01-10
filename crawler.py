from selenium import webdriver
import requests
import os
from queue import Queue
import json


def get_info(url):
    print(url)
    data = {}
    new_urls = []
    id = os.path.basename(url)
    authors = []
    references = []

    page = requests.get(url)
    t = page.text

    title_s = 'paper-detail-title'
    title_e = '</h1>'
    try:
        title = t[t.index(title_s) + len(title_s) + 2: t.index(title_e)]
    except ValueError:
        pass
        print(t.index(title_s))
        print(t.index(title_e))
        print(t)

    abstract_s = 'name="description" content="'
    abstract_e = '<meta name="twitter:description"'
    abstract = t[t.index(abstract_s) + len(abstract_s): t.index(abstract_e) - 7]

    date_s = '<span data-selenium-selector="paper-year"><span class=""><span>'
    date_e = '</span></span></span></li>'
    date = t[t.index(date_s) + len(date_s): t.index(date_e)]

    authors_s = '<meta name="citation_author" content="'
    authors_e = '<meta name="citation_'
    index_s = -1
    n = 1
    while True:
        index_s = t.find(authors_s, index_s + 1)
        index_e = t.find(authors_e, index_s + 1 + len(authors_s))
        if index_s == -1:
            break
        n += 1
        authors.append(t[index_s + len(authors_s): index_e - 4])

    references_s = 'data-heap-citation-type="citedPapers" data-heap-has-intents="true">' \
                   '<a data-selenium-selector="title-link" href="'
    references_e = '"><span class=""><span>'
    n = 1
    index_s = -1
    while True:
        index_s = t.find(references_s, index_s + 1)
        if index_s == -1:
            break
        index_e = t[index_s + len(references_s):].index(references_e)
        ref_url = "https://www.semanticscholar.org" + t[index_s + len(references_s): index_s + len(
            references_s) + index_e]
        ref_id = os.path.basename(ref_url)
        if n <= 5:
            new_urls.append(ref_url)
        references.append(ref_id)
        n += 1
    # print(t)

    data["id"] = id
    data["title"] = title
    data["abstract"] = abstract
    data["date"] = date
    data["authors"] = authors
    data["references"] = references

    return new_urls, data


def crawl(num):
    urls = Queue()
    urls.put("https://www.semanticscholar.org/paper/Tackling-Climate-Change-with-Machine-Learning-Rolnick-Donti/"
             "998039a4876edc440e0cabb0bc42239b0eb29644")
    urls.put("https://www.semanticscholar.org/paper/Sublinear-Algorithms-for-(%CE%94%2B-1)-Vertex-Coloring-Assadi-Chen/"
             "eb4e84b8a65a21efa904b6c30ed9555278077dd3")
    urls.put("https://www.semanticscholar.org/paper/Processing-Data-Where-It-Makes-Sense%3A-Enabling-Mutlu-Ghose/"
             "4f17bd15a6f86730ac2207167ccf36ec9e6c2391")

    all_urls = [
        "https://www.semanticscholar.org/paper/Tackling-Climate-Change-with-Machine-Learning-Rolnick-Donti/"
        "998039a4876edc440e0cabb0bc42239b0eb29644",
        "https://www.semanticscholar.org/paper/Sublinear-Algorithms-for-(%CE%94%2B-1)-Vertex-Coloring-Assadi-Chen/"
        "eb4e84b8a65a21efa904b6c30ed9555278077dd3",
        "https://www.semanticscholar.org/paper/Processing-Data-Where-It-Makes-Sense%3A-Enabling-Mutlu-Ghose/"
        "4f17bd15a6f86730ac2207167ccf36ec9e6c2391"
    ]
    cnt = 0
    res = {"papers": []}
    while cnt < num and urls.qsize() > 0:
        new_urls, data = get_info(urls.get())
        for url in new_urls:
            if url not in all_urls:
                all_urls.append(url)
                urls.put(url)
        if data not in res["papers"]:
            res["papers"].append(data)
            cnt += 1

        print(cnt)
        print(len(res["papers"]))
        print("-------")

    with open('papers_info.json', 'w') as outfile:
        json.dump(res, outfile)
        outfile.close()


if __name__ == '__main__':
    crawl(5000)
