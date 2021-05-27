import requests
from bs4 import BeautifulSoup
import os
import re


def url_join(comp):
    comp = [i.strip('/') for i in comp]
    return '/'.join(comp)


class Crawler:
    def __init__(self, url, saved_path, header=None):
        self.url = url
        self.header = header
        self.save_path = saved_path
        os.makedirs(saved_path, exist_ok=True)

    def parse(self, year_list, mode='train'):
        availble_conf = self._parse_menu(url_join([self.url, 'menu.py']))
        availble_conf = [i for i in availble_conf if i[1] in year_list]
        conf_urls = [url_join([self.url, i[2]['conference']]) for i in availble_conf]
        work_urls = [url_join([self.url, i[2]['workshop']]) for i in availble_conf]

        conf_save_path = os.path.join(self.save_path, mode, 'conference')
        os.makedirs(conf_save_path, exist_ok=True)
        for i in conf_urls:
            paper_list = self._parse_papers(i)
            for j in paper_list:
                self._save_file(j, conf_save_path)

        workshop_save_path = os.path.join(self.save_path, mode, 'workshop')
        os.makedirs(workshop_save_path, exist_ok=True)
        for page_urls in work_urls:
            work_shop_pages = self._parse_wokshop_page(page_urls)
            for i in work_shop_pages:
                paper_list = self._parse_papers(i)
                for j in paper_list:
                    self._save_file(j, workshop_save_path)

        return

    def _parse_menu(self, menu_link):
        r = requests.get(menu_link)
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'html.parser')
        x = soup.find(id='content').dl.find_all(name='dd')

        def get_single(soup_item):
            conf_name,pub_year = re.split('[ ,]', soup_item.text.strip())[:2]
            href_list = soup_item.find_all('a', href=True)
            url_dict = {'conference': href_list[0]['href'], 'workshop': href_list[1]['href']}
            return conf_name, int(pub_year), url_dict

        conf_list = [get_single(i) for i in x]
        return conf_list

    def _parse_papers(self, conf_link):
        r = requests.get(conf_link)
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'html.parser').find(id='content').find_all('a', href=True)
        paper_urls = []
        for x in soup:
            if x.text == 'pdf':
                paper_urls.append(url_join([self.url, x['href']]))
        return paper_urls

    def _save_file(self, url, save_path):
        pdf_name = url.split('/')[-1]
        pdf_save_path = os.path.join(save_path, pdf_name)
        try:
            if not os.path.exists(pdf_save_path):
                with open(pdf_save_path, 'wb') as f:
                    content = requests.get(url=url, stream=True)
                    for chunk in content.iter_content(256):
                        if chunk:
                            f.write(chunk)
                    print(f'Successfully download {pdf_name}')
            return 1
        except:
            print(f'Failed while parsing from {url}!')
            if os.path.exists(pdf_save_path):
                os.remove(pdf_save_path)
            return 0

    def _parse_wokshop_page(self, page_urls):
        r = requests.get(page_urls)
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'html.parser').find(id='content').dl.find_all('a', href=True)
        subpage_list = [i['href'] for i in soup if 'menu.py' not in i['href']]
        sub_url = '/'.join(page_urls.split('/')[:-1])
        subpage_list = [url_join([sub_url, i]) for i in subpage_list]
        return subpage_list


def _save_file(url, save_path):
    pdf_name = url.split('/')[-1]
    pdf_save_path = os.path.join(save_path, pdf_name)
    try:
        if not os.path.exists(pdf_save_path):
            with open(pdf_save_path, 'wb') as f:
                content = requests.get(url=url, stream=True)
                for chunk in content.iter_content(256):
                    if chunk:
                        f.write(chunk)
                print(f'Successfully download {pdf_name}')
        return 1
    except:
        print(f'Failed while parsing from {url}!')
        if os.path.exists(pdf_save_path):
            os.remove(pdf_save_path)
        return 0


# if __name__ == '__main__':
#     root = "https://openaccess.thecvf.com/"
#     conference_dir_path = "./CVPR/conference2014/"
#     conference_urls = [
#         "https://openaccess.thecvf.com/CVPR2014"
#     ]
#     for i in conference_urls:
#         r = requests.get(i)
#         r.encoding = 'utf-8'
#         soup = BeautifulSoup(r.text, 'html.parser')
#         paper_urls = []
#         for i in soup.find(id='content').find_all('a', href=True):
#             if i.text == 'pdf':
#                 # https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Dual_Super-Resolution_Learning_for_Semantic_Segmentation_CVPR_2020_paper.pdf
#                 #pdf_url = url_join([root, i['href']])
#                 pdf_url = root + i['href']
#                 paper_urls.append(pdf_url)
#                 print(pdf_url)
#         for j in paper_urls:
#             _save_file(j, conference_dir_path)


if __name__ == "__main__":
    root = "https://openaccess.thecvf.com/"
    workshop_path = "https://openaccess.thecvf.com/CVPR2014_workshops/menu"
    workshop_root = "https://openaccess.thecvf.com/CVPR2014_workshops/"
    workshop_dir_path = "./CVPR/workshop2014/"
    r = requests.get(workshop_path)
    r.encoding = 'utf-8'
    soup = BeautifulSoup(r.text, 'html.parser')
    workshop_urls = []
    for i in soup.find(id='content').find_all('a', href=True):
        pdf_url = workshop_root + i['href']
        workshop_urls.append(pdf_url)

    for workshop in workshop_urls:
        r = requests.get(workshop)
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'html.parser')
        paper_urls = []
        for i in soup.find(id='content').find_all('a', href=True):
            if i.text == 'pdf':
                # pdf_url = url_join([root, i['href']])
                pdf_url = root + i['href']
                paper_urls.append(pdf_url)
                print(pdf_url)
        for j in paper_urls:
            _save_file(j, workshop_dir_path)
