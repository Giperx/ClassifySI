import argparse
from datetime import datetime
import logging
import os
import requests
import time
import random

# Set up logging to file
logger = logging.getLogger('crawler_tmp')
logger.setLevel(logging.INFO)
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = f'logs/response_log_hd_{current_time}.log'
handler = logging.FileHandler(log_path)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

parser = argparse.ArgumentParser(description='Download HD images from Baidu Image Search')
parser.add_argument('-q', '--query', type=str, default='tmp', help='Search query')
parser.add_argument('-o', '--output', type=str, default='tmp', help='Output folder')
parser.add_argument('-n', '--number', type=int, default=20, help='Number of images to download')
args = parser.parse_args()

for arg in vars(args):
    logger.info(f"{arg}: {getattr(args, arg)}")
    print(f"{arg}: {getattr(args, arg)}")

search_query = args.query # 搜索关键词
# output_folder = args.output # 图片保存目录
output_folder = os.path.join('./raw_data/', args.output) # 图片保存目录
max_images = args.number # 最大下载图片数量
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

images_downloaded = 0
images_per_page = 30

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Linux; Android 11; Pixel 4 XL Build/RQ1A.201205.002) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Mobile Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:58.0) Gecko/20100101 Firefox/58.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36',
    # 更多 User-Agent...
]

save_time = []
start_all_time = time.time()
tmp_time = time.time()
while images_downloaded < max_images:
    offset = (images_downloaded // images_per_page) * images_per_page
    url = f'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=r&ct=201326592&cl=2&lm=-1&ie=utf-8&word={search_query}&pn={offset}&rn={images_per_page}'

    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "Connection": "keep-alive"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # 确保使用正确的编码
        response.encoding = response.apparent_encoding

        logger.info(f'Response Content for URL: {url}')
        # logger.info(response.text[:1000])
        logger.info('-' * 40)
    

        try:
            data = response.json()
            images = data.get('data', [])

            if not images:
                print("No images found in the response.")
                break

            for img in images:
                if images_downloaded >= max_images:
                    break

                # 确保 img 是字典类型
                if isinstance(img, dict):
                    # 打印 img 的内容以调试
                    print(f'Debug: Processing img: {img}')  # 调试输出
                    logger.info(f'Debug: Processing img fromTitleEnc: {img.get("fromPageTitleEnc")}')
                    # 使用高清图片链接（从 replaceUrl 列表中提取 'ObjURL'）
                    replace_url_list = img.get('replaceUrl', [])
                    if isinstance(replace_url_list, list) and replace_url_list:
                        img_url = replace_url_list[0].get('ObjURL')  # 从列表中获取第一个字典的 ObjURL
                    else:
                        print(f'Warning: replaceUrl is not a valid list for img: {img}')  # 调试输出
                        img_url = None  # 设为 None 以避免后续错误

                    if img_url and img_url.startswith('http'):
                        try:
                            down_start_time = time.time()
                            img_data = requests.get(img_url, headers=headers)
                            img_data.raise_for_status()

                            # 保存高清图片
                            with open(f'{output_folder}/{images_downloaded + 1}.jpg', 'wb') as handler:
                                handler.write(img_data.content)
                            images_downloaded += 1
                            print(f'Successfully downloaded HD image {images_downloaded}: {img_url}')
                            down_end_time = time.time()
                            print(f'Time taken to download image {images_downloaded}: {down_end_time - down_start_time:.2f} seconds')
                            logger.info(f'Time taken to download image {images_downloaded}: {down_end_time - down_start_time:.2f} seconds')
                            print(f'running time: {time.time() - tmp_time:.2f} seconds')
                            logger.info(f'running time: {time.time() - tmp_time:.2f} seconds')
                            tmp_time = time.time()
                            logger.info('-' * 80)
                            print('-' * 80)
                            time.sleep(random.uniform(3, 5))  # 增加请求间隔
                        except Exception as e:
                            print(f'Error downloading image {img_url}: {e}')
                else:
                    print('Warning: img is not a dictionary.')
        except requests.exceptions.RequestException as e:
            print(f'Error fetching data from Baidu: {e}')
        except Exception as e:
            print(f'An error occurred: {e}')

    except requests.exceptions.RequestException as e:
        print(f'Error fetching data from Baidu: {e}')
        break
end_all_time = time.time()
print(f'Total time taken: {end_all_time - start_all_time:.2f} seconds')
logger.info(f'Total time taken: {end_all_time - start_all_time:.2f} seconds')
print(f'Download complete! Total HD images downloaded: {images_downloaded}')
