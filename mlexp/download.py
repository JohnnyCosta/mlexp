# -*- coding: utf-8 -*-
""" dl.py

	A module to search google and download images corresponding 
	to search terms. From:

		https://github.com/hardikvasa/google-images-download
"""

from google_images_download import google_images_download


response = google_images_download.googleimagesdownload()

args = ['french identity card']
		
def run():
	for arg in args:
		absolute_image_paths = response.download({'keywords' : arg, 
												  'limit': 200,  # Requires `chromedriver` for more than 100 image scrapes.
												  # To download: https://sites.google.com/a/chromium.org/chromedriver/downloads (link live 8/30/18)
												   'chromedriver': 'C:/apps/chromedriver/chromedriver.exe',
                                                  'proxy' : 'fr-proxy.groupinfra.com:3128'})

if __name__ == '__main__':
	run()