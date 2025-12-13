The structure of the search results can be examined with your browser tools, as shown here:

![Screenshot of Google search results in a web browser, showing HTML elements and a CSS selector tool](page_184_320_1207_496.png)

Here, we see that the search results are structured as links whose parent element is a <h3> tag with class "r".

To scrape the search results, we will use a CSS selector, which was introduced in Chapter 2, Scraping the Data:

>>> from lxml.html import fromstring
>>> import requests
>>> html = requests.get('https://www.google.com/search?q=test')
>>> tree = fromstring(html.content)
>>> results = tree.cssselect('h3.r a')
>>> results
[<Element a at 0x7f3d9affeaf8>,
 <Element a at 0x7f3d9affe890>,
 <Element a at 0x7f3d9affe8e8>,
 <Element a at 0x7f3d9affeaa0>,
 <Element a at 0x7f3d9b1a9e68>,
 <Element a at 0x7f3d9b1a9c58>,
 <Element a at 0x7f3d9b1a9ec0>,
 <Element a at 0x7f3d9b1a9f18>,
 <Element a at 0x7f3d9b1a9f70>,
 <Element a at 0x7f3d9b1a9fc8>]